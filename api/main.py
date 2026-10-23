"""FastAPI application exposing CosyVoice2 / CosyVoice3 over HTTP.

Endpoint layout mirrors indextts25-fastapi (OpenAI speech compatible):

    GET    /health                  - liveness + model status (incl. VRAM)
    GET    /v1/models               - OpenAI-compatible model list
    GET    /v1/voices               - list voices (alias: /v1/audio/voices)
    POST   /v1/voices               - upload a reference audio as a voice
    DELETE /v1/voices/{voice_id}    - delete a voice
    POST   /v1/audio/speech         - synthesize speech (stream=true => progressive audio)
    POST   /v1/audio/speech/stream  - synthesize speech as base64 PCM16 SSE events

The default model is loaded in a background thread at startup, so ``/health``
answers immediately while the model is still loading.
"""

import asyncio
import base64
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Iterator, Optional, Tuple

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from .models import SpeechRequest
from .service import (
    MAX_VOICE_BYTES,
    CosyVoiceService,
    ModelNotFoundError,
    TTSNotReadyError,
    VoiceNotFoundError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("cosyvoice.api")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- configuration

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.abspath(os.environ.get("COSYVOICE_REPO") or os.path.join(PROJECT_DIR, "..", "CosyVoice"))


def _default_models(repo: str) -> dict:
    return {
        "cosyvoice2": os.path.join(repo, "pretrained_models", "CosyVoice2-0.5B"),
        "cosyvoice3": os.path.join(repo, "pretrained_models", "Fun-CosyVoice3-0.5B"),
    }


def _parse_models(value: str) -> dict:
    models = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, path = item.partition("=")
        models[key.strip()] = os.path.abspath(path.strip() or item.strip())
    return models


_models_env = os.environ.get("COSYVOICE_MODELS", "").strip()
MODEL_DIRS = _parse_models(_models_env) if _models_env else _default_models(REPO_DIR)
DEFAULT_MODEL = os.environ.get("COSYVOICE_DEFAULT_MODEL", "cosyvoice2").strip()
VOICES_DIR = os.environ.get("COSYVOICE_VOICES_DIR", "voices")
WETEXT_DIR = os.environ.get("COSYVOICE_WETEXT_DIR") or os.path.join(REPO_DIR, "pretrained_models", "wetext")
FP16 = _env_bool("COSYVOICE_FP16", False)
MAX_LOADED = int(os.environ.get("COSYVOICE_MAX_LOADED", "1"))
API_TOKEN = os.environ.get("COSYVOICE_API_TOKEN", "").strip()
RUN_WARMUP = _env_bool("COSYVOICE_WARMUP", True)

CONTENT_TYPES = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg", "pcm": "audio/pcm"}

service = CosyVoiceService(
    repo_dir=REPO_DIR,
    model_dirs=MODEL_DIRS,
    voices_dir=VOICES_DIR,
    wetext_dir=WETEXT_DIR,
    default_model=DEFAULT_MODEL,
    fp16=FP16,
    max_loaded=MAX_LOADED,
)


def _load_model_background() -> None:
    try:
        service.load(DEFAULT_MODEL)
        if RUN_WARMUP:
            service.warmup(DEFAULT_MODEL)
    except Exception:  # noqa: BLE001 - keep the server alive, report via /health
        logger.exception("Failed to load CosyVoice model")


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "Starting CosyVoice API server (repo=%s, models=%s, default=%s, fp16=%s, max_loaded=%d)",
        REPO_DIR, ", ".join(MODEL_DIRS), DEFAULT_MODEL, FP16, MAX_LOADED,
    )
    loader = asyncio.get_running_loop().run_in_executor(None, _load_model_background)
    yield
    try:
        await asyncio.wait_for(loader, timeout=5)
    except Exception:  # noqa: BLE001
        pass
    logger.info("CosyVoice API server stopped")


app = FastAPI(
    title="CosyVoice API",
    description="OpenAI-compatible speech synthesis API for CosyVoice2 / Fun-CosyVoice3",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    """Static bearer-token check. Disabled when no token is configured."""
    if not API_TOKEN:
        return
    token = credentials.credentials if credentials else ""
    if token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ------------------------------------------------------------------- helpers


def _require_ready() -> None:
    if not service.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still loading, retry later (see /health)",
        )


def _resolve_model(name: str) -> str:
    if name in MODEL_DIRS:
        return name
    lowered = {key.lower(): key for key in MODEL_DIRS}
    if name.lower() in lowered:
        return lowered[name.lower()]
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"unknown model '{name}', configured: {', '.join(MODEL_DIRS)}",
    )


def _audio_response_meta(fmt: str, sample_rate: int, content_length: Optional[int] = None) -> Tuple[str, dict]:
    fmt = fmt.lower()
    media_type = CONTENT_TYPES.get(fmt, "application/octet-stream")
    headers = {
        "Content-Disposition": f"attachment; filename=speech.{fmt}",
        "Cache-Control": "no-store",
    }
    if fmt == "pcm":
        headers.update({
            "X-Sample-Rate": str(sample_rate),
            "X-Channels": "1",
            "X-Format": "s16le",
        })
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    return media_type, headers


# --------------------------------------------------------------------- routes


@app.get("/health")
async def health_check():
    payload = {
        "status": "ok" if service.ready else "loading",
        "default_model": DEFAULT_MODEL,
        "model_loaded": service.ready,
        "loaded_models": service.loaded_models(),
        "models": service.model_report(),
        "voices": len(service.list_voices()),
        "max_loaded": MAX_LOADED,
        "repo": REPO_DIR,
    }
    vram = service.vram_report()
    if vram:
        payload["vram"] = vram
    return JSONResponse(payload, status_code=200 if service.ready else 503)


@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": key,
                "object": "model",
                "created": 0,
                "owned_by": "cosyvoice",
            }
            for key in MODEL_DIRS
        ],
    }


# --------------------------------------------------------------- voice management


@app.get("/v1/voices", dependencies=[Depends(verify_token)])
@app.get("/v1/audio/voices", dependencies=[Depends(verify_token)])
async def list_voices():
    return {"object": "list", "data": service.list_voices()}


@app.post("/v1/voices", dependencies=[Depends(verify_token)], status_code=status.HTTP_201_CREATED)
async def upload_voice(
    file: UploadFile = File(..., description="Reference audio (wav/mp3/flac/ogg/..., 1-28 s, <= 20 MiB)"),
    name: str = Form(..., description="Voice id used later in the 'voice' field (letters/digits/_/-)"),
    prompt_text: Optional[str] = Form(
        None,
        description="Transcript of the reference audio; strongly recommended for zero-shot cloning",
    ),
    consent: Optional[str] = Form(None, description="Optional consent declaration, logged server-side"),
):
    data = await file.read(MAX_VOICE_BYTES + 1)
    ext = os.path.splitext(file.filename or "")[1].lower()
    try:
        info = await run_in_threadpool(service.save_voice, name, data, ext, prompt_text or "")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if consent:
        logger.info("voice upload '%s' consent: %s", name, consent[:120])
    return info


@app.delete("/v1/voices/{voice_id}", dependencies=[Depends(verify_token)])
async def delete_voice(voice_id: str):
    try:
        removed = await run_in_threadpool(service.delete_voice, voice_id)
    except VoiceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return {"object": "voice.deleted", "id": voice_id, "removed": removed}


# --------------------------------------------------------------------- speech


@app.post("/v1/audio/speech", dependencies=[Depends(verify_token)])
async def generate_speech(request: SpeechRequest):
    _require_ready()
    model = _resolve_model(request.model)

    logger.info(
        "speech request: model=%s voice=%s format=%s stream=%s frontend=%s text=%r",
        model, request.voice, request.response_format, request.stream,
        request.text_frontend, request.input[:60],
    )

    synth_kwargs = dict(
        model=model,
        text=request.input,
        voice=request.voice,
        speed=request.speed,
        text_frontend=request.text_frontend,
        prompt_text=request.prompt_text,
        seed=request.seed,
    )

    if request.stream:
        try:
            slot = await run_in_threadpool(service.load, model)
            await run_in_threadpool(service.resolve_voice, request.voice)
        except (VoiceNotFoundError, ModelNotFoundError) as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        chunks = service.synthesize_stream(**synth_kwargs)
        sample_rate = slot.sample_rate
        body = service.iter_encoded_stream(
            chunks, sample_rate, request.response_format, request.sample_rate, request.gain,
        )
        media_type, headers = _audio_response_meta(request.response_format, sample_rate)
        headers["X-Accel-Buffering"] = "no"
        return StreamingResponse(body, media_type=media_type, headers=headers)

    try:
        sample_rate, samples = await run_in_threadpool(service.synthesize, **synth_kwargs)
        audio = await run_in_threadpool(
            service.encode,
            samples,
            sample_rate,
            request.response_format,
            request.sample_rate,
            request.gain,
        )
    except (TTSNotReadyError, VoiceNotFoundError, ModelNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Speech synthesis failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    media_type, headers = _audio_response_meta(request.response_format, sample_rate, content_length=len(audio))
    return Response(content=audio, media_type=media_type, headers=headers)


@app.post("/v1/audio/speech/stream", dependencies=[Depends(verify_token)])
async def generate_speech_sse(request: SpeechRequest):
    """SSE variant: metadata + base64 PCM16 chunks.

    Event payloads: ``start`` / ``audio`` / ``done`` / ``error``.
    """
    _require_ready()
    model = _resolve_model(request.model)
    try:
        slot = await run_in_threadpool(service.load, model)
        await run_in_threadpool(service.resolve_voice, request.voice)
    except (VoiceNotFoundError, ModelNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    sample_rate = slot.sample_rate

    logger.info(
        "speech SSE request: model=%s voice=%s text=%r",
        model, request.voice, request.input[:60],
    )

    synth_kwargs = dict(
        model=model,
        text=request.input,
        voice=request.voice,
        speed=request.speed,
        text_frontend=request.text_frontend,
        prompt_text=request.prompt_text,
        seed=request.seed,
    )

    def event_stream() -> Iterator[str]:
        yield "retry: 1000\n\n"
        yield "data: " + json.dumps({
            "type": "start",
            "model": model,
            "sample_rate": sample_rate,
            "format": "pcm16le",
            "channels": 1,
        }) + "\n\n"
        try:
            for _sr, samples in service.synthesize_stream(**synth_kwargs):
                yield "data: " + json.dumps({
                    "type": "audio",
                    "sample_rate": sample_rate,
                    "audio": base64.b64encode(samples.astype("<i2", copy=False).tobytes()).decode("ascii"),
                }) + "\n\n"
            yield 'data: {"type": "done"}\n\n'
        except Exception as exc:  # noqa: BLE001 - the stream already started; report inline
            logger.exception("SSE synthesis failed")
            yield "data: " + json.dumps({"type": "error", "message": str(exc)}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
