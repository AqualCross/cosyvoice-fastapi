# cosyvoice-fastapi

OpenAI-compatible **HTTP API server** for [CosyVoice2 / Fun-CosyVoice3](https://github.com/FunAudioLLM/CosyVoice)
— Alibaba's zero-shot voice-cloning TTS models.

This is a **thin wrapper**: it contains no upstream code and no model weights.
It imports the cloned `CosyVoice` repository and loads the weights from its
`pretrained_models/` directory, so both stay external dependencies.

```
✅ OpenAI-compatible POST /v1/audio/speech (works with the OpenAI SDK)
✅ Progressive streaming (stream: true) + a base64-PCM16 SSE endpoint
✅ Voice management over HTTP: upload / list / delete reference voices
✅ One service for both models: model = cosyvoice2 | cosyvoice3
✅ mp3 / wav / ogg via ffmpeg and raw PCM; speed / gain / sample_rate
✅ Text normalization (wetext) attached from a local FST directory
✅ Background model load (health endpoint up immediately), GPU serialization,
   lazy model switching, live VRAM stats
✅ Endpoint layout mirrors indextts25-fastapi, so clients can switch by base URL
```

## Requirements

- [uv](https://docs.astral.sh/uv/) + Python 3.10 (pinned in `.python-version`)
- A cloned [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) repo (`--recursive`) with weights downloaded
- NVIDIA GPU; measured on an RTX 3080 10 GB: CosyVoice2 idle 2.65 GB / peak 2.97 GB,
  Fun-CosyVoice3 idle 3.49 GB / peak 3.79 GB (fp32, one model resident)
- `ffmpeg` / `ffprobe` in `PATH`

## Setup

```bat
:: 1) engine + weights (skip if already deployed)
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
uv venv --python 3.10
uv pip install -r requirements.txt --index-strategy unsafe-best-match
uv pip install "setuptools<81"
:: weights: uv run python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')"

:: 2) this wrapper (uv project: pyproject.toml + uv.lock)
cd ..\cosyvoice-fastapi
uv sync
```

The wrapper's `pyproject.toml` already pins the engine stack (mirroring the
upstream `requirements.txt`, with torch/torchaudio 2.8.0+cu128 from the PyTorch
index). Training-only and TensorRT/webui extras (`deepspeed`, `tensorrt-cu12*`,
`gradio`, `fastapi-cli`) are intentionally not part of this service.

`text_frontend`: the `wetext` package normally downloads its FSTs from
ModelScope on first use. Put the four `zh/tn` and `en/tn` FST files into
`CosyVoice/pretrained_models/wetext/` (or point `COSYVOICE_WETEXT_DIR` at
them); without them the server starts with the frontend disabled and says so
in `/health` and the startup log.

## Run

```bat
start_api.bat
:: or
uv run run_api.py --gpu 0 --port 8001
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `--gpu` | – | physical GPU index (`CUDA_VISIBLE_DEVICES`) |
| `--port` | `8001` | listen port |
| `--repo` | `../CosyVoice` | upstream clone |
| `--models` | `cosyvoice2=<repo>/pretrained_models/CosyVoice2-0.5B,cosyvoice3=<repo>/pretrained_models/Fun-CosyVoice3-0.5B` | model registry |
| `--default-model` | `cosyvoice2` | loaded at startup |
| `--max-loaded` | `1` | resident models; extra loads evict the least recently used |
| `--fp16` | off | fp16 inference |
| `--warmup` | on | one warmup inference after load |
| `--api-token` | – | require `Authorization: Bearer <token>` on `/v1/*` |

Environment variables mirror the arguments: `COSYVOICE_REPO`, `COSYVOICE_MODELS`,
`COSYVOICE_DEFAULT_MODEL`, `COSYVOICE_VOICES_DIR`, `COSYVOICE_WETEXT_DIR`,
`COSYVOICE_FP16`, `COSYVOICE_WARMUP`, `COSYVOICE_MAX_LOADED`, `COSYVOICE_API_TOKEN`.

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | status, loaded/configured models, native sample rate, frontend, VRAM (503 while loading) |
| GET | `/v1/models` | `cosyvoice2`, `cosyvoice3` |
| GET | `/v1/voices`, `/v1/audio/voices` | voices with transcripts |
| POST | `/v1/voices` | multipart: `file`, `name`, `prompt_text`, `consent`; transcoded to mono WAV, 1–28 s |
| DELETE | `/v1/voices/{id}` | removes audio + metadata |
| POST | `/v1/audio/speech` | OpenAI-compatible; `stream: true` returns progressive audio |
| POST | `/v1/audio/speech/stream` | SSE: `start` / `audio` (base64 PCM16) / `done` / `error` |

## Examples

```bash
# upload a reference voice (prompt_text strongly recommended)
python client_example.py --upload-voice alex --voice-file ref.wav \
  --prompt-text "希望你以后能够做的比我还好呦。"

# synthesize
python client_example.py --text "你好，世界" --voice alex --output out.mp3

# OpenAI SDK
python - <<'PY'
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="none")
speech = client.audio.speech.create(model="cosyvoice3", voice="alex", input="你好，世界")
open("out.mp3", "wb").write(speech.content)
PY
```

## Notes

- `prompt_text` must match what the reference audio actually says; it is stored
  with the voice and can be overridden per request. Without it the model still
  runs, but clone quality drops (the server logs a warning once per voice).
- Fun-CosyVoice3 needs its prompt text prefixed with
  `You are a helpful assistant.<|endofprompt|>`; the server adds this
  automatically when the transcript does not already contain it.
- Requests are serialized on the GPU with a lock; long text is split into
  segments by the upstream frontend, and `stream: true` emits audio per segment.
- Output sample rate is 24000 Hz for both models.
- Synthesis is stochastic by default (upstream LLM sampling, top_p 0.8 / top_k 25),
  so the same request yields different audio — including slightly different
  durations. Pass `seed` to reset the RNG and make a request reproducible.

## License

This wrapper is MIT (see `LICENSE`). It depends on external projects:
CosyVoice (Apache-2.0) and its Matcha-TTS submodule (MIT). Model weights are
downloaded from ModelScope/HuggingFace and remain under their own model-card
licenses. No upstream source is vendored in this repository.
