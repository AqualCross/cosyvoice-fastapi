"""Pydantic models for the CosyVoice HTTP API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """Request body of ``POST /v1/audio/speech`` (OpenAI-compatible shape)."""

    model: str = Field(
        "cosyvoice2",
        description="Configured model id (default: cosyvoice2; also cosyvoice3).",
    )
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(
        ...,
        description=(
            "Voice id inside the voices directory, an explicit audio path, or "
            "'<anything>:<voice>' where the prefix is ignored."
        ),
    )
    response_format: Literal["mp3", "wav", "ogg", "pcm"] = Field(
        "mp3",
        description="pcm = raw 16-bit little-endian mono at the model's native sample rate.",
    )
    sample_rate: int = Field(24000, ge=8000, le=48000, description="Ignored for response_format=pcm.")
    stream: bool = Field(
        False,
        description="Emit audio while later text segments are still being generated.",
    )
    speed: float = Field(1.0, gt=0.0, le=3.0, description="1.0 = normal.")
    gain: float = Field(0.0, ge=-30.0, le=30.0, description="Output gain in dB.")
    text_frontend: bool = Field(
        True,
        description="Run text normalization (numbers, units, ...) before synthesis.",
    )
    prompt_text: Optional[str] = Field(
        None,
        description=(
            "Transcript of the reference audio. Overrides the transcript stored "
            "with the voice; CosyVoice needs it for zero-shot cloning."
        ),
    )
    seed: Optional[int] = Field(
        None, ge=0, le=2 ** 31 - 1,
        description=(
            "Resets the RNG before synthesis, making the output reproducible. "
            "Without it each request samples new LLM tokens (upstream top_p 0.8)."
        ),
    )
