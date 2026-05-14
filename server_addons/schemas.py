from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


VoiceKind = Literal["builtin", "cloned"]


class GenerationParams(BaseModel):
    """Mirror of the upstream `OmniVoiceGenerationConfig` knobs that
    `model.generate(...)` accepts.

    Doc: docs/generation-parameters.md (in this repo). Defaults follow
    upstream; the wrapper only forwards values the caller set so the
    upstream defaults stay authoritative.
    """

    # Decoding
    num_step: Optional[int] = Field(default=None, ge=1, le=128, description="Diffusion steps (32 default, 16 faster).")
    denoise: Optional[bool] = Field(default=None, description="Prepend <|denoise|> token (default True).")
    guidance_scale: Optional[float] = Field(default=None, gt=0.0, le=10.0, description="Classifier-free guidance scale (default 2.0).")
    t_shift: Optional[float] = Field(default=None, gt=0.0, le=2.0, description="Time-step shift for noise schedule (default 0.1).")

    # Sampling
    position_temperature: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Mask-position selection temperature (0=greedy, default 5.0).")
    class_temperature: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Token sampling temperature (0=greedy, default 0.0).")
    layer_penalty_factor: Optional[float] = Field(default=None, ge=0.0, le=10.0, description="Penalty on deeper codebook layers (default 5.0).")

    # Duration / speed (priority: duration > speed)
    duration: Optional[float] = Field(default=None, gt=0.0, le=120.0, description="Fixed output seconds (overrides speed).")
    speed: Optional[float] = Field(default=None, gt=0.0, le=4.0, description="Speed factor (>1 faster, <1 slower; default 1.0).")

    # Pre / post processing
    preprocess_prompt: Optional[bool] = Field(default=None, description="Apply preprocessing to ref audio + add punctuation to ref_text (default True).")
    postprocess_output: Optional[bool] = Field(default=None, description="Strip long silences from generated audio (default True).")

    # Long-form chunking
    audio_chunk_duration: Optional[float] = Field(default=None, gt=0.0, le=60.0, description="Target chunk duration when splitting long text (default 15.0s).")
    audio_chunk_threshold: Optional[float] = Field(default=None, gt=0.0, le=120.0, description="Chunking activates when estimated audio exceeds this (default 30.0s).")


class SpeechRequest(GenerationParams):
    model: str = Field(description="Must equal 'omnivoice'.")
    input: str = Field(description="Text to synthesize. May embed inline tokens (e.g. [laughter]).")
    voice: str = Field(description="Reference id (predefined filename under voices/ or cloned id).")
    response_format: Optional[Literal["wav"]] = Field(default="wav")
    language: Optional[str] = Field(default=None, description="ISO-ish language label (Portuguese, English, …). Currently informational.")
    ref_text: Optional[str] = Field(default=None, description="Optional ICL transcript; if absent OmniVoice auto-transcribes via Whisper.")


class DesignRequest(GenerationParams):
    model: str = Field(description="Must equal 'omnivoice'.")
    input: str = Field(description="Text to synthesize. May embed inline tokens (e.g. [laughter]).")
    instruct: str = Field(description="Comma-separated attribute prompt (gender, age, pitch, style, accent).")
    response_format: Optional[Literal["wav"]] = Field(default="wav")
    language: Optional[str] = Field(default=None)


class AutoRequest(GenerationParams):
    model: str = Field(description="Must equal 'omnivoice'.")
    input: str = Field(description="Text to synthesize. May embed inline tokens.")
    response_format: Optional[Literal["wav"]] = Field(default="wav")
    language: Optional[str] = Field(default=None)


class VoiceItem(BaseModel):
    id: str
    label: str
    kind: VoiceKind
    language: str
    description: Optional[str] = None


class VoiceList(BaseModel):
    voices: list[VoiceItem]


class CloneResponse(BaseModel):
    id: str
    label: str
    kind: VoiceKind
    language: str


class ErrorBody(BaseModel):
    message: str
    actual: Optional[float] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
