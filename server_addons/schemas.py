from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


VoiceKind = Literal["builtin", "cloned"]


class SpeechRequest(BaseModel):
    model: str = Field(description="Must equal 'omnivoice'.")
    input: str = Field(description="Text to synthesize. May embed inline tokens (e.g. [laughter]).")
    voice: str = Field(description="Reference id (predefined filename under voices/ or cloned id).")
    response_format: Optional[Literal["wav"]] = Field(default="wav")
    speed: Optional[float] = Field(default=None, gt=0.0, le=4.0, description="Speaking rate factor.")
    duration: Optional[float] = Field(default=None, gt=0.0, le=120.0, description="Fixed output seconds (overrides speed).")
    num_step: Optional[int] = Field(default=None, ge=1, le=128, description="Diffusion steps (32 default, 16 faster).")
    language: Optional[str] = Field(default=None, description="ISO-ish language label (Portuguese, English, …). Currently informational.")
    ref_text: Optional[str] = Field(default=None, description="Optional ICL transcript; if absent OmniVoice auto-transcribes via Whisper.")


class DesignRequest(BaseModel):
    model: str = Field(description="Must equal 'omnivoice'.")
    input: str = Field(description="Text to synthesize. May embed inline tokens (e.g. [laughter]).")
    instruct: str = Field(description="Comma-separated attribute prompt (gender, age, pitch, style, accent).")
    response_format: Optional[Literal["wav"]] = Field(default="wav")
    speed: Optional[float] = Field(default=None, gt=0.0, le=4.0)
    duration: Optional[float] = Field(default=None, gt=0.0, le=120.0)
    num_step: Optional[int] = Field(default=None, ge=1, le=128)
    language: Optional[str] = Field(default=None)


class AutoRequest(BaseModel):
    model: str = Field(description="Must equal 'omnivoice'.")
    input: str = Field(description="Text to synthesize. May embed inline tokens.")
    response_format: Optional[Literal["wav"]] = Field(default="wav")
    speed: Optional[float] = Field(default=None, gt=0.0, le=4.0)
    duration: Optional[float] = Field(default=None, gt=0.0, le=120.0)
    num_step: Optional[int] = Field(default=None, ge=1, le=128)
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
