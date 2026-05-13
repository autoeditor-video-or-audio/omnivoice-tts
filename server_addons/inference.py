"""Singleton wrapper over upstream OmniVoice."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Selectable knobs via env (documented in README-fork.md).
OMNIVOICE_MODEL = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
OMNIVOICE_DTYPE = os.environ.get("OMNIVOICE_DTYPE", "float16")
OMNIVOICE_DEVICE = os.environ.get("OMNIVOICE_DEVICE", "cuda:0")
OMNIVOICE_NUM_STEP_DEFAULT = int(os.environ.get("OMNIVOICE_NUM_STEP", "32"))
OMNIVOICE_SKIP_MODEL_LOAD = os.environ.get("OMNIVOICE_SKIP_MODEL_LOAD", "").lower() in (
    "1",
    "true",
    "yes",
)

_model = None
_sample_rate = 24000  # OmniVoice generates at 24 kHz per upstream README.


def _resolve_dtype(name: str):
    import torch

    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return table.get(name.lower(), torch.float16)


def is_ready() -> bool:
    return OMNIVOICE_SKIP_MODEL_LOAD or _model is not None


def init_model():
    """Load the OmniVoice model once."""
    global _model
    if _model is not None:
        return _model
    if OMNIVOICE_SKIP_MODEL_LOAD:
        logger.warning("OMNIVOICE_SKIP_MODEL_LOAD set — model load skipped")
        return None

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "OmniVoice requires CUDA. No GPU detected; container will not start.",
        )

    logger.info(
        "loading OmniVoice %s dtype=%s device=%s",
        OMNIVOICE_MODEL,
        OMNIVOICE_DTYPE,
        OMNIVOICE_DEVICE,
    )
    from omnivoice import OmniVoice

    _model = OmniVoice.from_pretrained(
        OMNIVOICE_MODEL,
        device_map=OMNIVOICE_DEVICE,
        dtype=_resolve_dtype(OMNIVOICE_DTYPE),
    )
    logger.info("OmniVoice model ready (sr=%d)", _sample_rate)
    return _model


def _wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    waveform = np.asarray(wav, dtype=np.float32)
    if waveform.ndim == 2 and waveform.shape[1] == 1:
        waveform = waveform[:, 0]
    buf = io.BytesIO()
    sf.write(buf, waveform, int(sr), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _generate_kwargs(
    *,
    num_step: Optional[int],
    speed: Optional[float],
    duration: Optional[float],
) -> dict:
    kwargs: dict = {}
    if num_step is not None:
        kwargs["num_step"] = int(num_step)
    else:
        kwargs["num_step"] = OMNIVOICE_NUM_STEP_DEFAULT
    if duration is not None:
        # When duration is set OmniVoice ignores speed per upstream README.
        kwargs["duration"] = float(duration)
    elif speed is not None:
        kwargs["speed"] = float(speed)
    return kwargs


def synthesize_wav(
    text: str,
    *,
    ref_audio_path: Path,
    ref_text: Optional[str] = None,
    num_step: Optional[int] = None,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
) -> bytes:
    """Voice-cloning synthesis. ref_text optional (Whisper auto-transcribes)."""
    if _model is None:
        raise RuntimeError("OmniVoice model not initialised")
    kwargs = _generate_kwargs(num_step=num_step, speed=speed, duration=duration)
    if ref_text and ref_text.strip():
        kwargs["ref_text"] = ref_text
    wavs = _model.generate(
        text=text,
        ref_audio=str(ref_audio_path),
        **kwargs,
    )
    return _wav_bytes(wavs[0], _sample_rate)


def synthesize_design_wav(
    text: str,
    *,
    instruct: str,
    num_step: Optional[int] = None,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
) -> bytes:
    """Voice-design synthesis. No reference audio."""
    if _model is None:
        raise RuntimeError("OmniVoice model not initialised")
    kwargs = _generate_kwargs(num_step=num_step, speed=speed, duration=duration)
    wavs = _model.generate(text=text, instruct=instruct, **kwargs)
    return _wav_bytes(wavs[0], _sample_rate)


def synthesize_auto_wav(
    text: str,
    *,
    num_step: Optional[int] = None,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
) -> bytes:
    """Auto-voice synthesis. OmniVoice picks a voice."""
    if _model is None:
        raise RuntimeError("OmniVoice model not initialised")
    kwargs = _generate_kwargs(num_step=num_step, speed=speed, duration=duration)
    wavs = _model.generate(text=text, **kwargs)
    return _wav_bytes(wavs[0], _sample_rate)
