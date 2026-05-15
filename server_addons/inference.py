"""Singleton wrapper over upstream OmniVoice."""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# OmniVoice has a closed set of inline non-verbal tags that the tokenizer
# recognises (see omnivoice/models/omnivoice.py:_NONVERBAL_PATTERN):
_UPSTREAM_TAGS: frozenset[str] = frozenset(
    {
        "laughter",
        "sigh",
        "confirmation-en",
        "question-en",
        "question-ah",
        "question-oh",
        "question-ei",
        "question-yi",
        "surprise-ah",
        "surprise-oh",
        "surprise-wa",
        "surprise-yo",
        "dissatisfaction-hnn",
    }
)

# Common synonyms / variants we rewrite to a known upstream tag so the
# operator can write naturally without memorising the closed set.
_SYNONYM_TO_UPSTREAM: dict[str, str] = {
    "laugh": "laughter",
    "laughs": "laughter",
    "laughing": "laughter",
    "chuckle": "laughter",
    "chuckles": "laughter",
    "sighs": "sigh",
    "annoyed sigh": "sigh",
    "deep sigh": "sigh",
    "long sigh": "sigh",
    "soft sigh": "sigh",
    "frustrated sigh": "sigh",
}

# Tags with no upstream equivalent: rewrite to plain prosodic punctuation.
_PAUSE_SYNONYMS: frozenset[str] = frozenset(
    {"pause", "break", "silence", "silent", "long pause", "short pause"}
)
_SOFT_BREAK_SYNONYMS: frozenset[str] = frozenset(
    {"breath", "inhale", "exhale", "deep breath", "gasp"}
)

# Any [bracketed token] up to 40 chars; we classify per-tag inside the callback.
_ANY_BRACKET_TAG = re.compile(
    r"(?P<lead>[\s.,;:!?…]*)\[(?P<body>[^\]]{1,40})\](?P<trail>[\s.,;:!?…]*)"
)
_PAUSE_DURATION_TAG = re.compile(r"^pause[:\s]", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def _prosodic_replace(separator: str, lead: str, trail: str) -> str:
    """Swap a tag for `separator` while keeping at most one strong punctuation."""
    keep = ""
    for ch in (lead + trail):
        if ch in ".!?":
            keep = ch
            break
    if separator == "…":
        return f"{keep} … " if keep else "… "
    return f"{keep} , " if keep else ", "


def _classify_and_rewrite(match: "re.Match[str]") -> str:
    body = (match.group("body") or "").strip()
    body_lc = body.lower()
    lead = match.group("lead") or ""
    trail = match.group("trail") or ""

    # 1) Already a recognised upstream tag — keep verbatim, lowercase
    #    (upstream regex is case-sensitive and lowercase).
    if body_lc in _UPSTREAM_TAGS:
        return f"{lead}[{body_lc}]{trail}"

    # 2) Known synonym for an upstream tag — rewrite, preserve context.
    if body_lc in _SYNONYM_TO_UPSTREAM:
        canonical = _SYNONYM_TO_UPSTREAM[body_lc]
        return f"{lead}[{canonical}]{trail}"

    # 3) Pause family (no upstream equivalent) — long prosodic pause.
    if body_lc in _PAUSE_SYNONYMS or _PAUSE_DURATION_TAG.match(body_lc):
        return _prosodic_replace("…", lead, trail)

    # 4) Soft-break family (breath/inhale/etc) — short prosodic pause.
    if body_lc in _SOFT_BREAK_SYNONYMS:
        return _prosodic_replace(", ", lead, trail)

    # 5) Anything else: drop the tag, collapse surrounding whitespace.
    logger.debug("sanitize_text dropped unknown bracket tag: [%s]", body)
    return f"{lead.rstrip()} {trail.lstrip()}"


def sanitize_text(text: str) -> str:
    """Rewrite inline markup so OmniVoice's tokenizer stays in distribution.

    - Recognised tags (`[laughter]`, `[sigh]`, `[question-*]`, etc.) are
      preserved verbatim and lower-cased so upstream's case-sensitive
      regex matches them.
    - Common synonyms (`[laughs]`, `[annoyed sigh]`, `[chuckle]`) are
      rewritten to the canonical upstream tag.
    - `[pause]` / `[break]` / `[silence]` / `[pause:Ns]` map to an
      ellipsis (long prosodic pause); breath/inhale/etc. map to comma.
    - Anything else inside `[...]` is dropped (logged at DEBUG).

    Without this rewrite the tokenizer treats unknown brackets as
    literal characters, the model tries to vocalise them, and the
    reference-audio conditioning drifts so subsequent words come out
    in a random voice instead of the cloned one.
    """
    if not text or "[" not in text:
        return text
    original = text
    text = _ANY_BRACKET_TAG.sub(_classify_and_rewrite, text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    if text != original:
        # INFO so it shows up in default container logs without flipping
        # OMNIVOICE_LOG_LEVEL — operators need to see this when they
        # paste a script with [pause]/[annoyed sigh] and want to confirm
        # the rewrite ran.
        logger.info("sanitize_text rewrote markup: %r -> %r", original, text)
    return text

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


# Knob names accepted by upstream OmniVoiceGenerationConfig and forwarded
# verbatim when the caller sets a value. Source of truth:
# omnivoice/models/omnivoice.py:OmniVoiceGenerationConfig +
# docs/generation-parameters.md.
_FORWARDED_KNOBS = (
    "num_step",
    "denoise",
    "guidance_scale",
    "t_shift",
    "position_temperature",
    "class_temperature",
    "layer_penalty_factor",
    "preprocess_prompt",
    "postprocess_output",
    "audio_chunk_duration",
    "audio_chunk_threshold",
)


def _generate_kwargs(
    *,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
    **knobs,
) -> dict:
    """Build the kwargs dict for `_model.generate(...)`.

    Only forwards knobs the caller set (None -> let upstream default).
    `duration` takes priority over `speed` per upstream contract.
    """
    kwargs: dict = {}
    for name in _FORWARDED_KNOBS:
        value = knobs.get(name)
        if value is not None:
            kwargs[name] = value
    # Default num_step to the env override when the caller did not set it,
    # so the operator can dial inference speed without rebuilding clients.
    if "num_step" not in kwargs:
        kwargs["num_step"] = OMNIVOICE_NUM_STEP_DEFAULT
    if duration is not None:
        kwargs["duration"] = float(duration)
    elif speed is not None:
        kwargs["speed"] = float(speed)
    return kwargs


def synthesize_wav(
    text: str,
    *,
    ref_audio_path: Path,
    ref_text: Optional[str] = None,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
    **knobs,
) -> bytes:
    """Voice-cloning synthesis. ref_text optional (Whisper auto-transcribes)."""
    if _model is None:
        raise RuntimeError("OmniVoice model not initialised")
    kwargs = _generate_kwargs(speed=speed, duration=duration, **knobs)
    if ref_text and ref_text.strip():
        kwargs["ref_text"] = sanitize_text(ref_text)
    wavs = _model.generate(
        text=sanitize_text(text),
        ref_audio=str(ref_audio_path),
        **kwargs,
    )
    return _wav_bytes(wavs[0], _sample_rate)


def synthesize_design_wav(
    text: str,
    *,
    instruct: str,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
    **knobs,
) -> bytes:
    """Voice-design synthesis. No reference audio."""
    if _model is None:
        raise RuntimeError("OmniVoice model not initialised")
    kwargs = _generate_kwargs(speed=speed, duration=duration, **knobs)
    wavs = _model.generate(text=sanitize_text(text), instruct=instruct, **kwargs)
    return _wav_bytes(wavs[0], _sample_rate)


def synthesize_auto_wav(
    text: str,
    *,
    speed: Optional[float] = None,
    duration: Optional[float] = None,
    **knobs,
) -> bytes:
    """Auto-voice synthesis. OmniVoice picks a voice."""
    if _model is None:
        raise RuntimeError("OmniVoice model not initialised")
    kwargs = _generate_kwargs(speed=speed, duration=duration, **knobs)
    wavs = _model.generate(text=sanitize_text(text), **kwargs)
    return _wav_bytes(wavs[0], _sample_rate)
