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
# `…` (U+2026) and triple-dot ASCII are out-of-distribution for the
# OmniVoice tokenizer in PT-BR — they hint a long pause to humans but
# the model treats the bytes as unknown punctuation, drifts the
# acoustic context, and the cloned voice slips. Normalise both forms
# to a plain period+space.
_ELLIPSIS_FORMS = re.compile(r"…|\.\.\.+")
# Collapse "X . Y" -> "X. Y" (left over from prosodic rewrites that
# add a period that turned out to be redundant with surrounding punct).
_DOUBLE_PERIOD = re.compile(r"\.\s*\.")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;!?])")


def _prosodic_replace(separator: str, lead: str, trail: str) -> str:
    """Swap a tag for `separator` while keeping at most one strong punctuation.

    `separator` is the prosodic mark we want to insert in place of the
    bracketed tag. We use plain period+space for long pauses (instead of
    the U+2026 ellipsis) because the tokenizer handles them in
    distribution; ellipsis was producing unknown-token drift on PT-BR
    cloned voices.
    """
    keep = ""
    for ch in (lead + trail):
        if ch in ".!?":
            keep = ch
            break
    if separator == "PAUSE":
        return f"{keep} . " if keep else ". "
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
        return _prosodic_replace("PAUSE", lead, trail)

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
    - `[pause]` / `[break]` / `[silence]` / `[pause:Ns]` map to a
      period+space (long prosodic pause); breath/inhale/etc. map to
      comma+space.
    - `…` (U+2026) and `...`/`....`/etc. ASCII triple-dot are
      normalised to `. ` for the same reason: both forms drift the
      cloned voice on PT-BR despite parsing fine in EN/ZH.
    - Anything else inside `[...]` is dropped (logged at DEBUG).

    Without this rewrite the tokenizer treats unknown brackets as
    literal characters, the model tries to vocalise them, and the
    reference-audio conditioning drifts so subsequent words come out
    in a random voice instead of the cloned one.
    """
    if not text:
        return text
    original = text
    if "[" in text:
        text = _ANY_BRACKET_TAG.sub(_classify_and_rewrite, text)
    text = _ELLIPSIS_FORMS.sub(". ", text)
    # Collapse the duplicated period that arises when a [pause] sat
    # between two punctuation marks: "X. . Y" -> "X. Y".
    while _DOUBLE_PERIOD.search(text):
        text = _DOUBLE_PERIOD.sub(".", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
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

# Generation defaults: aligned with what upstream's gradio demo
# (`omnivoice/cli/demo.py:187`) actually passes — that combination
# (num_step=32, guidance_scale=2.0, denoise=True, preprocess_prompt=True,
# postprocess_output=True, plus the upstream library defaults for
# everything we don't override) is the one isolated REPL tests on
# `Adam-Padra.wav` confirmed reproduces the upstream demo's audio
# quality on PT-BR clones: cloned voice stays consistent across
# repeated calls AND the first word of the input is preserved.
#
# Earlier defaults that overrode those values (low chunking, greedy
# sampling, postprocess_output=False, per-request torch.manual_seed)
# were reasoned chases that each introduced their own regression.
# Stick with the demo's combo unless someone *measures* a better one.
#
# Operators can still tune any of these per-request via the
# GenerationParams body or by overriding the matching OMNIVOICE_*
# env var. None of these defaults inject the knob when the caller
# already set it.
OMNIVOICE_GUIDANCE_SCALE_DEFAULT = float(
    os.environ.get("OMNIVOICE_GUIDANCE_SCALE", "2.0")
)
OMNIVOICE_DENOISE_DEFAULT = os.environ.get("OMNIVOICE_DENOISE", "true").lower() in (
    "1",
    "true",
    "yes",
)
OMNIVOICE_PREPROCESS_PROMPT_DEFAULT = os.environ.get(
    "OMNIVOICE_PREPROCESS_PROMPT", "true"
).lower() in ("1", "true", "yes")
OMNIVOICE_POSTPROCESS_OUTPUT_DEFAULT = os.environ.get(
    "OMNIVOICE_POSTPROCESS_OUTPUT", "true"
).lower() in ("1", "true", "yes")
# Chunking + sampling: leave at upstream library defaults
# (audio_chunk_threshold=30.0, audio_chunk_duration=15.0,
# position_temperature=5.0, class_temperature=0.0). We DO NOT
# inject overrides for these in _generate_kwargs anymore.

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


def preload_asr_model() -> None:
    """Force-load the Whisper ASR model so the first clone-creation
    request doesn't pay a ~3-5s latency spike.

    OmniVoice's `create_voice_clone_prompt` auto-transcribes the
    reference audio when no ref_text is supplied; on the first such
    call it lazy-loads Whisper. Pre-loading from the lifespan hook
    keeps clone creation snappy and also lets us auto-transcribe at
    clone-creation time (persisting ref_text in the index so future
    synth requests skip the transcribe call entirely).
    """
    if _model is None:
        logger.warning("preload_asr_model: main model not loaded yet; skipping")
        return
    try:
        if getattr(_model, "_asr_pipe", None) is None:
            logger.info("preload_asr_model: loading Whisper ASR pipeline ...")
            _model.load_asr_model()
            logger.info("preload_asr_model: ASR ready")
    except Exception:
        # Never block startup on ASR load failure — clone-creation will
        # still work on-demand if the lazy path can recover.
        logger.exception("preload_asr_model failed; will fall back to lazy load")


def transcribe_reference(ref_audio_path: Path) -> Optional[str]:
    """Run Whisper on a reference audio file and return the transcript.

    Returns None on failure (caller persists None so the legacy lazy
    transcribe path kicks in per request — old behaviour, no
    regression).
    """
    if _model is None:
        logger.warning("transcribe_reference: model not loaded; skipping")
        return None
    try:
        if getattr(_model, "_asr_pipe", None) is None:
            _model.load_asr_model()
        import soundfile as sf
        import torch

        wav_np, sr = sf.read(str(ref_audio_path), always_2d=False)
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)
        wav_tensor = torch.from_numpy(wav_np).unsqueeze(0).float()
        text = _model.transcribe((wav_tensor, int(sr)))
        if text and text.strip():
            logger.info("transcribe_reference: %s -> %r", ref_audio_path.name, text.strip())
            return text.strip()
    except Exception:
        logger.exception("transcribe_reference failed for %s", ref_audio_path)
    return None


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
    # Inject the same combo upstream's gradio demo passes
    # (`omnivoice/cli/demo.py:187-192`): num_step, guidance_scale,
    # denoise, preprocess_prompt, postprocess_output. Chunking +
    # sampling are intentionally NOT overridden — they stay at the
    # upstream library defaults (audio_chunk_threshold=30.0,
    # audio_chunk_duration=15.0, position_temperature=5.0,
    # class_temperature=0.0).
    if "num_step" not in kwargs:
        kwargs["num_step"] = OMNIVOICE_NUM_STEP_DEFAULT
    if "guidance_scale" not in kwargs:
        kwargs["guidance_scale"] = OMNIVOICE_GUIDANCE_SCALE_DEFAULT
    if "denoise" not in kwargs:
        kwargs["denoise"] = OMNIVOICE_DENOISE_DEFAULT
    if "preprocess_prompt" not in kwargs:
        kwargs["preprocess_prompt"] = OMNIVOICE_PREPROCESS_PROMPT_DEFAULT
    if "postprocess_output" not in kwargs:
        kwargs["postprocess_output"] = OMNIVOICE_POSTPROCESS_OUTPUT_DEFAULT
    if duration is not None:
        kwargs["duration"] = float(duration)
    elif speed is not None:
        kwargs["speed"] = float(speed)
    logger.info("generate kwargs: %s", kwargs)
    return kwargs


def _seed_rng() -> None:
    """No-op shim kept to avoid breaking callers.

    Earlier iteration reset PyTorch's RNG per-request to mask
    sampling drift; that turned out to be a chase. With upstream
    defaults restored (position_temperature=5.0, the demo combo)
    the RNG advance is no longer the dominant factor.
    """
    return


# Per-process cache of pre-built VoiceClonePrompt objects, keyed by
# (ref_audio_path, ref_text_or_None). Upstream's `model.generate(
# ref_audio=path, ref_text=...)` recomputes the audio-tokeniser encode
# and re-runs Whisper auto-transcription on every call when the path
# is a string — those re-runs are the dominant drift source on PT-BR
# clones across consecutive requests. Building a VoiceClonePrompt once
# and forwarding it via `voice_clone_prompt=` keeps the conditioning
# bit-identical for every line of the same clone.
_clone_prompt_cache: dict = {}


def _get_or_build_clone_prompt(ref_audio_path: Path, ref_text: Optional[str]):
    """Return a (cached) VoiceClonePrompt for this ref_audio_path.

    Cache key includes ref_text so a clone that's later backfilled
    (None -> "Hello world") doesn't keep returning the auto-transcribed
    prompt forever.
    """
    if _model is None:
        return None
    key = (str(ref_audio_path), ref_text or None)
    cached = _clone_prompt_cache.get(key)
    if cached is not None:
        return cached
    try:
        prompt = _model.create_voice_clone_prompt(
            ref_audio=str(ref_audio_path),
            ref_text=ref_text,
        )
    except Exception:
        logger.exception(
            "create_voice_clone_prompt failed for %s; falling back to per-request build",
            ref_audio_path,
        )
        return None
    _clone_prompt_cache[key] = prompt
    logger.info(
        "cached voice_clone_prompt for %s (ref_text=%r)",
        ref_audio_path.name,
        (ref_text or "")[:50],
    )
    return prompt


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
    sanitised_ref_text = sanitize_text(ref_text) if ref_text and ref_text.strip() else None

    _seed_rng()

    # Preferred path: build the VoiceClonePrompt once and reuse it.
    # Matches what upstream's gradio demo does (omnivoice/cli/demo.py)
    # and avoids re-running the audio tokeniser + Whisper on every call.
    prompt = _get_or_build_clone_prompt(ref_audio_path, sanitised_ref_text)
    if prompt is not None:
        wavs = _model.generate(
            text=sanitize_text(text),
            voice_clone_prompt=prompt,
            **kwargs,
        )
    else:
        # Fallback (build failed): preserve the legacy per-call path so
        # the request still completes — just with the historical drift
        # characteristic.
        if sanitised_ref_text:
            kwargs["ref_text"] = sanitised_ref_text
        wavs = _model.generate(
            text=sanitize_text(text),
            ref_audio=str(ref_audio_path),
            **kwargs,
        )
    return _wav_bytes(wavs[0], _sample_rate)


def invalidate_clone_prompt(ref_audio_path: Path) -> None:
    """Drop every cached VoiceClonePrompt entry tied to this path.

    Called from VoiceIndex when a clone's ref_text is backfilled or
    when a clone is deleted, so the next synth picks up the fresh
    state instead of replaying the stale prompt.
    """
    key_prefix = str(ref_audio_path)
    for k in list(_clone_prompt_cache.keys()):
        if k[0] == key_prefix:
            _clone_prompt_cache.pop(k, None)


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
    _seed_rng()
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
    _seed_rng()
    wavs = _model.generate(text=sanitize_text(text), **kwargs)
    return _wav_bytes(wavs[0], _sample_rate)
