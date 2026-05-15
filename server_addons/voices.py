"""Voice index merging predefined files + uploaded clones (mirrors sibling forks)."""

from __future__ import annotations

import json
import logging
import os
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

CLONE_ID_PREFIX = "cl_"
DEFAULT_DATA_DIR = Path(os.environ.get("OMNIVOICE_DATA_DIR", "/app/data/voices")).resolve()
DEFAULT_VOICES_DIR = Path(os.environ.get("OMNIVOICE_VOICES_DIR", "/app/voices")).resolve()
DEFAULT_REFERENCE_DIR = Path(
    os.environ.get("OMNIVOICE_REFERENCE_DIR", "/app/reference_audio")
).resolve()
INDEX_FILENAME = "index.json"
MAX_STORED_SECONDS = float(os.environ.get("OMNIVOICE_CLONE_STORE_MAX_SECONDS", "15"))
MIN_SECONDS = float(os.environ.get("OMNIVOICE_CLONE_MIN_SECONDS", "3"))


@dataclass
class ClonedVoiceRecord:
    id: str
    label: str
    language: str
    ref_path: Path
    ref_text: Optional[str]
    created_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "language": self.language,
            "ref_path": str(self.ref_path),
            "ref_text": self.ref_text,
            "created_at": self.created_at,
        }


class CloneValidationError(Exception):
    def __init__(self, message: str, actual: Optional[float] = None) -> None:
        super().__init__(message)
        self.actual = actual


class VoiceNotFoundError(Exception):
    pass


class BuiltinDeletionError(Exception):
    pass


def _sanitise(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name.strip())
    return safe or "clone"


class VoiceIndex:
    """Source of truth for cloned voices, persisted as JSON on disk."""

    def __init__(
        self,
        data_dir: Path | None = None,
        voices_dir: Path | None = None,
        reference_dir: Path | None = None,
    ) -> None:
        self.data_dir = (data_dir or DEFAULT_DATA_DIR).resolve()
        self.voices_dir = (voices_dir or DEFAULT_VOICES_DIR).resolve()
        self.reference_dir = (reference_dir or DEFAULT_REFERENCE_DIR).resolve()
        self.index_path = self.data_dir / INDEX_FILENAME
        self._clones: dict[str, ClonedVoiceRecord] = {}
        self._ensure_dirs()
        self._load()

    def _ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if not self.index_path.exists():
            self._clones = {}
            return
        try:
            with self.index_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.exception("failed to read voice index; starting empty")
            self._clones = {}
            return
        records: dict[str, ClonedVoiceRecord] = {}
        for entry in raw.get("clones", []):
            try:
                records[entry["id"]] = ClonedVoiceRecord(
                    id=entry["id"],
                    label=entry["label"],
                    language=entry["language"],
                    ref_path=Path(entry["ref_path"]),
                    ref_text=entry.get("ref_text"),
                    created_at=entry.get("created_at", ""),
                )
            except KeyError:
                logger.warning("skipping malformed clone entry: %s", entry)
        self._clones = records

    def _save_atomic(self) -> None:
        payload = {"clones": [r.to_dict() for r in self._clones.values()]}
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.data_dir), suffix=".json.tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.index_path)
        except Exception:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink(missing_ok=True)
            raise

    def list_all(self) -> list[dict]:
        out: list[dict] = []
        for entry in sorted(self.voices_dir.glob("*.wav")):
            out.append(
                {
                    "id": entry.name,
                    "label": entry.stem,
                    "kind": "builtin",
                    "language": "multi",
                    "description": None,
                    "ref_path": str(entry),
                    "ref_text": None,
                }
            )
        for rec in self._clones.values():
            out.append(
                {
                    "id": rec.id,
                    "label": rec.label,
                    "kind": "cloned",
                    "language": rec.language,
                    "description": None,
                    "ref_path": str(rec.ref_path),
                    "ref_text": rec.ref_text,
                }
            )
        return out

    def resolve(self, voice: str) -> tuple[str, Path, Optional[str]]:
        """Return (kind, ref_path, ref_text_or_None).

        For cloned voices that don't yet have a persisted `ref_text`
        (e.g. created before the auto-transcribe-at-clone-time fix
        landed), transparently backfill the transcript via Whisper on
        first resolve and persist it. Subsequent calls return the
        cached value — no per-request Whisper run, no drift.
        """
        rec = self._clones.get(voice)
        if rec is not None:
            if not rec.ref_path.exists():
                raise VoiceNotFoundError(f"{voice}: reference file missing")
            if not (rec.ref_text and rec.ref_text.strip()):
                self._backfill_ref_text(rec)
            return "cloned", rec.ref_path, rec.ref_text
        predefined = self.voices_dir / voice
        if predefined.is_file():
            return "builtin", predefined, None
        raise VoiceNotFoundError(voice)

    def _backfill_ref_text(self, rec: ClonedVoiceRecord) -> None:
        """Transcribe the reference audio once and persist the result.

        Called from `resolve` on legacy clones (created before the
        auto-transcribe-at-clone-creation fix). Failure is logged and
        swallowed so the synth path can still proceed — upstream will
        fall back to lazy per-request Whisper transcription, which is
        the old behaviour.
        """
        try:
            from server_addons import inference

            text = inference.transcribe_reference(rec.ref_path)
            if text and text.strip():
                rec.ref_text = text.strip()
                self._save_atomic()
                logger.info(
                    "backfilled ref_text into index for legacy clone %s", rec.id
                )
        except Exception:
            logger.exception(
                "ref_text backfill failed for %s; falling back to lazy transcribe",
                rec.id,
            )

    def add_clone(
        self,
        *,
        name: str,
        language: str,
        audio_bytes: bytes,
        ref_text: Optional[str] = None,
        suffix: str = ".wav",
    ) -> ClonedVoiceRecord:
        base_name = _sanitise(name) + ".wav"
        target = self.reference_dir / base_name
        while target.exists():
            base_name = f"{_sanitise(name)}_{secrets.token_hex(2)}.wav"
            target = self.reference_dir / base_name

        with tempfile.NamedTemporaryFile(
            suffix=suffix, dir=str(self.reference_dir), delete=False
        ) as tmp_fh:
            tmp_path = Path(tmp_fh.name)
            tmp_fh.write(audio_bytes)

        try:
            samples, sr = sf.read(str(tmp_path), always_2d=False)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise CloneValidationError(f"could not decode audio: {exc}") from exc

        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        samples = np.asarray(samples, dtype=np.float32)
        duration = float(samples.shape[0]) / float(sr or 1)
        if duration < MIN_SECONDS:
            tmp_path.unlink(missing_ok=True)
            raise CloneValidationError(
                f"audio duration must be at least {MIN_SECONDS} seconds",
                actual=duration,
            )

        max_samples = int(sr * MAX_STORED_SECONDS)
        if samples.shape[0] > max_samples:
            logger.info(
                "trimming clone ref from %.2fs to %.2fs for memory safety",
                duration,
                MAX_STORED_SECONDS,
            )
            samples = samples[:max_samples]

        sf.write(str(target), samples, sr, subtype="PCM_16")
        tmp_path.unlink(missing_ok=True)

        # Auto-transcribe at clone-creation time when the caller did
        # not supply a ref_text. OmniVoice falls back to Whisper on
        # every synth request otherwise; that per-request transcribe
        # produces small variations across calls that drift the cloned
        # voice on PT-BR (out-of-distribution for upstream's training).
        # Doing it once here + persisting the transcript in index.json
        # makes the conditioning bit-identical across every subsequent
        # synth.
        effective_ref_text = ref_text
        if not (effective_ref_text and effective_ref_text.strip()):
            try:
                # Late import: voices.py is imported from server_app
                # before the model lifespan loads, so a top-level
                # import of inference would create a cycle.
                from server_addons import inference

                effective_ref_text = inference.transcribe_reference(target)
                if effective_ref_text:
                    logger.info(
                        "auto-transcribed ref_text saved to index for clone %s",
                        base_name,
                    )
            except Exception:
                logger.exception(
                    "auto-transcribe at clone-creation failed; "
                    "Whisper will be invoked per-request instead"
                )
                effective_ref_text = None

        rec = ClonedVoiceRecord(
            id=base_name,
            label=name,
            language=language,
            ref_path=target,
            ref_text=effective_ref_text,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._clones[base_name] = rec
        self._save_atomic()
        return rec

    def delete(self, voice_id: str) -> None:
        predefined = self.voices_dir / voice_id
        if predefined.is_file():
            raise BuiltinDeletionError(voice_id)
        rec = self._clones.pop(voice_id, None)
        if rec is None:
            raise VoiceNotFoundError(voice_id)
        try:
            rec.ref_path.unlink(missing_ok=True)
        finally:
            self._save_atomic()
