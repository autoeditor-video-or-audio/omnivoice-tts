"""FastAPI app exposing the nifty-star wire contract on top of OmniVoice."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from server_addons import inference
from server_addons.schemas import (
    AutoRequest,
    CloneResponse,
    DesignRequest,
    ErrorBody,
    ErrorResponse,
    SpeechRequest,
    VoiceItem,
    VoiceList,
)
from server_addons.voices import (
    BuiltinDeletionError,
    CloneValidationError,
    VoiceIndex,
    VoiceNotFoundError,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("OMNIVOICE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

MODEL_NAME = "omnivoice"

_synth_lock = asyncio.Lock()

_VOICE_DESIGN_TRAINED = {"english", "chinese", "en", "zh"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, inference.init_model)
    except Exception:
        logger.exception("model load failed")
        raise
    yield


app = FastAPI(title="OmniVoice TTS server (autoeditor fork)", version="0.1.0", lifespan=lifespan)
app.state.voice_index = VoiceIndex()


def _err(message: str, status: int = 400, actual: float | None = None) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(message=message, actual=actual)).model_dump(exclude_none=True)
    return JSONResponse(status_code=status, content=body)


@app.api_route("/health", methods=["GET", "HEAD"], response_class=Response)
async def health() -> Response:
    if inference.is_ready():
        return Response(content="ok", status_code=200, media_type="text/plain")
    return Response(content="loading", status_code=503, media_type="text/plain")


@app.get("/v1/audio/voices", response_model=VoiceList)
async def list_voices(request: Request) -> VoiceList:
    index: VoiceIndex = request.app.state.voice_index
    items = [
        VoiceItem(
            id=entry["id"],
            label=entry["label"],
            kind=entry["kind"],  # type: ignore[arg-type]
            language=entry["language"],
            description=entry.get("description"),
        )
        for entry in index.list_all()
    ]
    return VoiceList(voices=items)


@app.post("/v1/audio/speech")
async def synthesize(req: SpeechRequest, request: Request) -> Response:
    if req.model != MODEL_NAME:
        return _err("unsupported model", status=400)
    if not req.input.strip():
        return _err("input must not be empty", status=400)

    index: VoiceIndex = request.app.state.voice_index
    try:
        kind, ref_path, stored_ref_text = index.resolve(req.voice)
    except VoiceNotFoundError:
        return _err("voice not found", status=404)

    if not inference.is_ready():
        return _err("runtime not ready", status=503)

    effective_ref_text: Optional[str] = req.ref_text or stored_ref_text

    loop = asyncio.get_running_loop()
    async with _synth_lock:
        try:
            audio = await loop.run_in_executor(
                None,
                lambda: inference.synthesize_wav(
                    req.input,
                    ref_audio_path=ref_path,
                    ref_text=effective_ref_text,
                    num_step=req.num_step,
                    speed=req.speed,
                    duration=req.duration,
                ),
            )
        except Exception:
            logger.exception("synthesis failed (kind=%s voice=%s)", kind, req.voice)
            return _err("synthesis failed", status=500)

    return Response(content=audio, media_type="audio/wav")


@app.post("/v1/audio/design")
async def synthesize_design(req: DesignRequest) -> Response:
    if req.model != MODEL_NAME:
        return _err("unsupported model", status=400)
    if not req.input.strip():
        return _err("input must not be empty", status=400)
    if not req.instruct.strip():
        return _err("instruct must not be empty", status=400)
    if not inference.is_ready():
        return _err("runtime not ready", status=503)

    lang_norm = (req.language or "").strip().lower()
    if lang_norm and lang_norm not in _VOICE_DESIGN_TRAINED:
        logger.warning(
            "voice design requested with language=%s; upstream OmniVoice was trained on EN+ZH only, "
            "results may be unstable",
            req.language,
        )

    loop = asyncio.get_running_loop()
    async with _synth_lock:
        try:
            audio = await loop.run_in_executor(
                None,
                lambda: inference.synthesize_design_wav(
                    req.input,
                    instruct=req.instruct,
                    num_step=req.num_step,
                    speed=req.speed,
                    duration=req.duration,
                ),
            )
        except Exception:
            logger.exception("design synthesis failed")
            return _err("synthesis failed", status=500)

    return Response(content=audio, media_type="audio/wav")


@app.post("/v1/audio/auto")
async def synthesize_auto(req: AutoRequest) -> Response:
    if req.model != MODEL_NAME:
        return _err("unsupported model", status=400)
    if not req.input.strip():
        return _err("input must not be empty", status=400)
    if not inference.is_ready():
        return _err("runtime not ready", status=503)

    loop = asyncio.get_running_loop()
    async with _synth_lock:
        try:
            audio = await loop.run_in_executor(
                None,
                lambda: inference.synthesize_auto_wav(
                    req.input,
                    num_step=req.num_step,
                    speed=req.speed,
                    duration=req.duration,
                ),
            )
        except Exception:
            logger.exception("auto synthesis failed")
            return _err("synthesis failed", status=500)

    return Response(content=audio, media_type="audio/wav")


@app.post("/v1/voices/clone", response_model=CloneResponse)
async def clone_voice(
    request: Request,
    name: str = Form(...),
    language: str = Form(...),
    audio: UploadFile = File(...),
    ref_text: Optional[str] = Form(None),
) -> CloneResponse:
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio file is empty")

    suffix = ".wav"
    if audio.filename:
        lower = audio.filename.lower()
        if lower.endswith(".mp3"):
            suffix = ".mp3"
        elif lower.endswith(".flac"):
            suffix = ".flac"
        elif lower.endswith(".ogg"):
            suffix = ".ogg"

    index: VoiceIndex = request.app.state.voice_index
    try:
        rec = index.add_clone(
            name=name,
            language=language,
            audio_bytes=raw,
            ref_text=ref_text,
            suffix=suffix,
        )
    except CloneValidationError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=ErrorBody(message=str(exc), actual=exc.actual)
            ).model_dump(exclude_none=True),
        )  # type: ignore[return-value]

    return CloneResponse(id=rec.id, label=rec.label, kind="cloned", language=rec.language)


@app.delete("/v1/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str, request: Request) -> Response:
    index: VoiceIndex = request.app.state.voice_index
    try:
        index.delete(voice_id)
    except BuiltinDeletionError:
        return _err("predefined voices cannot be deleted", status=404)
    except VoiceNotFoundError:
        return _err("voice not found", status=404)
    return Response(status_code=204)
