# Add OmniVoice TTS server fork (5th provider for nifty-star)

## Why
nifty-star already ships Kokoro, Chatterbox, Qwen3-TTS Base, and Qwen3-TTS
VoiceDesign as TTS providers. Two pain points remain on the Qwen3-Design
path:

1. Inline event cues (`[sigh]`, `[laughter]`, `[cry]`) are not honoured by
   Qwen3 — the model pronounces the bracketed word literally. vibetalker
   currently folds these tags into the `instruct` string client-side, but
   the result is descriptive rather than expressive.
2. Qwen3 only accepts a generic `"Portuguese"` language label, so PT-BR
   output occasionally drifts to European Portuguese.

OmniVoice (k2-fsa, Apache-2.0) advertises both gaps as first-class
features:

- Native `[laughter]`, `[sigh]`, `[confirmation-*]`, `[question-*]`,
  `[surprise-*]`, `[dissatisfaction-hnn]` tokens embedded in the input
  text.
- 600+ languages including Portuguese with **16,855 hours** of PT
  training data (verified against `docs/lang_id_name_map.tsv`).
- Voice cloning with optional auto-transcription via Whisper.
- Voice Design via comma-separated attributes (gender, age, pitch, style,
  English accent, Chinese dialect). NOTE: Voice Design was trained only
  on Chinese + English; PT-BR Voice Design results may be unstable.
- Fast diffusion-LM architecture (RTF 0.025 ≈ 40× realtime).

The upstream Python package (`pip install omnivoice`) exposes a clean
`OmniVoice.from_pretrained(...).generate(text, ref_audio?, ref_text?,
instruct?, num_step?, speed?, duration?)` API. Three CLI tools exist
(`omnivoice-demo` Gradio, `omnivoice-infer`, `omnivoice-infer-batch`) but
**no FastAPI / OpenAI-compatible server** is shipped upstream.

## What Changes
- Fork `k2-fsa/OmniVoice` to `autoeditor-video-or-audio/omnivoice-tts`
  (already done).
- Add `server_addons/` package implementing a FastAPI wrapper exposing
  the nifty-star wire contract (mirrors the qwen3-tts and Chatterbox
  forks):
  - `POST /v1/audio/speech` — voice cloning with optional `ref_text` ICL.
  - `POST /v1/audio/design` — voice design via natural-language instruct.
  - `POST /v1/audio/auto` — auto voice (no ref, no instruct).
  - `POST /v1/voices/clone` — multipart upload of a reference clip with
    optional transcript and 15-second auto-trim.
  - `GET  /v1/audio/voices` — list of stored reference clips + presets.
  - `DELETE /v1/voices/{id}` — drop a clone.
  - `GET/HEAD /health` — readiness once the model is warm.
- All endpoints forward OmniVoice inline tokens
  (`[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`,
  `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`,
  `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`,
  `[dissatisfaction-hnn]`) verbatim — the server MUST NOT rewrite them.
- Ship `Dockerfile.gpu` parameterised on `INSTALL_FLASH_ATTN` and
  `CUDA_BASE_TAG`. Default base image: `nvidia/cuda:12.6.3-base-ubuntu22.04`
  + Python 3.12 via deadsnakes. PyTorch 2.8 + cu128 to match upstream
  install instructions.
- Ship `docker-compose.gpu.prod.yml` + `.env.example` pinned to host port
  **8007** so OmniVoice can coexist with the existing Qwen3 stacks on
  port 8005.
- Ship CI matrix (`.github/workflows/ci.yml`) that publishes
  `:vX.Y.Z-gpu` (sdpa, lean base CUDA layer) and `:vX.Y.Z-gpu-flash`
  (flash-attn 2, devel CUDA layer) to GHCR, mirroring the qwen3-tts
  pipeline. Helm + k8s manifests in tree.
- Reference clip auto-trim at 15 seconds (carry the moss-tts-nano OOM
  lesson).

## Impact
- Affected specs: NEW `omnivoice-tts-server` capability.
- Affected code: new fork repo only; the vibetalker client gets a
  separate change proposal (`add-omnivoice-provider`).
- Breaking? No — this is a brand-new fork. No existing deployments depend
  on it.

## Discarded after capability check
*Reserved.* Add a finding under this heading if a capability collapses
during execution (e.g. PT-BR drift on real audio, event tokens ignored
on PT input). Halts further work; the proposal is then archived as
discarded.
