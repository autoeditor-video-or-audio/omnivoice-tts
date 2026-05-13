# Design — OmniVoice TTS server fork

## Context
Three nifty-star sibling forks (`moss-tts-nano`, `chatterbox-tts-server`,
`qwen3-tts`) already converged on a stable layout:

- Upstream model wrapped by a `server_addons/` FastAPI app exposing the
  OpenAI-compatible nifty-star contract (`/v1/audio/{speech,design,voices}`,
  `/v1/voices/clone`, `/v1/voices/{id}`, `/health`).
- One `Dockerfile.gpu` parameterised by `INSTALL_FLASH_ATTN` +
  `CUDA_BASE_TAG`, building two CI flavours (`gpu` sdpa + `gpu-flash`).
- helm/k8s manifests and a CI matrix that publishes both flavours to
  GHCR, bumps helm `values.yaml`, and prunes old GHCR tags.
- OpenSpec changes archived once the fork is verified on the GPU host.

OmniVoice is a clean fit for this layout. The only delta vs the
qwen3-tts fork is the **inference call shape**.

## Decisions
- **One FastAPI wrapper, three modes, one model load**: OmniVoice exposes
  a single `model.generate(...)` method that gates between cloning,
  voice-design, and auto-voice via which of `ref_audio` / `instruct` are
  passed. The server reuses one warm `OmniVoice.from_pretrained(...)`
  instance across all three routes — no checkpoint swap. Cheaper than the
  qwen3-tts variant split (which had to load Base or VoiceDesign
  separately).
- **Inline tokens forwarded verbatim**: the request payload's `input` /
  `text` string is forwarded to OmniVoice with `[laughter]`, `[sigh]`,
  etc. intact. The server MUST NOT strip or rewrite them — the upstream
  model expects them as input tokens. This is what differentiates
  OmniVoice from the qwen3 Base/VoiceDesign servers, where vibetalker
  has to fold the tags into the `instruct`.
- **Optional `ref_text` enables ICL; omit it for Whisper auto-transcribe**:
  OmniVoice falls back to its bundled Whisper ASR when `ref_text` is
  absent. We default to ICL ON when the clone-store has a stored
  transcript, OFF when only the reference audio is available, mirroring
  the qwen3-tts behaviour.
- **Reference clip auto-trim ≤ 15 s on upload**: same as Chatterbox and
  qwen3-tts (lesson carried from moss-tts-nano OOM).
- **Host port 8007**: the qwen3 Base and qwen3 VoiceDesign stacks both
  publish on 8005 (only one runs at a time). OmniVoice runs on 8007 so
  the operator can A/B both providers without juggling compose down/up.
- **Voice Design caveat exposed in the response**: the upstream README
  states Voice Design was trained on Chinese + English data only. The
  server logs a WARN on `/v1/audio/design` requests when the requested
  language is neither Chinese nor English, but does NOT block the
  request — operator can still experiment.
- **Dockerfile base CUDA layer**: OmniVoice's upstream install
  instruction pins `torch==2.8.0+cu128`, so the base CUDA image is bumped
  to `12.8.x` (vs `12.6.3` in qwen3-tts). The flash-attn prebuilt wheel
  must match (`flash_attn-2.7.4.post1+cu12torch2.8cxx11abiFALSE-cp312`
  if available; fall back to source build via `*-devel-*` base if not).
- **PyPI install vs source install**: install the upstream `omnivoice`
  package from PyPI (`pip install omnivoice==0.1.5`). The fork tree
  contains all upstream files; we only add `server_addons/` on top.

## Alternatives considered
- **Wrap one of the named community projects** (`omnivoice-server` on
  PyPI, `OmniVoice-local` on GitHub). Rejected at the design stage —
  status of those projects is unverified, and the upstream PyPI package
  exposes a small, stable surface (`OmniVoice.from_pretrained` +
  `.generate(...)`). Writing our own FastAPI wrapper is ~150 lines and
  keeps the integration aligned with the qwen3-tts / Chatterbox layout
  we already operate.
- **Reuse the qwen3-tts FastAPI verbatim, change only model name**.
  Tempting but the qwen3-tts server hard-codes two endpoints
  (`/v1/audio/speech` for cloning, `/v1/audio/design` for VoiceDesign)
  on top of a variant-gated model load. OmniVoice serves both modes from
  one model, so the variant gate is gone and the cloning path accepts
  `ref_audio=None` as a valid "auto voice" request. Cleaner to write
  fresh than to monkey-patch the qwen3 file.

## Risks
- OmniVoice's Voice Design is Chinese+English-only by training. PT-BR
  Voice Design quality may be poor. Mitigation: the `/v1/audio/design`
  log warns, vibetalker UI surfaces the same warning.
- `torch==2.8.0+cu128` is more recent than what the qwen3-tts fork uses
  (`torch==2.5/cu126`). Flash-attn prebuilt wheels for cu128 + torch 2.8
  + py3.12 + cxx11abiFALSE must exist or the build falls back to source.
- 5,835 GitHub stars and an arXiv ID with a 2026-04 prefix suggest a
  brand-new release. If a regression turns up in the v0.1.5 PyPI
  package, pin the install at this exact version and avoid `latest`.

## Migration
None. New fork, new image, new compose file. Existing Qwen3 deployments
keep working unchanged on port 8005.
