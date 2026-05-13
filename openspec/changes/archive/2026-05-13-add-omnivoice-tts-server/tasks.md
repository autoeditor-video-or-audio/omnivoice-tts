# Tasks — OmniVoice TTS server fork

## 1. Repo scaffolding
- [x] 1.1 Fork `k2-fsa/OmniVoice` to `autoeditor-video-or-audio/omnivoice-tts`
- [x] 1.2 Clone to `/home/vagrant/mvp/editaudiotomovie/omnivoice-tts`
- [x] 1.3 `openspec init . --tools claude`
- [x] 1.4 Write `proposal.md`, `design.md`, this `tasks.md`

## 2. Capability claims verification
- [x] 2.1 Confirm license = Apache-2.0 (from GitHub sidebar + HF card)
- [x] 2.2 Confirm Portuguese in `docs/lang_id_name_map.tsv` (`pt`, 16855.05 h)
- [x] 2.3 Confirm event token list in README (`[laughter]`, `[sigh]`, …)
- [x] 2.4 Confirm upstream Python API surface
  (`OmniVoice.from_pretrained` + `model.generate(text, ref_audio?, ref_text?, instruct?, num_step?, speed?, duration?)`)

## 3. Spec
- [x] 3.1 `specs/omnivoice-tts-server/spec.md` (strict-validated)

## 4. FastAPI wrapper (`server_addons/`)
- [ ] 4.1 `server_addons/inference.py` — singleton model load, three call
      paths (`speech` / `design` / `auto`), inline-token pass-through
- [ ] 4.2 `server_addons/voices.py` — clone store (`data/voices/index.json`),
      auto-trim ≤ 15 s on upload, list/resolve/delete
- [ ] 4.3 `server_addons/schemas.py` — Pydantic models for the request
      bodies (mirror qwen3-tts schemas; add `instruct` and `auto` shapes)
- [ ] 4.4 `server_addons/server_app.py` — FastAPI lifespan, asyncio
      synth lock, all routes
- [ ] 4.5 `server_addons/tests/test_smoke.py` — minimal HTTP smoke that
      can run with `QWEN3_SKIP_MODEL_LOAD`-style flag (port to
      `OMNIVOICE_SKIP_MODEL_LOAD`)

## 5. Docker + ops parity
- [ ] 5.1 `Dockerfile.gpu` — CUDA 12.8 base, Python 3.12, torch
      2.8 + cu128, `pip install omnivoice==0.1.5 + hf_transfer`
- [ ] 5.2 Flash-attn variant arg + prebuilt wheel for torch 2.8 / cu12 /
      cp312 (fall back to source build via devel CUDA base)
- [ ] 5.3 `docker-compose.gpu.prod.yml` — host port 8007, `OMNIVOICE_*`
      env vars, named volume `cache-omnivoice-hf`
- [ ] 5.4 `.env.example` — defaults for production deploys
- [ ] 5.5 `helm/{Chart.yaml,values.yaml,templates/*}` — copy from
      qwen3-tts helm chart, rename refs
- [ ] 5.6 `k8s/{configmap,deployment,pvc,service}.yaml` — same as above
- [ ] 5.7 `.github/workflows/ci.yml` — matrix `publish_gpu` +
      `publish_gpu_flash`, helm bump, GHCR cleanup keeping both
      `latest-gpu(-flash)?`
- [ ] 5.8 `.releaserc.json` — disable PR/issue lookups (lesson from
      qwen3-tts release flow)
- [ ] 5.9 `NOTICE-fork.md` — list of additions on top of upstream
- [ ] 5.10 `README-fork.md` — quick-start + dual-image flavours

## 6. Verification
- [ ] 6.1 `docker compose -f docker-compose.gpu.prod.yml up -d` on GPU
      host → `/health` 200 within 240 s
- [ ] 6.2 Smoke PT-BR clone (no event token): `POST /v1/audio/speech`
      returns playable WAV in PT-BR
- [ ] 6.3 Smoke event-token: `POST /v1/audio/speech` with
      `"[laughter] Você é demais!"` produces audible laughter, not the
      word "laughter"
- [ ] 6.4 Smoke voice-design (English): `POST /v1/audio/design` with
      `"female, low pitch, british accent"` returns a British-accented
      English WAV
- [ ] 6.5 Smoke voice-design (PT-BR): document quality — expected to be
      unstable per upstream README

## 7. Spec validate + archive (post-bake-off)
- [ ] 7.1 `openspec validate add-omnivoice-tts-server --strict`
- [ ] 7.2 `openspec archive add-omnivoice-tts-server --yes` once GPU
      smoke + vibetalker bake-off both green
