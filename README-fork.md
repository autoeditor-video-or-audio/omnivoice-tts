# omnivoice-tts (fork of k2-fsa/OmniVoice)

OpenAI-compatible FastAPI wrapper around [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice),
deployed as a GPU Docker image for the nifty-star sequencer.

## Fork-specific stability fixes for PT-BR cloned voices

Three layers stack on top of upstream. The default generation config
matches what upstream's gradio demo passes
(`omnivoice/cli/demo.py:187`) — isolated REPL tests on PT-BR clones
confirmed that combo is what reproduces the demo's audio quality
(cloned voice stays consistent across repeated calls + the first
word of the input is preserved).

1. **Pre-built `VoiceClonePrompt` cache.** Matches upstream's gradio
   demo: build the prompt once via
   `model.create_voice_clone_prompt(...)` and reuse for every synth.
   Eliminates per-request audio-tokeniser encode + Whisper transcribe
   noise.
2. **Clone-time auto-transcription + lazy backfill.** When a clone is
   uploaded without a `ref_text`, the fork transcribes it ONCE with
   Whisper and persists the result in `index.json`. Legacy clones
   created before this fix get their transcript backfilled on first
   resolve. Whisper is pre-loaded from the lifespan hook so clone
   creation never pays the ASR load tax.
3. **`sanitize_text` rewrites + quote stripping.** `[pause]` /
   `[break]` / `[silence]` map to `. `; `[annoyed sigh]` / `[laughs]`
   / etc. map to upstream-recognised tags (`[sigh]` / `[laughter]`);
   `…` and `...`/`....` normalise to `. `. ASCII `"` and typographic
   curly quotes (`“`, `”`, `‘`, `’`) are stripped — a sentence that
   *starts* with `"` made the upstream tokenizer drop the entire
   first clause from the rendered audio on PT-BR. The prosodic
   information (`forte, né?`) is unchanged; only the bracketing
   chars go.

History — earlier iterations of the fork overrode chunking
(`audio_chunk_threshold=5.0`), sampling
(`position_temperature=0.0`), and post-processing
(`postprocess_output=False`) trying to chase a cloned-voice drift.
Each override fixed one symptom and broke another: low chunking
dropped the first word; greedy sampling didn't anchor the voice on
its own; disabling postprocess kept onset but didn't help drift.
The demo combo (everything at the upstream defaults the Gradio UI
uses) is the actual fix. All those env knobs still exist for tuning
but default to the demo combo now.

## Upstream feature surface

The upstream model ships under Apache 2.0 and supports:

- 600+ languages (including Portuguese, 16,855 h of training data).
- Voice cloning with optional ICL transcript (`ref_text`) or Whisper
  auto-transcription.
- Voice design via comma-separated attributes
  (`gender, age, pitch, style, English accent, Chinese dialect`).
- Inline event tokens forwarded verbatim:
  `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`,
  `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`,
  `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`,
  `[dissatisfaction-hnn]`.
- Pronunciation correction via pinyin (Chinese) or CMU dict (English).

## Hardware

| Mode | Tested on | Notes |
|------|-----------|-------|
| GPU (recommended) | RTX 4060 8 GB | Comfortable with float16 + flash-attn 2 |
| CPU | not supported | Use the `chatterbox-tts-server` sibling fork |

## Quick start (GPU)

```bash
cd /path/to/editaudiotomovie/omnivoice-tts

cp .env.example .env
docker compose --env-file .env \
    -f docker-compose.gpu.prod.yml pull
docker compose --env-file .env \
    -f docker-compose.gpu.prod.yml up -d

# First run downloads the configured OmniVoice checkpoint to the named
# volume cache-omnivoice-hf. /health returns 200 after model warmup.
```

Smoke test:

```bash
# Health
curl http://localhost:8007/health

# Auto voice (no ref, no instruct)
curl -X POST http://localhost:8007/v1/audio/auto \
    -H 'Content-Type: application/json' \
    -d '{"model":"omnivoice","input":"Bom dia, tudo bem?","language":"Portuguese"}' \
    --output /tmp/ov-auto.wav

# Voice design (English-only training data — works best on EN/ZH)
curl -X POST http://localhost:8007/v1/audio/design \
    -H 'Content-Type: application/json' \
    -d '{"model":"omnivoice","input":"Hello there.","instruct":"female, low pitch, british accent"}' \
    --output /tmp/ov-design.wav

# Inline event token
curl -X POST http://localhost:8007/v1/audio/auto \
    -H 'Content-Type: application/json' \
    -d '{"model":"omnivoice","input":"[laughter] Você é demais!","language":"Portuguese"}' \
    --output /tmp/ov-laugh.wav
```

## Local development on WSL (Windows) — iterate without remote rebuilds

The published GHCR image cycles through CI on every commit and that
loop is too slow when you're iterating on `server_addons/` or chasing
a generation-quality bug. On a Windows host with WSL2 + Docker
Desktop + an NVIDIA GPU you can bind-mount the repo into the
container and reload the server in seconds.

### Prerequisites

- Windows 10/11 with WSL2 enabled.
- Ubuntu 22.04 (or newer) WSL distro.
- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
  with **WSL2 backend** enabled in *Settings → General*.
- NVIDIA GPU + recent driver + **WSL2 GPU passthrough** enabled
  (Docker Desktop *Settings → Resources → WSL Integration* + the
  `nvidia-container-toolkit` Docker Desktop ships).
- Verify GPU is visible:
  ```bash
  wsl
  nvidia-smi   # should list your GPU
  docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
  ```

### Clone + first boot

Inside WSL (e.g. `/mnt/wsl/projects/`):

```bash
cd ~/works/services
git clone https://github.com/autoeditor-video-or-audio/omnivoice-tts.git
cd omnivoice-tts

# .env defaults are fine for local dev; flip OMNIVOICE_PORT if 8007
# is already taken.
cp .env.example .env

# Pull the latest published image so the first boot is fast (model
# weights + flash-attn wheel are already inside). After this you can
# bind-mount the repo and your code edits override the baked
# server_addons/.
docker compose --env-file .env -f docker-compose.gpu.prod.yml pull
```

### Bind-mount the repo for live edits

Add a local override (`docker-compose.dev.yml`) so your repo's
`server_addons/` shadows the one baked into the image. Don't commit
this file — it's local-only.

```yaml
# docker-compose.dev.yml
services:
  omnivoice-tts:
    container_name: omnivoice-tts-server
    volumes:
      - ./server_addons:/app/server_addons:ro
      - omnivoice-hf-cache:/root/.cache/huggingface
      - omnivoice-data:/app/data
      - omnivoice-voices:/app/voices
      - omnivoice-ref-audio:/app/reference_audio
volumes:
  omnivoice-hf-cache:
  omnivoice-data:
  omnivoice-voices:
  omnivoice-ref-audio:
```

Run with both files (the second overrides the first):

```bash
docker compose \
  --env-file .env \
  -f docker-compose.gpu.prod.yml \
  -f docker-compose.dev.yml \
  up -d
```

### Iterate

After editing `server_addons/inference.py` (or any other file under
`server_addons/`), restart the container — the bind mount picked up
your changes already, you just need uvicorn to re-import:

```bash
docker compose restart omnivoice-tts
docker logs --tail 50 omnivoice-tts-server
```

For Python-side experiments that don't need uvicorn (e.g. trying a
new generation-config combo on a cloned voice), drop into the
container and run a one-shot REPL — way faster than pushing + waiting
for CI:

```bash
docker exec omnivoice-tts-server python3 - <<'PY'
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
import torch, soundfile as sf
m = OmniVoice.from_pretrained('k2-fsa/OmniVoice', device_map='cuda:0', dtype=torch.float16)
m.load_asr_model()
prompt = m.create_voice_clone_prompt(ref_audio='/app/reference_audio/Adam-Padra.wav')
cfg = OmniVoiceGenerationConfig(num_step=32, guidance_scale=2.0, denoise=True,
                                preprocess_prompt=True, postprocess_output=True)
a = m.generate(text='Oi pessoal, tudo bem?', voice_clone_prompt=prompt, generation_config=cfg)
sf.write('/tmp/repl.wav', a[0], 24000); print('wrote /tmp/repl.wav', a[0].shape)
PY
docker cp omnivoice-tts-server:/tmp/repl.wav .
# Open repl.wav in a Windows player (Explorer auto-mounts \\wsl$\Ubuntu\…).
```

### When you're done iterating

- `docker compose -f docker-compose.gpu.prod.yml -f docker-compose.dev.yml down`
- Commit your changes; push triggers the CI matrix that publishes
  `:vX.Y.Z-gpu` / `:vX.Y.Z-gpu-flash` to GHCR for the rest of the
  team. **No manual `gh workflow run` — the push event already
  triggers CI on this repo.**

### Persisting clones across `docker compose down`

`omnivoice-voices` / `omnivoice-ref-audio` / `omnivoice-data` are
named volumes (separate from the bind-mounted code), so a `down`
keeps your reference audio + `index.json` between restarts. Add or
remove them in `docker-compose.dev.yml` if you prefer host paths
(e.g. `./data:/app/data`) for easier inspection.

## Image flavours

CI publishes two tags per release:

| Tag | Attention kernel | CUDA base layer |
|-----|------------------|------------------|
| `:latest-gpu` | sdpa | `nvidia/cuda:12.8.1-base-ubuntu22.04` (lean) |
| `:latest-gpu-flash` | flash_attention_2 | `nvidia/cuda:12.8.1-devel-ubuntu22.04` (nvcc available) |

The flash-attn variant pulls a prebuilt wheel
(`flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312`); falls back to source
build if the wheel URL ever 404s.

## Switching variants on the same GPU host

OmniVoice runs on host port 8007 by default. The sibling Qwen3 forks
publish on 8005. They can coexist on a single GPU host as long as VRAM
budget allows (OmniVoice ~2 GB float16, Qwen3-1.7B-VoiceDesign ~3.4 GB
bfloat16 — comfortable on an 8 GB card with one Qwen3 stack at a time).
