# omnivoice-tts (fork of k2-fsa/OmniVoice)

OpenAI-compatible FastAPI wrapper around [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice),
deployed as a GPU Docker image for the nifty-star sequencer.

## Fork-specific stability fixes for PT-BR cloned voices

Two layers stack on top of upstream. The default generation config
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
