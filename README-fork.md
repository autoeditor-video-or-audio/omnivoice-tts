# omnivoice-tts (fork of k2-fsa/OmniVoice)

OpenAI-compatible FastAPI wrapper around [`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice),
deployed as a GPU Docker image for the nifty-star sequencer.

## Fork-specific stability fixes for PT-BR cloned voices

Five layers stack on top of upstream to keep a PT-BR cloned voice
stable across a multi-line sequencer playlist. Empirical finding from
isolated REPL tests on `Adam-Padra.wav`: upstream's single-shot
`_generate_iterative` path holds the cloned voice for the first word
or two and then drifts mid-sentence on PT-BR (out-of-distribution
for upstream's EN+ZH training corpus). The fixes below stack so the
final synth path matches what stays on-voice end-to-end.

1. **Low chunking threshold (5s) + short chunks (3s).** This is the
   load-bearing fix. `_generate_chunked` re-conditions the reference
   audio tokens at every chunk boundary, anchoring the clone for the
   full utterance. Without chunking, PT-BR clones drift even with
   deterministic sampling. Override via
   `OMNIVOICE_AUDIO_CHUNK_THRESHOLD` / `OMNIVOICE_AUDIO_CHUNK_DURATION`
   or per-request `GenerationParams`.
2. **Pre-built `VoiceClonePrompt` cache.** Matches upstream's gradio
   demo (`omnivoice/cli/demo.py:209`): build the prompt once via
   `model.create_voice_clone_prompt(...)` and reuse for every synth.
   Eliminates per-request audio-tokeniser encode + Whisper transcribe
   noise.
3. **Greedy sampling by default.** Upstream's
   `position_temperature=5.0` injects Gumbel noise inside the
   diffusion loop, and PyTorch's global RNG advances across requests
   — on PT-BR that drift compounds. The fork defaults
   `OMNIVOICE_POSITION_TEMPERATURE=0.0` and pins
   `OMNIVOICE_CLASS_TEMPERATURE=0.0` (greedy = deterministic).
4. **Per-request RNG reseed.** `OMNIVOICE_REQUEST_SEED=0` (default)
   resets PyTorch's CPU + CUDA RNG at the top of every
   `synthesize_*` call. Belt-and-braces against drift if temperature
   gets flipped back.
5. **Clone-time auto-transcription + lazy backfill.** When a clone is
   uploaded without a `ref_text`, the fork transcribes it ONCE with
   Whisper and persists the result in `index.json`. Legacy clones
   created before this fix get their transcript backfilled on first
   resolve. Whisper is pre-loaded from the lifespan hook so clone
   creation never pays the ASR load tax.

If you're running EN or ZH (upstream's trained languages) and want
to skip the chunking overhead, set
`OMNIVOICE_AUDIO_CHUNK_THRESHOLD=30.0`
`OMNIVOICE_AUDIO_CHUNK_DURATION=15.0` (upstream defaults) in your
`.env`. For stochastic generation, also raise
`OMNIVOICE_POSITION_TEMPERATURE=5.0` and set
`OMNIVOICE_REQUEST_SEED=-1`.

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
