# omnivoice-tts-server — Spec delta (ADDED capability)

## ADDED Requirements

### Requirement: nifty-star Wire Contract Parity
The fork SHALL expose the same OpenAI-compatible endpoints used by the
sibling chatterbox-tts-server and qwen3-tts forks so nifty-star's
provider abstraction can switch by URL prefix and provider id alone.

#### Scenario: OpenAI speech endpoint (voice cloning + ICL)
- **WHEN** the client posts `POST /v1/audio/speech` with body
  `{model, input, voice, response_format?, speed?, language?, ref_text?, temperature?, top_p?, top_k?, repetition_penalty?, duration?, num_step?}`
- **THEN** the server MUST validate `model` equals `omnivoice`, resolve
  `voice` to a reference path under `voices/` (predefined) or
  `reference_audio/` (cloned), forward the request text verbatim
  (including any `[laughter]`, `[sigh]`, etc. tokens) to
  `model.generate(text, ref_audio, ref_text?)`, and return HTTP 200 with
  `Content-Type: audio/wav`

#### Scenario: OpenAI design endpoint (voice design)
- **WHEN** the client posts `POST /v1/audio/design` with body
  `{model, input, instruct, response_format?, language?, ...}`
- **THEN** the server MUST forward the request to
  `model.generate(text=input, instruct=instruct)` with `ref_audio=None`
- **AND** the server MUST log a WARN when `language` is set and is not
  `English`, `Chinese`, `en`, or `zh` (Voice Design is upstream-trained
  only on EN+ZH)

#### Scenario: Auto-voice endpoint
- **WHEN** the client posts `POST /v1/audio/auto` with body
  `{model, input, response_format?, language?}`
- **THEN** the server MUST forward to `model.generate(text=input)` with
  neither `ref_audio` nor `instruct`, letting OmniVoice pick a voice
  automatically

#### Scenario: Voice list endpoint
- **WHEN** the client `GET`s `/v1/audio/voices`
- **THEN** the server MUST return HTTP 200 with body
  `{"voices":[{id, label, kind, language, description?}, …]}` merging
  files under `voices/` (`kind:"builtin"`) and the persisted clone index
  (`kind:"cloned"`)

#### Scenario: Voice cloning endpoint
- **WHEN** the client posts `POST /v1/voices/clone` as multipart form
  with fields `name`, `language`, `audio`, optional `ref_text`
- **THEN** the server MUST decode the audio file, downmix to mono,
  trim to at most 15 seconds, persist as PCM-16 WAV under
  `reference_audio/<name>.wav`, append a record to `index.json` with the
  optional `ref_text`, and return HTTP 200 with body
  `{id, label, kind:"cloned", language}`

#### Scenario: Voice deletion endpoint
- **WHEN** the client sends `DELETE /v1/voices/{voice_id}` for a cloned
  voice
- **THEN** the server MUST remove the index entry, delete the reference
  audio file, and return HTTP 204

#### Scenario: Health endpoint readiness gate
- **WHEN** the client sends `GET /health` or `HEAD /health`
- **THEN** the server MUST return 200 only after the OmniVoice model has
  finished loading; while the model is loading the server MUST return
  503 with body `loading`

### Requirement: Inline-Token Pass-Through
The server SHALL NOT mutate the user-supplied text. Tokens recognised by
OmniVoice (`[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`,
`[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`,
`[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`,
`[dissatisfaction-hnn]`) MUST reach the model verbatim.

#### Scenario: Tag survives round-trip
- **WHEN** the client posts `POST /v1/audio/speech` with
  `input="[laughter] Você é demais!"`
- **THEN** the model invocation MUST receive the same string,
  byte-for-byte
- **AND** the generated WAV MUST include an audible laughter event at
  the start

### Requirement: Reference Audio Auto-Trim
The fork SHALL trim every uploaded reference clip to at most 15 seconds
before storing it.

#### Scenario: Long reference trimmed
- **WHEN** the user uploads a 60-second WAV via `/v1/voices/clone`
- **THEN** the server MUST decode the file, downmix to mono, take the
  first 15 seconds, re-encode as PCM-16 WAV, and store the trimmed
  version
- **AND** log `trimming clone ref from 60.00s to 15.00s for memory safety`

### Requirement: GPU-Only Deployment
The fork SHALL deploy as GPU-only. CPU inference is not provided.

#### Scenario: CUDA required at startup
- **WHEN** the container starts and `torch.cuda.is_available()` returns
  `False`
- **THEN** the server MUST refuse to load the model and exit with a
  clear error message

#### Scenario: docker-compose targets nvidia
- **WHEN** the operator runs
  `docker compose --env-file .env -f docker-compose.gpu.prod.yml up -d`
- **THEN** the compose definition MUST request a single nvidia GPU via
  `deploy.resources.reservations.devices` and export
  `NVIDIA_VISIBLE_DEVICES=all` plus
  `NVIDIA_DRIVER_CAPABILITIES=compute,utility`

### Requirement: Two-Flavour GHCR Image Matrix
The fork SHALL publish two GPU image flavours per release tag.

#### Scenario: Lean default image
- **WHEN** the CI release job promotes a new semantic-release tag
  `vX.Y.Z`
- **THEN** GHCR MUST receive
  `ghcr.io/autoeditor-video-or-audio/omnivoice-tts:vX.Y.Z-gpu` and
  `:latest-gpu`, built with `INSTALL_FLASH_ATTN=0` on the lean
  `*-base-*` CUDA layer

#### Scenario: flash-attn variant image
- **WHEN** the CI release job promotes a new semantic-release tag
  `vX.Y.Z`
- **THEN** GHCR MUST receive
  `:vX.Y.Z-gpu-flash` and `:latest-gpu-flash`, built with
  `INSTALL_FLASH_ATTN=1` on the `*-devel-*` CUDA layer

#### Scenario: Cleanup preserves both latest tags
- **WHEN** the `cleanup_ghcr` job runs after a release
- **THEN** it MUST preserve container versions tagged `latest-gpu` and
  `latest-gpu-flash` while keeping at least the 10 most recent versioned
  tags
