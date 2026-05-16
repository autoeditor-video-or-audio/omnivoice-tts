## [1.5.5](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.5.4...v1.5.5) (2026-05-16)


### Bug Fixes

* **inference:** align defaults to upstream gradio demo combo ([58f11e5](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/58f11e5d4a9b97b717993499ff8b7f6f76436aa7))

## [1.5.4](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.5.3...v1.5.4) (2026-05-15)


### Bug Fixes

* **inference:** default postprocess_output=False so chunked output keeps first word ([2ad7586](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/2ad75866fd8611b3000ba0c18fed4a6eee07c9ae))

## [1.5.3](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.5.2...v1.5.3) (2026-05-15)


### Bug Fixes

* **inference:** default audio_chunk_threshold to 5s for PT-BR clones ([e928f94](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/e928f94ad25f6aa0193b66e6affc3ff3f73d97eb))

## [1.5.2](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.5.1...v1.5.2) (2026-05-15)


### Bug Fixes

* **inference:** cache VoiceClonePrompt per clone + reuse instead of rebuilding per call ([9a34308](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/9a34308c1a0c61afc874fd2927fa4a6f74a407bb))

## [1.5.1](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.5.0...v1.5.1) (2026-05-15)


### Bug Fixes

* **voices:** backfill ref_text on first resolve for legacy clones ([b8e793e](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/b8e793e9cad5b442b9b07d1464d81d05fe2024fe))

# [1.5.0](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.4.0...v1.5.0) (2026-05-15)


### Features

* **release:** force republish to land v1.4.0 image artifacts ([befea0b](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/befea0b915f89d351b117e1c25315a86d0c25a0d))

# [1.4.0](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.3.1...v1.4.0) (2026-05-15)


### Features

* **inference,voices:** pin cloned voice across multi-line synth on PT-BR ([1cab6ad](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/1cab6ad1292c348003547a2ebfaf2b1dd4f762e6))

## [1.3.1](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.3.0...v1.3.1) (2026-05-15)


### Bug Fixes

* **inference:** bump audio_chunk_threshold default to 120s ([faff5af](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/faff5af37a92d93e6f255c89e6f09178b84c5c7a))

# [1.3.0](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.2.2...v1.3.0) (2026-05-15)


### Features

* **ci:** forced rebuild to republish gpu-flash with sanitize ellipsis fix ([5490e3e](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/5490e3eaaaef9db409a0932820798c5c06d8ffa7))

## [1.2.2](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.2.1...v1.2.2) (2026-05-15)


### Bug Fixes

* **sanitize:** normalise ellipsis + use period instead of … for pause ([a430cfc](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/a430cfc64b2a3c96a95e58aa0883c0b160c0756d))

## [1.2.1](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.2.0...v1.2.1) (2026-05-15)


### Bug Fixes

* **inference:** bump sanitize_text rewrite log to INFO ([d9e357e](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/d9e357edf3274813a99fcba6979170036baa62d4))

# [1.2.0](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.1.0...v1.2.0) (2026-05-15)


### Features

* **ci:** forced rebuild to verify pipeline reaches publish stage ([d83f344](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/d83f3446331f1081599ea8e07c9bdeb189f47cf3))

# [1.1.0](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.0.1...v1.1.0) (2026-05-14)


### Features

* **server:** expose full OmniVoice generation params via shared GenerationParams base ([8251c45](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/8251c45164376146684c809863093253f4aefe40))

## [1.0.1](https://github.com/autoeditor-video-or-audio/omnivoice-tts/compare/v1.0.0...v1.0.1) (2026-05-14)


### Bug Fixes

* **inference:** sanitize inline markup so cloned voice does not drift ([5302261](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/53022616a2447dadfcc673f451f2f29b097e3c8a))

# 1.0.0 (2026-05-13)


### Bug Fixes

* batch_inference without ref_text or ref_audio_path ([#70](https://github.com/autoeditor-video-or-audio/omnivoice-tts/issues/70)) ([88596a0](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/88596a0759c9d57b7149060fea0e47288f943539))
* instruct in infer_batch ([19f6d7b](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/19f6d7b99bd392191f0a0a73e48d8a41b0cf67df))


### Features

* nifty-star compat OpenAI server + Docker.gpu + ops parity ([bc2eea0](https://github.com/autoeditor-video-or-audio/omnivoice-tts/commit/bc2eea07cbb61d82d10a2a8fba72e1e7acaca9ac))
