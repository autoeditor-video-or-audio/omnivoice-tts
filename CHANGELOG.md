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
