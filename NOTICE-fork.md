# Fork Modifications Notice

This fork (`autoeditor-video-or-audio/omnivoice-tts`) is derived from
[`k2-fsa/OmniVoice`](https://github.com/k2-fsa/OmniVoice), licensed under
the Apache License 2.0. Upstream `LICENSE` and `README.md` are preserved
unchanged.

Per Apache 2.0 §4(b), the following files were **added** by this fork
(no upstream source files were modified):

```
server_addons/__init__.py
server_addons/server_app.py
server_addons/voices.py
server_addons/inference.py
server_addons/schemas.py
server_addons/tests/__init__.py
Dockerfile.gpu
docker-compose.gpu.prod.yml
.env.example
helm/Chart.yaml
helm/values.yaml
helm/templates/configmap.yaml
helm/templates/deployment.yaml
helm/templates/pvc.yaml
helm/templates/service.yaml
k8s/configmap.yaml
k8s/deployment.yaml
k8s/pvc.yaml
k8s/service.yaml
.github/workflows/ci.yml
.releaserc.json
NOTICE-fork.md
README-fork.md
openspec/AGENTS.md
openspec/project.md
openspec/changes/add-omnivoice-tts-server/proposal.md
openspec/changes/add-omnivoice-tts-server/design.md
openspec/changes/add-omnivoice-tts-server/tasks.md
openspec/changes/add-omnivoice-tts-server/specs/omnivoice-tts-server/spec.md
```

Purpose: expose a nifty-star-compatible OpenAI-style FastAPI server
(`/v1/audio/speech`, `/v1/audio/design`, `/v1/audio/auto`,
`/v1/audio/voices`, `/v1/voices/clone`, `/v1/voices/{id}`, `/health`)
around the upstream `omnivoice` PyPI package, ship a GPU Dockerfile +
compose + helm + k8s + CI matching the sibling forks
(`chatterbox-tts-server`, `qwen3-tts`).

OmniVoice inline event tokens (`[laughter]`, `[sigh]`, `[confirmation-en]`,
`[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`,
`[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`,
`[surprise-yo]`, `[dissatisfaction-hnn]`) are forwarded verbatim from the
request text — the server intentionally does not strip or rewrite them.
