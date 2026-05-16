"""Tests pinning the env-driven generation defaults.

Defaults match what upstream's gradio demo (`omnivoice/cli/demo.py:187`)
passes — that combination is the one isolated REPL tests on PT-BR
clones confirmed reproduces the demo's audio quality (cloned voice
stays consistent + first word of input is preserved).
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def inference(monkeypatch):
    """Reload server_addons.inference under controlled env so module-
    level constants reflect the test's overrides."""
    import server_addons.inference as mod

    return importlib.reload(mod)


def test_default_num_step(inference):
    assert inference.OMNIVOICE_NUM_STEP_DEFAULT == 32


def test_default_guidance_scale(inference):
    assert inference.OMNIVOICE_GUIDANCE_SCALE_DEFAULT == 2.0


def test_default_denoise_true(inference):
    assert inference.OMNIVOICE_DENOISE_DEFAULT is True


def test_default_preprocess_prompt_true(inference):
    assert inference.OMNIVOICE_PREPROCESS_PROMPT_DEFAULT is True


def test_default_postprocess_output_true(inference):
    assert inference.OMNIVOICE_POSTPROCESS_OUTPUT_DEFAULT is True


def test_generate_kwargs_injects_demo_combo(inference):
    kwargs = inference._generate_kwargs()
    assert kwargs["num_step"] == 32
    assert kwargs["guidance_scale"] == 2.0
    assert kwargs["denoise"] is True
    assert kwargs["preprocess_prompt"] is True
    assert kwargs["postprocess_output"] is True


def test_generate_kwargs_does_not_force_chunking(inference):
    """Chunking knobs must be left at upstream library defaults so the
    demo combo's behaviour stays intact (low chunking caused dropped
    first words on PT-BR)."""
    kwargs = inference._generate_kwargs()
    assert "audio_chunk_threshold" not in kwargs
    assert "audio_chunk_duration" not in kwargs


def test_generate_kwargs_does_not_force_sampling(inference):
    """Sampling temperatures must be left at upstream library defaults
    (greedy mode was a chase that didn't fix the real drift)."""
    kwargs = inference._generate_kwargs()
    assert "position_temperature" not in kwargs
    assert "class_temperature" not in kwargs


def test_generate_kwargs_caller_overrides_win(inference):
    kwargs = inference._generate_kwargs(
        guidance_scale=3.5,
        postprocess_output=False,
        audio_chunk_threshold=10.0,
    )
    assert kwargs["guidance_scale"] == 3.5
    assert kwargs["postprocess_output"] is False
    assert kwargs["audio_chunk_threshold"] == 10.0


def test_generate_kwargs_duration_overrides_speed(inference):
    kwargs = inference._generate_kwargs(speed=1.5, duration=8.0)
    assert kwargs["duration"] == 8.0
    assert "speed" not in kwargs


def test_env_override_guidance_scale(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_GUIDANCE_SCALE", "3.0")
    import server_addons.inference as mod

    mod = importlib.reload(mod)
    assert mod.OMNIVOICE_GUIDANCE_SCALE_DEFAULT == 3.0
    kwargs = mod._generate_kwargs()
    assert kwargs["guidance_scale"] == 3.0


def test_env_override_postprocess_output_false(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_POSTPROCESS_OUTPUT", "false")
    import server_addons.inference as mod

    mod = importlib.reload(mod)
    assert mod.OMNIVOICE_POSTPROCESS_OUTPUT_DEFAULT is False


def test_seed_rng_is_noop(inference):
    """The per-request reseed was rolled back; ensure the shim doesn't
    raise so callers in synthesize_* keep working."""
    inference._seed_rng()  # must not raise
