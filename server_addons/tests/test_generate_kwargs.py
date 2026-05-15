"""Tests pinning the env-driven generation defaults.

The cloned-voice drift on PT-BR multi-line synth was traced to two
upstream defaults: `position_temperature=5.0` (Gumbel noise +
PyTorch global RNG advancing across requests) and a per-request
Whisper transcribe of unstored ref_text. The fork's defaults are
deterministic by design; this test pins that contract so a future
upstream rebase can't silently flip it.
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


def test_default_position_temperature_is_zero(inference):
    assert inference.OMNIVOICE_POSITION_TEMPERATURE_DEFAULT == 0.0


def test_default_class_temperature_is_zero(inference):
    assert inference.OMNIVOICE_CLASS_TEMPERATURE_DEFAULT == 0.0


def test_default_request_seed_is_zero(inference):
    assert inference.OMNIVOICE_REQUEST_SEED == 0


def test_default_audio_chunk_threshold_is_5(inference):
    # Empirical: 5s threshold + 3s chunks keep PT-BR cloned voices
    # stable. Higher thresholds let _generate_iterative drift mid-line.
    assert inference.OMNIVOICE_AUDIO_CHUNK_THRESHOLD_DEFAULT == 5.0


def test_default_audio_chunk_duration_is_3(inference):
    assert inference.OMNIVOICE_AUDIO_CHUNK_DURATION_DEFAULT == 3.0


def test_generate_kwargs_injects_greedy_sampling(inference):
    kwargs = inference._generate_kwargs()
    assert kwargs["position_temperature"] == 0.0
    assert kwargs["class_temperature"] == 0.0


def test_generate_kwargs_injects_chunking_threshold(inference):
    kwargs = inference._generate_kwargs()
    assert kwargs["audio_chunk_threshold"] == 5.0
    assert kwargs["audio_chunk_duration"] == 3.0


def test_generate_kwargs_caller_overrides_win(inference):
    kwargs = inference._generate_kwargs(
        position_temperature=2.5,
        class_temperature=1.0,
        audio_chunk_threshold=30.0,
    )
    assert kwargs["position_temperature"] == 2.5
    assert kwargs["class_temperature"] == 1.0
    assert kwargs["audio_chunk_threshold"] == 30.0


def test_generate_kwargs_duration_overrides_speed(inference):
    kwargs = inference._generate_kwargs(speed=1.5, duration=8.0)
    assert kwargs["duration"] == 8.0
    assert "speed" not in kwargs


def test_generate_kwargs_speed_when_no_duration(inference):
    kwargs = inference._generate_kwargs(speed=1.5)
    assert kwargs["speed"] == 1.5
    assert "duration" not in kwargs


def test_env_override_position_temperature(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_POSITION_TEMPERATURE", "5.0")
    import server_addons.inference as mod

    mod = importlib.reload(mod)
    assert mod.OMNIVOICE_POSITION_TEMPERATURE_DEFAULT == 5.0
    kwargs = mod._generate_kwargs()
    assert kwargs["position_temperature"] == 5.0


def test_env_override_request_seed_negative_disables_reseed(monkeypatch):
    monkeypatch.setenv("OMNIVOICE_REQUEST_SEED", "-1")
    import server_addons.inference as mod

    mod = importlib.reload(mod)
    assert mod.OMNIVOICE_REQUEST_SEED == -1
    # _seed_rng should no-op without raising even when torch is absent.
    mod._seed_rng()
