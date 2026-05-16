"""Tests for the inline-markup sanitizer.

OmniVoice's tokenizer recognises a closed set of non-verbal tags
(see omnivoice/models/omnivoice.py:_NONVERBAL_PATTERN). Anything else
inside `[...]` tokenises as literal characters and the model desyncs
from the cloned reference voice. The `…` (U+2026) ellipsis and the
`...`/`....` ASCII triple-dot sequence behave the same way on PT-BR
clones, so we normalise them to a plain period+space.

Each case below pins one rewrite rule so a regression in
`sanitize_text` shows up as a clean unit-test failure rather than as
"voice sounded weird in production".
"""

from __future__ import annotations

import pytest

from server_addons.inference import sanitize_text


@pytest.mark.parametrize(
    "raw,expected",
    [
        # No markup: identity.
        ("Texto normal sem markup.", "Texto normal sem markup."),
        ("", ""),
        # Upstream-recognised tags are preserved verbatim (lowercased).
        ("Test [laughter] keep.", "Test [laughter] keep."),
        ("Test [sigh] keep.", "Test [sigh] keep."),
        ("Test [LAUGHTER] case.", "Test [laughter] case."),
        ("Test [question-en] x.", "Test [question-en] x."),
        # Synonyms map to the canonical upstream tag.
        ("Test [laughs] x.", "Test [laughter] x."),
        ("Test [chuckle] x.", "Test [laughter] x."),
        ("Test [annoyed sigh] x.", "Test [sigh] x."),
        ("Test [deep sigh] x.", "Test [sigh] x."),
        # Pause family -> period+space; surrounding punct collapses.
        (
            "Oi, pessoal. [pause] Se eu desaparecer amanhã.",
            "Oi, pessoal. Se eu desaparecer amanhã.",
        ),
        ("Test [pause:2s] dur.", "Test. dur."),
        ("Test [break] x.", "Test. x."),
        ("Test [silence] x.", "Test. x."),
        # Soft-break family -> short prosodic pause (comma).
        ("Test [breath] x.", "Test, x."),
        ("Test [inhale] x.", "Test, x."),
        # Unknown tags get dropped.
        ("aaa [unknown_tag] bbb", "aaa bbb"),
        # Ellipsis (Unicode + ASCII triple-dot) normalised to ". ".
        ("Just dots... and ellipsis…", "Just dots. and ellipsis."),
        # ASCII + typographic quotes stripped (else first clause is
        # dropped from rendered PT-BR audio).
        ('"Criou" é forte, né?', 'Criou é forte, né?'),
        ('“Criou” é forte', 'Criou é forte'),
        ("‘Criou’ é forte", "Criou é forte"),
        # Mixed input from the production bug report.
        (
            'Criou é forte, né? [pause] Você juntou código.',
            'Criou é forte, né?. Você juntou código.',
        ),
        (
            "É exatamente assim. [annoyed sigh] Você mal sabe.",
            "É exatamente assim. [sigh] Você mal sabe.",
        ),
    ],
)
def test_sanitize_text(raw: str, expected: str) -> None:
    assert sanitize_text(raw) == expected
