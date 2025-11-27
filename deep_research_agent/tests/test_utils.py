# -*- coding: utf-8 -*-
from deep_research_agent.core.utils import truncate_by_words, get_structure_output


def test_truncate_by_words():
    text = "word " * 6000
    out = truncate_by_words(text)
    assert isinstance(out, str)
    assert len(out.split()) <= 5000


def test_get_structure_output():
    blocks = [
        {"type": "tool_use", "input": {"a": 1}},
        {"type": "text", "content": "x"},
        {"type": "tool_use", "input": {"b": 2}},
    ]
    out = get_structure_output(blocks)
    assert out == {"a": 1, "b": 2}