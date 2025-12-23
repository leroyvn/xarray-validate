"""Tests for pattern matching utilities."""

import re

import pytest

from xarray_validate._match import (
    find_matched_keys,
    is_glob_pattern,
    is_pattern_key,
    is_regex_pattern,
    pattern_to_regex,
    separate_keys,
)


@pytest.mark.parametrize(
    "s, expected",
    [
        ("{x_\\d+}", True),
        ("{[a-z]+_[0-9]{2,4}}", True),
        ("x_\\d+", False),
        ("{x_\\d+", False),
        ("x_\\d+}", False),
        ("", False),
        ("simple_name", False),
        ("{}", True),
    ],
    ids=[
        "simple",
        "complex",
        "no_curly",
        "only_opening",
        "only_closing",
        "empty",
        "plain_text",
        "empty_braces",
    ],
)
def test_is_regex_pattern(s, expected):
    assert is_regex_pattern(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("x_*", True),
        ("x_?", True),
        ("x_*_?", True),
        ("*_*_*", True),
        ("simple_name", False),
        ("", False),
        ("{x_\\d+}", False),
    ],
    ids=[
        "asterisk",
        "question_mark",
        "both_wildcards",
        "multiple_asterisks",
        "plain_text",
        "empty",
        "regex_pattern",
    ],
)
def test_is_glob_pattern(s, expected):
    assert is_glob_pattern(s) is expected


@pytest.mark.parametrize(
    "s, expected",
    [
        ("x_*", True),
        ("{x_\\d+}", True),
        ("prefix_*_suffix", True),
        ("x_?", True),
        ("simple_name", False),
        ("", False),
    ],
    ids=[
        "glob",
        "regex",
        "glob_with_asterisk",
        "glob_with_question",
        "plain",
        "empty",
    ],
)
def test_is_pattern_key(s, expected):
    assert is_pattern_key(s) is expected


class TestPatternToRegex:
    """Tests for pattern_to_regex() function."""

    def test_regex_pattern_conversion(self):
        """Test conversion of regex pattern."""
        pattern = pattern_to_regex("{x_\\d+}")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("x_0")
        assert pattern.match("x_123")
        assert not pattern.match("x_")
        assert not pattern.match("x_foo")

    def test_glob_pattern_asterisk_conversion(self):
        """Test conversion of glob pattern with asterisk."""
        pattern = pattern_to_regex("x_*")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("x_0")
        assert pattern.match("x_foo")
        assert pattern.match("x_bar_baz")
        assert not pattern.match("y_0")

    def test_glob_pattern_question_conversion(self):
        """Test conversion of glob pattern with question mark."""
        pattern = pattern_to_regex("x_?")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("x_0")
        assert pattern.match("x_a")
        assert not pattern.match("x_00")
        assert not pattern.match("x_")

    def test_glob_pattern_mixed_conversion(self):
        """Test conversion of glob pattern with mixed wildcards."""
        pattern = pattern_to_regex("prefix_*_?_suffix")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("prefix_foo_a_suffix")
        assert pattern.match("prefix_bar_1_suffix")
        assert not pattern.match("prefix_foo_suffix")
        assert not pattern.match("prefix_foo_ab_suffix")

    def test_exact_match_conversion(self):
        """Test conversion of plain text to exact match regex."""
        pattern = pattern_to_regex("exact_name")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("exact_name")
        assert not pattern.match("exact_name_suffix")
        assert not pattern.match("prefix_exact_name")

    def test_exact_match_with_special_chars(self):
        """Test exact match with regex special characters."""
        pattern = pattern_to_regex("name.with.dots")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("name.with.dots")
        assert not pattern.match("namexwithxdots")

    def test_complex_regex_pattern(self):
        """Test complex regex pattern with fullmatch."""
        pattern = pattern_to_regex("{[a-z]+_[0-9]{2,4}}")
        assert isinstance(pattern, re.Pattern)
        # Use fullmatch to match entire string (as done in actual code)
        assert pattern.fullmatch("foo_12")
        assert pattern.fullmatch("bar_1234")
        assert not pattern.fullmatch("foo_1")
        assert not pattern.fullmatch("FOO_12")
        assert not pattern.fullmatch("foo_12345")

    def test_regex_with_anchors(self):
        """Test regex pattern with anchors."""
        pattern = pattern_to_regex("{^x_\\d+$}")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("x_123")
        assert not pattern.match("prefix_x_123")
        assert not pattern.match("x_123_suffix")

    def test_glob_empty_match(self):
        """Test glob pattern that can match empty string."""
        pattern = pattern_to_regex("*")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("")
        assert pattern.match("anything")
        assert pattern.match("x_0_y_1")

    def test_regex_alternation(self):
        """Test regex pattern with alternation."""
        pattern = pattern_to_regex("{(foo|bar)_\\d+}")
        assert isinstance(pattern, re.Pattern)
        assert pattern.match("foo_1")
        assert pattern.match("bar_2")
        assert not pattern.match("baz_3")


class TestSeparateKeys:
    """Tests for separate_keys function."""

    def test_all_exact_keys(self):
        """Test separation when all keys are exact (non-pattern)."""
        schema_keys = {"x": 1, "y": 2, "z": 3}
        exact, pattern, compiled = separate_keys(schema_keys)

        assert exact == {"x": 1, "y": 2, "z": 3}
        assert pattern == {}
        assert compiled == {}

    def test_all_pattern_keys(self):
        """Test separation when all keys are patterns."""
        schema_keys = {"x_*": 1, "{y_\\d+}": 2}
        exact, pattern, compiled = separate_keys(schema_keys)

        assert exact == {}
        assert pattern == {"x_*": 1, "{y_\\d+}": 2}
        assert len(compiled) == 2
        assert "x_*" in compiled
        assert "{y_\\d+}" in compiled
        assert isinstance(compiled["x_*"], re.Pattern)
        assert isinstance(compiled["{y_\\d+}"], re.Pattern)

    def test_mixed_keys(self):
        """Test separation with both exact and pattern keys."""
        schema_keys = {"x": 1, "y_*": 2, "z": 3, "{w_\\d+}": 4}
        exact, pattern, compiled = separate_keys(schema_keys)

        assert exact == {"x": 1, "z": 3}
        assert pattern == {"y_*": 2, "{w_\\d+}": 4}
        assert len(compiled) == 2
        assert "y_*" in compiled
        assert "{w_\\d+}" in compiled

    def test_empty_dict(self):
        """Test separation with empty dictionary."""
        schema_keys = {}
        exact, pattern, compiled = separate_keys(schema_keys)

        assert exact == {}
        assert pattern == {}
        assert compiled == {}

    def test_compiled_patterns_functional(self):
        """Test that compiled patterns work correctly."""
        schema_keys = {"x_*": 1, "{y_\\d+}": 2}
        exact, pattern, compiled = separate_keys(schema_keys)

        # Test glob pattern
        assert compiled["x_*"].fullmatch("x_foo")
        assert compiled["x_*"].fullmatch("x_0")
        assert not compiled["x_*"].fullmatch("y_foo")

        # Test regex pattern
        assert compiled["{y_\\d+}"].fullmatch("y_0")
        assert compiled["{y_\\d+}"].fullmatch("y_123")
        assert not compiled["{y_\\d+}"].fullmatch("y_foo")


@pytest.mark.parametrize(
    "actual_keys, exact_keys, pattern_specs, expected",
    [
        # All exact matches
        ({"x": 1, "y": 2, "z": 3}, {"x": 1, "y": 2}, {}, {"x", "y"}),
        # All pattern matches
        ({"x_0": 1, "x_1": 2, "y_foo": 3}, {}, {"x_*": "x_*"}, {"x_0", "x_1"}),
        # Mixed matches
        (
            {"x": 1, "y_0": 2, "y_1": 3, "z": 4},
            {"x": 1},
            {"y_*": "y_*"},
            {"x", "y_0", "y_1"},
        ),
        # No matches
        ({"a": 1, "b": 2}, {"x": 1}, {"y_*": "y_*"}, set()),
        # Regex pattern matches
        (
            {"x_0": 1, "x_1": 2, "x_foo": 3},
            {},
            {"{x_\\d+}": "{x_\\d+}"},
            {"x_0", "x_1"},
        ),
        # Multiple patterns
        (
            {"x_0": 1, "y_foo": 2, "z_1": 3},
            {},
            {"{x_\\d+}": "{x_\\d+}", "y_*": "y_*"},
            {"x_0", "y_foo"},
        ),
        # Exact takes precedence
        ({"x_0": 1, "x_1": 2}, {"x_0": 1}, {"x_*": "x_*"}, {"x_0", "x_1"}),
        # Empty actual keys
        ({}, {"x": 1}, {"y_*": "y_*"}, set()),
    ],
    ids=[
        "all_exact_matches",
        "all_pattern_matches",
        "mixed_matches",
        "no_matches",
        "regex_pattern_matches",
        "multiple_patterns",
        "exact_takes_precedence",
        "empty_actual_keys",
    ],
)
def test_find_matched_keys(actual_keys, exact_keys, pattern_specs, expected):
    """Test finding keys that match either exact or pattern keys."""
    # Convert pattern specs to compiled patterns
    compiled_patterns = {k: pattern_to_regex(v) for k, v in pattern_specs.items()}
    matched = find_matched_keys(actual_keys, exact_keys, compiled_patterns)
    assert matched == expected
