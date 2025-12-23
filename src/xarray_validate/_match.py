"""Pattern matching support functions."""

import fnmatch
import re
from typing import Any, Dict, Mapping, Set, Tuple


def is_regex_pattern(key: str) -> bool:
    """Check if a key is a regex pattern (enclosed in curly braces)."""
    return key.startswith("{") and key.endswith("}")


def is_glob_pattern(key: str) -> bool:
    """Check if a key is a glob pattern (contains * or ?)."""
    return "*" in key or "?" in key


def is_pattern_key(key: str) -> bool:
    """Check if a key is any kind of pattern (glob or regex)."""
    return is_glob_pattern(key) or is_regex_pattern(key)


def pattern_to_regex(pattern: str) -> re.Pattern:
    r"""
    Convert a pattern key to a compiled regex.

    Supports two pattern types:

    - glob patterns: ``'x_*'`` matches ``x_0``, ``x_1``, ``x_foo``, etc.
    - regex patterns: ``'{x_\\d+}'`` matches ``x_0``, ``x_1``, but not ``x_foo``

    Parameters
    ----------
    pattern : str
        The pattern string (regex in curly braces or glob).

    Returns
    -------
    re.Pattern
        Compiled regex pattern
    """
    if is_regex_pattern(pattern):
        # Remove curly braces and compile as regex
        regex_str = pattern[1:-1]
        return re.compile(regex_str)

    elif is_glob_pattern(pattern):
        # Convert glob to regex
        regex_str = fnmatch.translate(pattern)
        return re.compile(regex_str)

    else:
        # Exact match
        return re.compile(re.escape(pattern) + "$")


def separate_keys(
    schema_keys: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, re.Pattern]]:
    """
    Separate schema keys into exact and pattern keys, and compile patterns.

    Parameters
    ----------
    schema_keys : dict
        Dictionary with string keys (exact or pattern) and schema values.

    Returns
    -------
    exact_keys : dict
        Dictionary with exact (non-pattern) keys.

    pattern_keys : dict
        Dictionary with pattern keys.

    compiled_patterns : dict
        Dictionary mapping pattern keys to compiled regex objects.
    """
    exact_keys = {k: v for k, v in schema_keys.items() if not is_pattern_key(k)}
    pattern_keys = {k: v for k, v in schema_keys.items() if is_pattern_key(k)}
    compiled_patterns = {k: pattern_to_regex(k) for k in pattern_keys}
    return exact_keys, pattern_keys, compiled_patterns


def find_matched_keys(
    actual_keys: Mapping[str, Any],
    exact_keys: Dict[str, Any],
    compiled_patterns: Dict[str, re.Pattern],
) -> Set[str]:
    """
    Find all actual keys that match either exact or pattern keys.

    Parameters
    ----------
    actual_keys : mapping
        The actual keys to check (*e.g.* ``coords`` or ``data_vars``).

    exact_keys : dict
        Dictionary with exact (non-pattern) keys.

    compiled_patterns : dict
        Dictionary mapping pattern keys to compiled regex objects.

    Returns
    -------
    set
        Set of actual keys that match either exact or pattern keys.
    """
    matched_keys = set()
    for key_name in actual_keys:
        # Check exact match
        if key_name in exact_keys:
            matched_keys.add(key_name)
            continue
        # Check pattern match
        for pattern, regex in compiled_patterns.items():
            if regex.fullmatch(key_name):
                matched_keys.add(key_name)
                break
    return matched_keys
