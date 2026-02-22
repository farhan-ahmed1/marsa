"""Input sanitization utilities to mitigate prompt injection attacks.

These helpers strip or flag potentially harmful content from user-supplied
text before it reaches the LLM prompt templates.

Strategy:
- Remove common prompt-injection patterns (system/instruction overrides).
- Strip invisible Unicode control characters.
- Enforce max length.
- Log sanitization events for audit.
"""

import re
import unicodedata

import structlog

logger = structlog.get_logger(__name__)

# Max allowed query length after sanitization
MAX_QUERY_LENGTH = 2000

# Patterns that attempt to override system instructions
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|rules?)\b"),
    re.compile(r"(?i)\byou\s+are\s+now\b"),
    re.compile(r"(?i)\bsystem\s*:\s*"),
    re.compile(r"(?i)\b(assistant|user|human)\s*:\s*"),
    re.compile(r"(?i)\bnew\s+instructions?\s*:\s*"),
    re.compile(r"(?i)\bdo\s+not\s+follow\b.*\binstructions?\b"),
    re.compile(r"(?i)\bforget\s+(everything|all)\b"),
    re.compile(r"(?i)<\|?(system|im_start|im_end)\|?>"),
    re.compile(r"(?i)\[INST\]"),
    re.compile(r"(?i)```\s*(system|instruction)"),
]


def _strip_control_chars(text: str) -> str:
    """Remove invisible Unicode control characters (except newline/tab)."""
    return "".join(
        ch
        for ch in text
        if ch in ("\n", "\t", "\r")
        or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def _detect_injection(text: str) -> list[str]:
    """Return list of matched injection pattern descriptions."""
    matches: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def sanitize_query(query: str) -> str:
    """Sanitize a user query, stripping dangerous patterns.

    Args:
        query: Raw user input.

    Returns:
        Cleaned query string.

    Raises:
        ValueError: If the query is empty after sanitization or exceeds
            the max length.
    """
    original_length = len(query)

    # Strip control characters
    query = _strip_control_chars(query)

    # Trim whitespace
    query = query.strip()

    if not query:
        raise ValueError("Query is empty after sanitization")

    # Detect injection attempts (log but don't block - just redact)
    injections = _detect_injection(query)
    if injections:
        logger.warning(
            "prompt_injection_detected",
            pattern_count=len(injections),
            patterns=injections[:3],  # only log first 3
            query_preview=query[:80],
        )
        # Redact matched patterns
        for pattern in _INJECTION_PATTERNS:
            query = pattern.sub("[REDACTED]", query)
        query = query.strip()

    # Enforce max length
    if len(query) > MAX_QUERY_LENGTH:
        logger.warning(
            "query_truncated",
            original_length=original_length,
            max_length=MAX_QUERY_LENGTH,
        )
        query = query[:MAX_QUERY_LENGTH]

    return query


def sanitize_feedback(text: str, max_length: int = 2000) -> str:
    """Sanitize HITL feedback text.

    Args:
        text: Raw feedback from the user.
        max_length: Cap for the returned string.

    Returns:
        Cleaned feedback string.
    """
    text = _strip_control_chars(text)
    text = text.strip()

    injections = _detect_injection(text)
    if injections:
        logger.warning(
            "feedback_injection_detected",
            pattern_count=len(injections),
        )
        for pattern in _INJECTION_PATTERNS:
            text = pattern.sub("[REDACTED]", text)
        text = text.strip()

    return text[:max_length]
