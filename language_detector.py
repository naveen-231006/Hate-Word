"""
language_detector.py — Unicode-Based Language Detection for Dravidian Languages
=================================================================================
Detects whether input text is Tamil, Malayalam, Kannada, or Latin-script (English/Romanized).
Uses Unicode range analysis — zero-dependency, fast, and accurate for Dravidian scripts.

Architecture:
    Input → Language Detection → Language-Specific Ensemble → Prediction
"""

import re
from collections import Counter

# Unicode ranges for Dravidian scripts
UNICODE_RANGES = {
    "tamil":     (0x0B80, 0x0BFF),
    "malayalam": (0x0D00, 0x0D7F),
    "kannada":   (0x0C80, 0x0CFF),
}

# Additional script ranges for exclusion / identification
DEVANAGARI_RANGE = (0x0900, 0x097F)   # Hindi
TELUGU_RANGE     = (0x0C00, 0x0C7F)   # Telugu
LATIN_RANGE      = (0x0041, 0x007A)   # Basic Latin A-z


def _char_script(ch):
    """Identify which script a character belongs to."""
    cp = ord(ch)
    for lang, (lo, hi) in UNICODE_RANGES.items():
        if lo <= cp <= hi:
            return lang
    if DEVANAGARI_RANGE[0] <= cp <= DEVANAGARI_RANGE[1]:
        return "devanagari"
    if TELUGU_RANGE[0] <= cp <= TELUGU_RANGE[1]:
        return "telugu"
    if LATIN_RANGE[0] <= cp <= LATIN_RANGE[1]:
        return "latin"
    return "other"


def detect_language(text):
    """
    Detect the primary language/script of a text.

    Returns:
        dict with keys:
            - language: "tamil" | "malayalam" | "kannada" | "latin" | "mixed" | "unknown"
            - confidence: float 0.0-1.0
            - script_counts: Counter of detected scripts
            - is_code_mixed: bool — True if multiple Dravidian scripts or Dravidian+Latin
    """
    if not text or not text.strip():
        return {
            "language": "unknown",
            "confidence": 0.0,
            "script_counts": Counter(),
            "is_code_mixed": False,
        }

    # Count characters by script
    script_counts = Counter()
    for ch in text:
        if ch.isspace() or ch.isdigit():
            continue
        script = _char_script(ch)
        if script != "other":
            script_counts[script] += 1

    total = sum(script_counts.values())
    if total == 0:
        return {
            "language": "unknown",
            "confidence": 0.0,
            "script_counts": script_counts,
            "is_code_mixed": False,
        }

    # Find dominant script
    dominant_script, dominant_count = script_counts.most_common(1)[0]
    confidence = dominant_count / total

    # Check for code-mixing
    dravidian_scripts = {s for s in script_counts if s in UNICODE_RANGES}
    has_latin = "latin" in script_counts
    is_code_mixed = (len(dravidian_scripts) > 1) or \
                    (len(dravidian_scripts) == 1 and has_latin and
                     script_counts["latin"] > 0.2 * total)

    # Determine language
    if dominant_script in UNICODE_RANGES:
        language = dominant_script
    elif dominant_script == "latin":
        # Pure Latin text — could be romanized Tamil/Malayalam/Kannada or English
        # We can't distinguish without n-gram models, so return "latin"
        language = "latin"
    else:
        language = "unknown"

    return {
        "language": language,
        "confidence": round(confidence, 3),
        "script_counts": dict(script_counts),
        "is_code_mixed": is_code_mixed,
    }


def detect_language_simple(text):
    """Simplified version — returns just the language string."""
    result = detect_language(text)
    return result["language"]


def route_to_model(text, available_models=None):
    """
    Route input text to the appropriate language-specific model.

    Args:
        text: input text
        available_models: dict of available model keys (e.g., {"tamil": ..., "malayalam": ...})

    Returns:
        recommended language key for model selection
    """
    if available_models is None:
        available_models = {"tamil", "malayalam", "kannada"}

    result = detect_language(text)
    lang = result["language"]

    # If we detected a specific Dravidian language and have a model for it
    if lang in available_models:
        return lang

    # If Latin script (romanized), default to Tamil (largest training set)
    # since most Tanglish content is Tamil
    if lang == "latin":
        return "tamil"

    # Fallback: use the model with the largest training set
    return "tamil"


# ──────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("தமிழ் மொழி அழகான மொழி", "tamil"),
        ("Padam vera level mass iruku", "latin"),
        ("ഈ സിനിമ കിടിലൻ ആണ്", "malayalam"),
        ("ಸೂಪರ್ ಗುರು ಚೆನ್ನಾಗಿ ಇದೆ", "kannada"),
        ("nalla iruku bro super acting", "latin"),
        ("enna da ivan oru waste fellow", "latin"),
        ("Ikka polikumen urappullavar like", "latin"),
        ("ಮಸ್ತ್ ಇದೆ ಈ ಹಾಡು super song", "kannada"),
        ("", "unknown"),
    ]

    print("Language Detection Tests:")
    print("-" * 70)
    for text, expected in test_cases:
        result = detect_language(text)
        status = "OK" if result["language"] == expected else "FAIL"
        text_preview = text[:50] if text else "(empty)"
        print(f"  {status:>4s} [{result['language']:>10s}] (conf={result['confidence']:.2f}, "
              f"mixed={result['is_code_mixed']}) | {text_preview}")

    print(f"\n  Routing test:")
    for text, _ in test_cases[:6]:
        routed = route_to_model(text)
        print(f"    → {routed:>10s} | {text[:50]}")
