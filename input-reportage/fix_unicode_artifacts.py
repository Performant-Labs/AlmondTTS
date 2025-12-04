from __future__ import annotations

"""
Utility to fix stray numeric unicode entity fragments (e.g., ``x2014``) in text files.
Scans all ``*.txt`` files in the current working directory and rewrites them in place.
"""

from pathlib import Path
import re

# Map of common mis-encoded hex codes to their intended characters.
ARTIFACT_MAP = {
    "2013": "–",  # en dash
    "2014": "—",  # em dash
    "2018": "‘",  # left single quote
    "2019": "’",  # right single quote / apostrophe
    "201c": "“",  # left double quote
    "201d": "”",  # right double quote
    "2026": "…",  # ellipsis
}

# Matches fragments like "x2014" or "x201D;".
PATTERN = re.compile(r"x([0-9a-fA-F]{4});?")


def fix_text(text: str) -> str:
    def _repl(match: re.Match[str]) -> str:
        code = match.group(1).lower()
        return ARTIFACT_MAP.get(code, match.group(0))

    return PATTERN.sub(_repl, text)


def main() -> None:
    for path in Path(".").glob("*.txt"):
        original = path.read_text()
        fixed = fix_text(original)
        if fixed != original:
            path.write_text(fixed)
            print(f"Rewrote {path}")


if __name__ == "__main__":
    main()
