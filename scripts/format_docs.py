#!/usr/bin/env python3
"""Wrap README.md and START_HERE.txt to 80 columns."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START_HERE = ROOT / "START_HERE.txt"


def _wrap_list_line(prefix: str, text: str, width: int, indent: str) -> list[str]:
    wrapped = fill(text, width=width, subsequent_indent=indent)
    parts = wrapped.split("\n")
    lines = [prefix + parts[0]]
    for part in parts[1:]:
        lines.append(indent + part)
    return lines


def format_readme() -> None:
    lines: list[str] = []
    in_code = False
    for raw in README.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            lines.append(line)
            continue
        if in_code or line.startswith("#") or not line.strip():
            lines.append(line)
            continue
        if line.startswith("- "):
            bullet = _wrap_list_line("- ", line[2:].lstrip(), width=78, indent="  ")
            lines.extend(bullet)
            continue
        lines.append(fill(line, width=80))
    README.write_text("\n".join(lines) + "\n")


def format_start_here() -> None:
    def should_skip(l: str) -> bool:
        return (
            not l.strip()
            or l.startswith("---")
            or l.startswith("===")
            or l.startswith("   ")
            or l.startswith("      ")
            or l.startswith("```")
            or l.startswith("<")
        )

    lines: list[str] = []
    for raw in START_HERE.read_text().splitlines():
        line = raw.rstrip()
        if should_skip(line):
            lines.append(line)
            continue
        if line.startswith("•"):
            bullet = _wrap_list_line("• ", line[1:].lstrip(), width=78, indent="  ")
            lines.extend(bullet)
            continue
        if line.startswith("- "):
            bullet = _wrap_list_line("- ", line[2:].lstrip(), width=78, indent="  ")
            lines.extend(bullet)
            continue
        if line.startswith("   -"):
            bullet = _wrap_list_line("   - ", line[4:].lstrip(), width=74, indent="     ")
            lines.extend(bullet)
            continue
        if len(line) > 2 and line[0].isdigit() and line[1] == ")":
            body = line[2:].lstrip()
            formatted = _wrap_list_line(f"{line[:2]} ", body, width=78, indent="   ")
            lines.extend(formatted)
            continue
        lines.append(fill(line, width=80))
    START_HERE.write_text("\n".join(lines) + "\n")


def main() -> None:
    format_readme()
    format_start_here()


if __name__ == "__main__":
    main()
