"""Tolerant reader for research/experiment-log.jsonl (local or retrieved remote copies).

The protocol warns that one box's ledger has a corrupt NUL region; a parse stop is never treated as
completion. We strip NULs, skip unparsable lines (counted), and keep the latest row per label.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

FLAG_TOKEN_RE = re.compile(r"(?<!\S)(--[a-z0-9][a-z0-9-]*)(?:[= ]([^\s]+))?")


@dataclass
class Ledger:
    rows: list[dict] = field(default_factory=list)
    skipped_lines: int = 0
    source: str = ""

    @classmethod
    def load(cls, path: Path) -> "Ledger":
        ledger = cls(source=str(path))
        if not path.is_file():
            return ledger
        text = path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace")
        return cls.parse(text, source=str(path))

    @classmethod
    def parse(cls, text: str, source: str = "") -> "Ledger":
        ledger = cls(source=source)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                ledger.skipped_lines += 1
                continue
            if isinstance(row, dict) and "label" in row:
                ledger.rows.append(row)
            else:
                ledger.skipped_lines += 1
        return ledger

    def labels(self) -> set[str]:
        return {str(r["label"]) for r in self.rows}

    def latest_by_label(self, label: str) -> dict | None:
        matches = [r for r in self.rows if r.get("label") == label]
        if not matches:
            return None
        return max(matches, key=lambda r: str(r.get("timestamp", "")))

    def rows_with_flags(self, flags: list[str], commit_prefix: str | None = None) -> list[dict]:
        """Rows whose launch command contains every given flag token (order-insensitive)."""
        wanted = set(flags)
        out = []
        for row in self.rows:
            cmd = str(row.get("launch_command", ""))
            present = {m.group(1) for m in FLAG_TOKEN_RE.finditer(cmd)}
            if wanted and wanted <= present:
                if commit_prefix is None or str(row.get("git_commit", "")).startswith(commit_prefix[:7]):
                    out.append(row)
        return out


def extract_flags(launch_command: str) -> list[str]:
    return sorted({m.group(1) for m in FLAG_TOKEN_RE.finditer(launch_command)})
