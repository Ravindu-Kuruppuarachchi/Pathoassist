"""Append-only, tamper-evident audit log (JSON lines, SHA-256 hash chain).

Every entry stores the hash of the previous entry; editing or deleting any line breaks the chain,
which verify() detects. The final hash is printed on the report so a report can be matched to its log.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path, user: str | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.user = user or getpass.getuser()
        self.last_hash = GENESIS
        if self.path.exists():
            lines = [l for l in self.path.read_text().splitlines() if l.strip()]
            if lines:
                self.last_hash = json.loads(lines[-1])["hash"]

    def log(self, event: str, **details) -> dict:
        e = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), user=self.user, event=event, details=details, prev=self.last_hash)
        e["hash"] = _digest(e)
        with open(self.path, "a") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self.last_hash = e["hash"]
        return e

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def verify(self) -> tuple[bool, str]:
        prev = GENESIS
        for i, e in enumerate(self.entries()):
            if e.get("prev") != prev:
                return False, f"entry {i + 1}: chain broken (entry removed or reordered)"
            if _digest(e) != e.get("hash"):
                return False, f"entry {i + 1}: content changed after it was written"
            prev = e["hash"]
        return True, "ok"
