"""Small, explicit JSON records; no pickle and no implicit overwrite."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A sibling temporary file allows replacement without a partially written manifest.
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}; preserve this file and repair the incomplete record before resuming") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical(row) + "\n")

