from __future__ import annotations

from reson.types import EmgSample


def parse_line(line: str) -> EmgSample | None:
    text = line.strip()
    if not text:
        return None

    parts = text.split()
    if len(parts) != 3:
        return None

    try:
        t_ms = int(parts[0])
        raw = int(parts[1])
        env = int(parts[2])
    except ValueError:
        return None

    return EmgSample(t_ms=t_ms, raw=raw, env=env)
