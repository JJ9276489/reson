# Contributing to Reson

Thanks for your interest in Reson. Contributions of all kinds are welcome:
bug reports, hardware reports, data-collection feedback, documentation
fixes, and code.

## Reporting issues

Please open a GitHub issue. For bugs, include:

- What you ran (command line, app, firmware revision).
- What you expected and what happened instead.
- Hardware setup if relevant (board, sensor, electrode placement).
- Logs or session metadata (`meta.json`) where applicable.

Reports from real hardware setups are especially valuable at this stage,
even when nothing is broken — see `docs/data_collection_protocol.md`.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The test suite runs against synthetic fixtures and test doubles; no
hardware is required.

## Pull requests

- Keep changes focused; one concern per PR.
- Add or update tests for behavior changes.
- Make sure `pytest` passes locally; CI runs the same suite.
- Match the existing code style and documentation tone. In particular,
  keep claims about signal quality or control performance consistent
  with `docs/validation_status.md` — implemented and validated are
  documented separately in this project.

## Scope

Reson's current product boundary is deliberately narrow: a reliable
binary intent signal from a single EMG channel. Features outside that
boundary (multi-channel support, cursor control, new sensors) are worth
discussing in an issue before writing code.
