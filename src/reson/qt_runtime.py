from __future__ import annotations

import os
from pathlib import Path


def validate_python_runtime() -> None:
    import sys

    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "Unsupported Python runtime detected (>=3.14). "
            "Recreate .venv with Python 3.11 or 3.12."
        )


def get_qt_plugin_dir() -> str | None:
    try:
        import PySide6
    except Exception:
        return None
    plugins_dir = Path(PySide6.__file__).resolve().parent / "Qt" / "plugins"
    return str(plugins_dir) if plugins_dir.exists() else None


def configure_qt_platform_plugin_path() -> None:
    try:
        import PySide6
    except Exception:
        return

    pyside_root = Path(PySide6.__file__).resolve().parent
    plugins_dir = pyside_root / "Qt" / "plugins"

    # Conda/other Qt installs can leak plugin/framework paths and break PySide6.
    for var in ("QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH"):
        os.environ.pop(var, None)

    for var in ("DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH"):
        value = os.environ.get(var, "")
        if "anaconda" in value.lower() or "conda" in value.lower():
            os.environ.pop(var, None)

    # Force UTF-8 locale so Qt does not start in fallback C locale.
    os.environ["LC_ALL"] = "en_US.UTF-8"
    os.environ["LANG"] = "en_US.UTF-8"
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")

    if plugins_dir.exists():
        os.environ["QT_PLUGIN_PATH"] = str(plugins_dir)
