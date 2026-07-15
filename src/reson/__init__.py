"""Public API for the Reson EMG switch package."""

from reson.api import ResonSerialConfig, ResonSwitch, SwitchUpdate, iter_serial_switch_updates
from reson.binary_model import load_binary_profile
from reson.types import EmgSample, SwitchEvent

__all__ = [
    "EmgSample",
    "ResonSerialConfig",
    "ResonSwitch",
    "SwitchEvent",
    "SwitchUpdate",
    "__version__",
    "iter_serial_switch_updates",
    "load_binary_profile",
]
__version__ = "0.1.0"
