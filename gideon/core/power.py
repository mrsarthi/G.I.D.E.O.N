"""
G.I.D.E.O.N Power State

On this hardware (GTX 1650 Ti laptop) the battery cannot supply the GPU's
draw while the chat model runs on it: the machine powers off. Everything that
decides where the model runs asks here first.
"""

from __future__ import annotations

import psutil


def on_battery() -> bool:
    """True when running from the battery. Desktops (no battery) are mains."""
    try:
        battery = psutil.sensors_battery()
    except Exception:
        return False
    return battery is not None and not battery.power_plugged


def battery_percent() -> int | None:
    try:
        battery = psutil.sensors_battery()
    except Exception:
        return None
    return None if battery is None else int(battery.percent)
