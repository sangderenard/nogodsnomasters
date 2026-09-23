"""THE TIME FIELD -- moved to ``turing/src/common/dt_system/time_field.py``.

It was staged here so it could be exercised against a real machine graph
before moving into the dt system, which is universal and serves every sim.
It has moved (2026-09-22). This module re-exports it unchanged so every
engine_toy caller keeps its import; the definition lives in one place.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.common.dt_system.time_field import (  # noqa: E402,F401
    ADAPTORS,
    StoreLedger,
    TimeAdaptor,
    TimeField,
    TimeFieldConfig,
    TimeForceStore,
    adaptor_for,
    time_angle,
)
