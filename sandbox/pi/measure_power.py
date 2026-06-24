"""Automated power sampling on the Pi (no manual USB-meter reading).

Auto-detects a backend, in priority order:
  1. Pi 5 on-board PMIC  -- `vcgencmd pmic_read_adc` (sum of per-rail V*I). No extra
     hardware; works on a Raspberry Pi 5.
  2. INA219 / INA260     -- an I2C current/power sensor on the supply (pip:
     pi-ina219). For Pi 4 and earlier, which lack on-board current sensing.
  3. Network meter       -- a smart plug / bench PSU exposing watts over HTTP. Set
     AMPLE_POWER_URL (and optionally AMPLE_POWER_JSON_KEY, dotted path) to enable.

Modes:
  sample  --seconds N --out F   sample for N s, write a summary JSON to F.
  monitor --out F               sample until SIGINT/SIGTERM, then write the summary
                                (run_on_pi.sh runs this in the background during the
                                workload and kills it when done).

The summary is {backend, mean_w, peak_w, min_w, n, hz}. run_on_pi.sh takes an idle
baseline + a during-run sample; the **core-attributable** power is peak_run -
mean_idle. If no backend is available it writes backend="manual" so the operator
knows to enter a USB-meter reading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen


def _pmic() -> float | None:
    """Pi 5 PMIC: sum V*I over rails that report both a *_V and a *_A reading."""
    try:
        out = subprocess.run(
            ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=2
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    volts: dict[str, float] = {}
    amps: dict[str, float] = {}
    for m in re.finditer(r"(\w+?)_([VA])\s+\w+\([0-9]+\)=([0-9.]+)", out):
        base, kind, val = m.group(1), m.group(2), float(m.group(3))
        (volts if kind == "V" else amps)[base] = val
    if not volts or not amps:
        return None
    p = sum(volts[b] * amps[b] for b in volts.keys() & amps.keys())
    return p if p > 0 else None


def _ina219() -> float | None:
    """INA219 I2C sensor (pip install pi-ina219). Bus voltage * current."""
    try:
        from ina219 import INA219  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        ina = INA219(shunt_ohms=0.1)
        ina.configure()
        return float(ina.voltage() * ina.current() / 1000.0)  # V * (mA->A)
    except Exception:  # noqa: BLE001 — sensor not wired
        return None


def _network() -> float | None:
    url = os.environ.get("AMPLE_POWER_URL")
    if not url:
        return None
    try:
        body = urlopen(url, timeout=2).read().decode()  # noqa: S310 — operator-set URL
    except Exception:  # noqa: BLE001
        return None
    key = os.environ.get("AMPLE_POWER_JSON_KEY")
    if key:
        try:
            obj = json.loads(body)
            for part in key.split("."):
                obj = obj[part]
            return float(obj)
        except Exception:  # noqa: BLE001
            return None
    m = re.search(r"[-+]?\d*\.?\d+", body)
    return float(m.group(0)) if m else None


def detect_backend() -> tuple[str, Callable[[], float | None]]:
    for name, fn in (("pi5_pmic", _pmic), ("ina219", _ina219), ("network", _network)):
        if fn() is not None:
            return name, fn
    return "manual", lambda: None


def sample(seconds: float, hz: float = 5.0) -> dict:
    name, fn = detect_backend()
    if name == "manual":
        return {"backend": "manual", "note": "no automated power source; read a USB meter"}
    vals: list[float] = []
    period = 1.0 / hz
    stop = time.monotonic() + seconds
    while time.monotonic() < stop:
        w = fn()
        if w is not None:
            vals.append(w)
        time.sleep(period)
    return _summary(name, vals)


def _summary(name: str, vals: list[float]) -> dict:
    if not vals:
        return {"backend": name, "note": "no samples"}
    return {
        "backend": name,
        "mean_w": round(sum(vals) / len(vals), 3),
        "peak_w": round(max(vals), 3),
        "min_w": round(min(vals), 3),
        "n": len(vals),
    }


def monitor(out: Path, hz: float = 5.0) -> None:
    name, fn = detect_backend()
    vals: list[float] = []
    running = {"go": True}

    def _stop(*_: object) -> None:
        running["go"] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    period = 1.0 / hz
    while running["go"]:
        if name != "manual":
            w = fn()
            if w is not None:
                vals.append(w)
        time.sleep(period)
    out.write_text(json.dumps(_summary(name, vals) if name != "manual" else {"backend": "manual"}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--seconds", type=float, default=8.0)
    s.add_argument("--out", required=True)
    m = sub.add_parser("monitor")
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "sample":
        Path(a.out).write_text(json.dumps(sample(a.seconds)))
    else:
        monitor(Path(a.out))


if __name__ == "__main__":
    main()
