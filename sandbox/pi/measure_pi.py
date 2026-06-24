#!/usr/bin/env python3
"""Parse the run_on_pi.sh logs into pi_a53_results.json (runs ON THE PI).

Stdlib only (no torch/transformers on the Pi). Reads:
  flight_loop.json / flight_loop.time  — controller+shield cycle + RSS
  commander.out / commander.time       — llama.cpp tok/s + RSS
  /proc/cpuinfo                         — the actual A-class core
Emits the results JSON to stdout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GOAL_TOKENS = 48  # representative PDDL+ goal length (the commander's output)


def _read(p: str) -> str:
    fp = Path(p)
    return fp.read_text(errors="ignore") if fp.exists() else ""


def _rss_mb(time_text: str) -> float | None:
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_text)
    return round(int(m.group(1)) / 1024.0, 1) if m else None


def _cpu_model() -> str:
    info = _read("/proc/cpuinfo")
    for key in ("Model name", "model name", "Hardware", "Model"):
        m = re.search(rf"{key}\s*:\s*(.+)", info)
        if m:
            return m.group(1).strip()
    return "unknown"


def _power() -> dict:
    """Combine the auto-sampled idle baseline + during-run power (measure_power.py).
    Core-attributable = peak during the run minus idle mean."""
    idle = json.loads(_read("power_idle.json") or "{}")
    run = json.loads(_read("power_run.json") or "{}")
    if run.get("backend", "manual") == "manual" or "peak_w" not in run:
        return {
            "backend": "manual",
            "power_w_peak": None,
            "note": "no automated power source detected; read a USB meter and set power_w_peak",
        }
    idle_w, peak_w = idle.get("mean_w"), run.get("peak_w")
    core = round(peak_w - idle_w, 3) if (peak_w is not None and idle_w is not None) else None
    return {
        "backend": run.get("backend"),
        "power_w_idle": idle_w,
        "power_w_peak": peak_w,
        "power_w_mean_run": run.get("mean_w"),
        "power_w_core_attributable": core,
        "note": "core-attributable = peak(run) - mean(idle); board-wide draw is higher.",
    }


def main() -> None:
    # --- controller + shield ---
    ctrl: dict = {}
    try:
        fl = json.loads(_read("flight_loop.json") or "{}")
        ns = fl.get("ns_per_iter")
        ctrl = {
            "ns_per_cycle": ns,
            "ms_per_cycle": round(ns / 1e6, 5) if ns else None,
            "shield_violations": fl.get("shield_violations"),
            "n_iters": fl.get("n_iters"),
            "checksum": fl.get("checksum"),
            "peak_rss_mb": _rss_mb(_read("flight_loop.time")),
            "latency_gate_100ms_pass": (ns is not None and ns / 1e6 < 100.0),
        }
    except Exception as e:  # noqa: BLE001
        ctrl = {"error": str(e)}

    # --- commander (llama.cpp) ---
    # Take the GENERATION eval rate ("eval time"), not prompt eval ("prompt eval
    # time") — both print "tokens per second". Works for old (llama_print_timings)
    # and new (common_perf_print) formats; falls back to any t/s figure.
    out = _read("commander.out") + _read("commander.time")
    tok_s = mspt_v = None
    for line in out.splitlines():
        if "eval time" in line and "prompt" not in line:
            m_ts = re.search(r"([\d.]+)\s*tokens per second", line)
            m_mp = re.search(r"([\d.]+)\s*ms per token", line)
            if m_ts:
                tok_s = float(m_ts.group(1))
            if m_mp:
                mspt_v = float(m_mp.group(1))
    if tok_s is None:
        m = re.search(r"([\d.]+)\s*tokens per second", out)
        tok_s = float(m.group(1)) if m else None
    cmdr = {
        "tokens_per_second": tok_s,
        "ms_per_token": mspt_v,
        "goal_gen_latency_s": round(GOAL_TOKENS / tok_s, 2) if tok_s else None,
        "goal_tokens_assumed": GOAL_TOKENS,
        "peak_rss_mb": _rss_mb(_read("commander.time")),
    }
    cmdr["fits_512mb"] = cmdr["peak_rss_mb"] is not None and cmdr["peak_rss_mb"] <= 512

    result = {
        "artifact": "real ARM Cortex-A sandbox (Raspberry Pi)",
        "cpu_model": _cpu_model(),
        "controller_shield": ctrl,
        "commander_int4_gguf": cmdr,
        "power": _power(),  # auto-sampled (Pi-5 PMIC / INA219 / net); manual fallback
        "notes": "Controller+shield is the 10 Hz real-time tier (≤100 ms gate). The commander "
        "runs at mission-event cadence (seconds) — its tok/s sets goal-generation latency, NOT "
        "the 100 ms controller budget. Cycle-accurate, real hardware (closes the QEMU caveat).",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
