#!/usr/bin/env python3
"""Ingest the Pi's pi_a53_results.json into the evidence bundle (run on the WORKSTATION).

Usage: python sandbox/pi/ingest_pi_results.py /path/to/pi_a53_results.json
Writes evidence/sandbox/pi_a53_gate.json; then `python -m evaluation.assemble_evidence`
picks it up (the Kind 2 / sandbox sections gain measured real-A53 numbers).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DST = Path("evidence/sandbox/pi_a53_gate.json")


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    src = Path(sys.argv[1])
    d = json.loads(src.read_text())
    for key in ("controller_shield", "commander_int4_gguf"):
        if key not in d:
            raise SystemExit(f"malformed results: missing '{key}'")
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, DST)
    ctrl = d["controller_shield"]
    cmdr = d["commander_int4_gguf"]
    print(f"Ingested → {DST}")
    print(f"  CPU: {d.get('cpu_model', '?')}")
    print(
        f"  controller+shield: {ctrl.get('ms_per_cycle', '?')} ms/cycle "
        f"(≤100 ms: {ctrl.get('latency_gate_100ms_pass', '?')}), "
        f"peak RSS {ctrl.get('peak_rss_mb', '?')} MB"
    )
    print(
        f"  commander SmolLM2-360M int5: {cmdr.get('tokens_per_second', '?')} tok/s, "
        f"goal-gen {cmdr.get('goal_gen_latency_s', '?')} s, peak RSS "
        f"{cmdr.get('peak_rss_mb', '?')} MB (≤512 MB: {cmdr.get('fits_512mb', '?')})"
    )
    pw = d.get("power", {})
    print(
        f"  power [{pw.get('backend', '?')}]: peak {pw.get('power_w_peak', '?')} W, "
        f"core-attributable {pw.get('power_w_core_attributable', '?')} W "
        f"(idle {pw.get('power_w_idle', '?')} W)"
    )
    print("Next: python -m evaluation.assemble_evidence")


if __name__ == "__main__":
    main()
