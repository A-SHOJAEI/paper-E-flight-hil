"""Air-bearing ADCS hardware loop — the deployed RMA controller on a real (or
simulated) 3-DOF platform, with physical-equivalent fault injection.

The control tier is unchanged from flight: it reuses the SAME RMA controller that
sandbox/qemu validates bit-faithful to Python. Only the Sensor/Actuator backend
swaps between SimBackend (a Basilisk rigid-body plant — runs anywhere) and
HwBackend (BNO055 IMU + reaction-wheel ESCs on the Pi).

Fault injection maps one-to-one onto our model (applied = command*g + b):
  --fault sign:0            wheel-0 polarity reversed (g0 = -1)
  --fault gain:1:0.5        wheel-1 at 50% effectiveness
  --fault bias:2:0.2        wheel-2 constant +0.2 offset
  --fault loss:0            wheel-0 dead (g0 = 0; the shield-only regime)
  (comma-separate to combine; physical faults: set --fault none and reverse leads.)

Run (anywhere):  python -m sandbox.airbearing.hardware_loop --backend sim --fault sign:0
Run (on the Pi): python -m sandbox.airbearing.hardware_loop --backend hw  --fault sign:0
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np


def parse_faults(spec: str) -> tuple[np.ndarray, np.ndarray]:
    g = np.ones(3, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    if spec and spec != "none":
        for tok in spec.split(","):
            parts = tok.split(":")
            kind, axis = parts[0], int(parts[1])
            if kind == "sign":
                g[axis] = -1.0
            elif kind == "loss":
                g[axis] = 0.0
            elif kind == "gain":
                g[axis] = float(parts[2])
            elif kind == "bias":
                b[axis] = float(parts[2])
            else:
                raise ValueError(f"unknown fault {tok}")
    return g, b


def _pointing_deg(sigma: np.ndarray) -> float:
    return float(np.rad2deg(4.0 * math.atan(np.linalg.norm(sigma))))


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class SimBackend:
    """Basilisk rigid-body plant (same sim as the controller was trained/eval'd on)."""

    def __init__(self, inertia_factor: float, seed: int) -> None:
        from controller.maml.meta_imitation_basilisk import TaskSpec, _task_env
        from controller.rma.rma_attitude import env_cfg

        self.env = _task_env(TaskSpec(inertia_factor, (1, 1, 1)), env_cfg(), seed=seed)
        self.obs, _ = self.env.reset(seed=seed)

    def sense(self) -> np.ndarray:
        return self.obs

    def actuate(self, applied: np.ndarray) -> None:
        self.obs, _, _, _, _ = self.env.step(applied.astype(np.float32))

    def close(self) -> None:
        self.env.close()


class HwBackend:
    """Real platform: BNO055 IMU + reaction-wheel ESCs (Raspberry Pi).

    Fill the two TODO sites for your ESC interface (ODrive/SimpleFOC/PWM). Imports
    the hardware libs lazily so the module still loads (in --backend sim) off-Pi."""

    def __init__(self, max_torque_nm: float) -> None:
        import adafruit_bno055  # type: ignore[import-not-found]
        import board  # type: ignore[import-not-found]

        self.max_torque = max_torque_nm
        self.imu = adafruit_bno055.BNO055_I2C(board.I2C())
        # TODO: open your 3 wheel ESCs here (ODrive: odrive.find_any(); SimpleFOC:
        #       serial.Serial(...); PWM: pigpio.pi()). Store handles on self.

    def sense(self) -> np.ndarray:
        q = self.imu.quaternion  # (w, x, y, z), fused
        w_b = self.imu.gyro  # rad/s, body
        qw = q[0] if q[0] is not None else 1.0
        sigma = np.array([q[1], q[2], q[3]], dtype=np.float32) / (1.0 + qw)  # MRP
        omega = np.array(w_b, dtype=np.float32)
        return np.concatenate([sigma, omega]).astype(np.float32)  # type: ignore[no-any-return]

    def actuate(self, applied: np.ndarray) -> None:
        _ = applied * self.max_torque  # N*m per wheel (reacts -tau onto the body)
        # TODO: command each wheel's torque/current (ODrive input_torque, SimpleFOC
        #       'T<val>', or PWM speed). Enforce momentum/speed limits + desaturate.
        raise NotImplementedError("wire your reaction-wheel ESCs here (see README §3)")

    def close(self) -> None:
        pass


def make_controller(kind: str):  # type: ignore[no-untyped-def]
    from controller.rma.rma_attitude import STUDENT_CKPT, RMAStudent, env_cfg, rma_policy

    if kind == "pd":
        cfg = env_cfg()

        def pd(obs: np.ndarray) -> np.ndarray:  # fault-UNAWARE baseline
            u = (-cfg.kp0 * obs[:3] - cfg.kd0 * obs[3:]) / cfg.max_torque_nm
            return np.clip(u, -1.0, 1.0).astype(np.float32)

        return pd
    import torch

    st = RMAStudent()
    st.load_state_dict(torch.load(STUDENT_CKPT, map_location="cpu", weights_only=True))
    st.eval()
    return rma_policy(st, env_cfg())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "hw"], default="sim")
    ap.add_argument("--controller", choices=["rma", "pd"], default="rma")
    ap.add_argument("--fault", default="none")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--inertia", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--shield", action="store_true", help="run the live LTL shield in the loop")
    ap.add_argument("--out", default="airbearing_run.json")
    a = ap.parse_args()

    g, b = parse_faults(a.fault)
    ctrl = make_controller(a.controller)
    if a.backend == "sim":
        backend: Any = SimBackend(a.inertia, a.seed)
    else:
        from controller.rma.rma_attitude import env_cfg

        backend = HwBackend(env_cfg().max_torque_nm)

    monitor = None
    honest_state = None
    severities: list[str] = []
    if a.shield:  # the SAME live LTLMonitor + honest-state builder as the WS5 software loop
        from shield.monitors.ltl_monitor import LTLMonitor
        from sil.integration.closed_loop import _honest_shield_state

        monitor, honest_state = LTLMonitor(), _honest_shield_state

    best = 180.0
    traj = []
    for _ in range(a.steps):
        obs = backend.sense()
        pd = _pointing_deg(obs[:3])
        best = min(best, pd)
        if monitor is not None and honest_state is not None:
            severities.append(monitor.evaluate(honest_state(obs)).severity.name)
        act = np.asarray(ctrl(obs), dtype=np.float32)
        applied = act * g + b  # software fault (physical faults: --fault none + reversed leads)
        backend.actuate(applied)
        traj.append(round(pd, 3))
    backend.close()

    from program import metrics

    res = {
        "backend": a.backend,
        "controller": a.controller,
        "fault": a.fault,
        "fault_gain": [float(x) for x in g],
        "fault_bias": [float(x) for x in b],
        "best_pointing_deg": round(best, 3),
        # Honest gates: SETTLED (held over a 20-step dwell), not the audit-flagged transient-min.
        "settled_operational_5deg": bool(metrics.settled_success(traj, thresh=5.0, dwell=20)),
        "settled_science_0p2deg": bool(metrics.settled_success(traj, thresh=0.2, dwell=20)),
        "transient_min_0p2_legacy": best <= 0.2,  # touch-once; reported for continuity only
        "pointing_trajectory_deg": traj,
    }
    if monitor is not None:
        win = severities[-20:]
        res["shield"] = {
            "live": True,
            "autonomous_recovery": bool(len(win) == 20 and all(s == "NOMINAL" for s in win)),
            "escalation_fraction": round(
                sum(s in ("VIOLATION", "CRITICAL") for s in severities) / len(severities), 4
            ),
            "note": "I4 (pointing) fed real sim state; I1–3,5–9 modeled-nominal (see "
            "sil/integration/closed_loop.py).",
        }
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    sh = f", shield-autonomous={res['shield']['autonomous_recovery']}" if monitor else ""
    print(
        f"[{a.backend}/{a.controller}] fault={a.fault}: best {best:.3f} deg, "
        f"settled-5deg={res['settled_operational_5deg']}{sh} -> {a.out}"
    )


if __name__ == "__main__":
    main()
