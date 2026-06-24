"""Paper E high-fidelity fault-campaign figure from committed evidence.

figs/fig_hifi.pdf is a grouped bar chart over four fault regimes showing RMA autonomous
recovery and verified-shield escalation, with the fault-unaware PD recovery as a reference on
the designed scope. Every value is read from evidence/program/hifi_fault_campaign.json; nothing
is fabricated. Run from repo root: python paper/hil/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HF = json.loads((ROOT / "evidence/program/hifi_fault_campaign.json").read_text())
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)


def _label_bars(ax: plt.Axes, bars: object) -> None:
    """Write each bar's height (already in percent) just above the bar."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.5,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main() -> None:
    """Build figs/fig_hifi.pdf from the held-out high-fidelity campaign evidence."""
    head = HF["headline"]
    res = HF["results"]
    designed = head["rma_hifi_designed_pooled"]
    pd_designed = head["pd_hifi_designed_pooled"]
    gain = res["rma_hifi_designed"]["gain"]
    gain_bias = res["rma_hifi_beyond_scope"]["gain_bias"]
    total_loss = res["rma_hifi_beyond_scope"]["total_loss"]
    momentum = res["rma_hifi_stored_momentum_gain"]

    regimes = [
        "Designed scope\n(gain, sign)",
        "Additive bias",
        "Dead axis",
        "Stored momentum",
    ]
    recovery = [
        100.0 * designed["autonomous_recovery_rate"],
        100.0 * gain_bias["autonomous_recovery_rate"],
        100.0 * total_loss["autonomous_recovery_rate"],
        100.0 * momentum["autonomous_recovery_rate"],
    ]
    escalation = [
        100.0 * gain["escalation_rate"],
        100.0 * gain_bias["escalation_rate"],
        100.0 * total_loss["escalation_rate"],
        100.0 * momentum["escalation_rate"],
    ]
    pd_reference = 100.0 * pd_designed["autonomous_recovery_rate"]

    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    positions = list(range(len(regimes)))
    width = 0.26

    rec_bars = ax.bar(
        [p - width / 2.0 for p in positions],
        recovery,
        width,
        label="RMA autonomous recovery",
        color="#2c7fb8",
    )
    esc_bars = ax.bar(
        [p + width / 2.0 for p in positions],
        escalation,
        width,
        label="Verified-shield escalation",
        color="#d95f0e",
    )
    pd_bar = ax.bar(
        [-width / 2.0 - width],
        [pd_reference],
        width,
        label="Fault-unaware PD recovery",
        color="#bdbdbd",
        hatch="//",
    )

    _label_bars(ax, rec_bars)
    _label_bars(ax, esc_bars)
    _label_bars(ax, pd_bar)

    ax.set_ylabel("rate (%)")
    ax.set_ylim(0, 115)
    ax.set_xticks(positions)
    ax.set_xticklabels(regimes)
    ax.set_title("High-fidelity fault campaign: recovery and shield escalation")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), fontsize=8, ncol=3, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = FIGS / "fig_hifi.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {out} | recovery={recovery} escalation={escalation} "
        f"pd_designed={pd_reference}"
    )


if __name__ == "__main__":
    main()
