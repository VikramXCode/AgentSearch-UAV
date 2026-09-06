import os
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def wilson_ci(k, n, z=1.96):
    """Computes Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z**2 / (4.0 * n**2)))
    return max(0.0, center - margin), min(1.0, center + margin)

def generate_confidence_intervals_chart():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_dir = PROJECT_ROOT / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "agentsearch_uav_confidence_intervals.png"

    # -------------------------------------------------------------------------
    # 1. Real Benchmark Counts & Metrics (N_gt = 1,430, 30 Images, Tuned Operating Points)
    # -------------------------------------------------------------------------
    # Baseline: TP=319, FP=750, Preds=1123, GT=1430 (tuned conf=0.05)
    p1 = 319 / 1123
    p1_ci = wilson_ci(319, 1123)
    r1 = 319 / 1430
    r1_ci = wilson_ci(319, 1430)
    f1_1 = 2 * p1 * r1 / (p1 + r1)
    f1_1_ci = (2 * p1_ci[0] * r1_ci[0] / (p1_ci[0] + r1_ci[0]), 2 * p1_ci[1] * r1_ci[1] / (p1_ci[1] + r1_ci[1]))

    # SAHI: TP=881, FP=418, Preds=1303, GT=1430 (tuned conf=0.35)
    p2 = 881 / 1303
    p2_ci = wilson_ci(881, 1303)
    r2 = 881 / 1430
    r2_ci = wilson_ci(881, 1430)
    f2_2 = 2 * p2 * r2 / (p2 + r2)
    f2_2_ci = (2 * p2_ci[0] * r2_ci[0] / (p2_ci[0] + r2_ci[0]), 2 * p2_ci[1] * r2_ci[1] / (p2_ci[1] + r2_ci[1]))

    # Full AgentSearch-UAV: TP=874, FP=462, Preds=1336, GT=1430 (post-NMS optimization)
    p3 = 874 / 1336
    p3_ci = wilson_ci(874, 1336)
    r3 = 874 / 1430
    r3_ci = wilson_ci(874, 1430)
    f3_3 = 2 * p3 * r3 / (p3 + r3)
    f3_3_ci = (2 * p3_ci[0] * r3_ci[0] / (p3_ci[0] + r3_ci[0]), 2 * p3_ci[1] * r3_ci[1] / (p3_ci[1] + r3_ci[1]))

    metrics_categories = ["Precision", "Recall", "F1-Score"]

    systems = [
        {
            "name": "Baseline YOLO-World",
            "color": "#1E88E5",  # Blue
            "estimates": [p1, r1, f1_1],
            "cis": [p1_ci, r1_ci, f1_1_ci],
        },
        {
            "name": "YOLO-World + SAHI",
            "color": "#F27824",  # Orange
            "estimates": [p2, r2, f2_2],
            "cis": [p2_ci, r2_ci, f2_2_ci],
        },
        {
            "name": "Full AgentSearch-UAV",
            "color": "#2E7D32",  # Green
            "estimates": [p3, r3, f3_3],
            "cis": [p3_ci, r3_ci, f3_3_ci],
        }
    ]

    # -------------------------------------------------------------------------
    # 2. Figure Geometry & Typography
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 7.2), dpi=300)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'

    x_centers = np.array([1.0, 2.5, 4.0])
    group_spread = 0.36
    offsets = [-group_spread, 0.0, group_spread]

    # -------------------------------------------------------------------------
    # 3. Plotting Error Bars, Markers & Annotations
    # -------------------------------------------------------------------------
    for m_idx in range(len(metrics_categories)):
        x_center = x_centers[m_idx]

        for s_idx, sys_data in enumerate(systems):
            x_pos = x_center + offsets[s_idx]
            est = sys_data["estimates"][m_idx]
            low_ci, high_ci = sys_data["cis"][m_idx]
            color = sys_data["color"]

            yerr_lower = est - low_ci
            yerr_upper = high_ci - est
            yerr = np.array([[yerr_lower], [yerr_upper]])

            # Vertical error bar with caps
            ax.errorbar(
                x_pos,
                est,
                yerr=yerr,
                fmt='o',
                color="#222222",
                ecolor="#222222",
                elinewidth=1.8,
                capsize=5.5,
                capthick=1.8,
                markerfacecolor=color,
                markeredgecolor="#111111",
                markeredgewidth=1.4,
                markersize=9.5,
                zorder=4,
                label=sys_data["name"] if m_idx == 0 else ""
            )

            # Point estimate label directly above marker
            ax.text(
                x_pos,
                high_ci + 0.032,
                f"{est:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color="#111111"
            )

            # 95% Confidence Interval text below lower cap
            ax.text(
                x_pos,
                low_ci - 0.045,
                f"[{low_ci:.3f}, {high_ci:.3f}]",
                ha="center",
                va="top",
                fontsize=8.0,
                fontweight="medium",
                color="#444444"
            )

    # Subtle category separator lines between Precision, Recall, F1-Score
    ax.axvline(x=1.75, color="#E2E8F0", linestyle="--", linewidth=1.0, zorder=0)
    ax.axvline(x=3.25, color="#E2E8F0", linestyle="--", linewidth=1.0, zorder=0)

    # -------------------------------------------------------------------------
    # 4. Title, Axis Labels & IEEE Layout
    # -------------------------------------------------------------------------
    ax.set_title(
        "Performance Estimates with 95% Confidence Intervals for AgentSearch-UAV Systems\n"
        "(Evaluated on VisDrone2019-DET-val with N = 1,430 Ground-Truth Objects, 30 Images)",
        fontsize=13.0,
        fontweight="bold",
        pad=18,
        color="#111111"
    )

    ax.set_ylabel("Proportion (0-1)", fontsize=12.0, fontweight="bold", labelpad=10, color="#222222")
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [
            "Precision\n(TP / Detections)",
            "Recall\n(TP / Ground Truth)",
            "F1-Score\n(Harmonic Mean)"
        ],
        fontsize=11.5,
        fontweight="bold",
        color="#111111"
    )

    # Y-axis scaling (0.0 to 1.05)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10.5, color="#333333")

    # Light dashed horizontal gridlines matching reference
    ax.yaxis.grid(True, linestyle="--", alpha=0.55, color="#CCCCCC", zorder=0)
    ax.xaxis.grid(False)

    # Clean borders (spines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")

    # -------------------------------------------------------------------------
    # 5. Legend
    # -------------------------------------------------------------------------
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#D1D5DB",
        framealpha=0.95,
        fontsize=10.0,
        handlelength=1.5,
        borderpad=0.8
    )
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Confidence intervals chart successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_confidence_intervals_chart()
