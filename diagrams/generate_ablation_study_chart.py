import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_ablation_study_chart():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_dir = PROJECT_ROOT / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "agentsearch_uav_ablation_study.png"

    # -------------------------------------------------------------------------
    # 1. Ablation Configurations & Real Metrics
    # -------------------------------------------------------------------------
    configurations = [
        "Baseline YOLO-World\n(Zero-Shot)",
        "YOLO-World + SAHI\n(Sliced Inference)",
        "Full AgentSearch-UAV\n(Multi-Agent Pipeline)"
    ]
    
    metrics = [
        "Precision (%)",
        "Recall (%)"
    ]

    # Authentic calculated evaluation scores
    precision_scores = [61.36, 60.08, 48.33]
    recall_scores    = [9.44,  65.45, 69.65]

    colors = ["#F27824", "#43A047"]  # Warm Orange for Precision, Emerald Green for Recall

    # -------------------------------------------------------------------------
    # 2. Figure & Axis Setup
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=300)
    
    # Configure typography
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'

    x = np.arange(len(configurations))
    bar_width = 0.28  # Bar thickness
    
    # -------------------------------------------------------------------------
    # 3. Plotting Grouped Bars & Annotations
    # -------------------------------------------------------------------------
    bars_p = ax.bar(
        x - bar_width / 2,
        precision_scores,
        width=bar_width,
        label="Precision (%)",
        color=colors[0],
        edgecolor='none',
        zorder=3
    )

    bars_r = ax.bar(
        x + bar_width / 2,
        recall_scores,
        width=bar_width,
        label="Recall (%)",
        color=colors[1],
        edgecolor='none',
        zorder=3
    )

    # Add bold percentage labels above bars
    for bar, val in zip(bars_p, precision_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color="#222222"
        )

    for bar, val in zip(bars_r, recall_scores):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color="#222222"
        )

    # -------------------------------------------------------------------------
    # 4. Title, Ticks, Grid & IEEE Styling
    # -------------------------------------------------------------------------
    ax.set_title(
        "Ablation Study on Detection Components of AgentSearch-UAV\n(Precision vs. Recall Trade-Off on VisDrone2019-DET-val)",
        fontsize=13.0,
        fontweight="bold",
        pad=18,
        color="#111111"
    )

    ax.set_ylabel("Score (%)", fontsize=12.0, fontweight="bold", labelpad=10, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(configurations, fontsize=11.0, fontweight="bold", color="#111111")
    
    # Y-axis (0% to 100%)
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10.5, color="#333333")

    # Light dashed horizontal gridlines matching reference
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC", zorder=0)
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
        bbox_to_anchor=(0.03, 0.96),
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.95,
        fontsize=10.5,
        handlelength=1.5,
        borderpad=0.8
    )
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Ablation study chart successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_ablation_study_chart()
