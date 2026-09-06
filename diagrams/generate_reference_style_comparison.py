import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_comparison_chart():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    output_dir = PROJECT_ROOT / "diagrams"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "agentsearch_uav_performance_comparison.png"

    # -------------------------------------------------------------------------
    # 1. Metric Categories (X-Axis) & System Configurations
    # -------------------------------------------------------------------------
    categories = [
        "Precision",
        "Recall",
        "mAP@50",
        "F1-Score"
    ]
    
    # 3 Systems evaluated on VisDrone2019-DET-val (tuned operating points)
    systems_data = [
        {
            "name": "Baseline YOLO-World",
            "scores": [29.84, 22.31, 7.06, 25.53],
            "color": "#1E88E5",  # Classic Royal Blue
        },
        {
            "name": "YOLO-World + SAHI",
            "scores": [67.82, 61.61, 37.21, 64.57],
            "color": "#F27824",  # Warm Orange (matching reference)
        },
        {
            "name": "Full AgentSearch-UAV",
            "scores": [65.42, 61.12, 33.59, 63.20],
            "color": "#4CAF50",  # Vivid Green (matching reference)
        }
    ]

    # -------------------------------------------------------------------------
    # 2. Figure & Axis Geometry
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    
    # Typography
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'

    x = np.arange(len(categories))
    num_systems = len(systems_data)
    bar_width = 0.24  # Width of each bar
    
    # Symmetric offsets for 3 bars per category
    offsets = [-bar_width, 0, bar_width]

    # -------------------------------------------------------------------------
    # 3. Plotting Bars & Value Annotations
    # -------------------------------------------------------------------------
    for i, system in enumerate(systems_data):
        scores = system["scores"]
        positions = x + offsets[i]
        
        bars = ax.bar(
            positions,
            scores,
            width=bar_width,
            label=system["name"],
            color=system["color"],
            edgecolor='none',
            zorder=3
        )

        # Value labels above bars (percentage format)
        for bar, val in zip(bars, scores):
            height = bar.get_height()
            y_pos = height + 1.2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.2,
                fontweight="bold",
                color="#222222"
            )

    # -------------------------------------------------------------------------
    # 4. Title, Ticks, Grid & IEEE Styling
    # -------------------------------------------------------------------------
    ax.set_title(
        "AgentSearch-UAV Performance Comparison\n(Precision, Recall, mAP@50, F1-Score on VisDrone2019-DET-val)",
        fontsize=13.5,
        fontweight="bold",
        pad=18,
        color="#111111"
    )

    ax.set_ylabel("Score (%)", fontsize=12.0, fontweight="bold", labelpad=10, color="#222222")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11.5, fontweight="bold", color="#111111")
    
    # Y-axis limits & ticks (0% to 100%)
    ax.set_ylim(0.0, 105.0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10.5, color="#333333")

    # Light dashed horizontal gridlines matching reference
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC", zorder=0)
    ax.xaxis.grid(False)

    # Spines (top & right hidden, left & bottom clean)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")

    # -------------------------------------------------------------------------
    # 5. Legend (Positioned in Upper Right Empty Space to Avoid Bar Overlap)
    # -------------------------------------------------------------------------
    legend = ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.96),
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=0.95,
        fontsize=10.0,
        handlelength=1.5,
        borderpad=0.8
    )
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Chart successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_comparison_chart()
