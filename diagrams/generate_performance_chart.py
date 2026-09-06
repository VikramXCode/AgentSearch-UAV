import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_performance_comparison(output_path=None):
    """
    Generates a high-resolution IEEE research-paper styled grouped bar chart 
    matching the exact real calculated evaluation metrics from the 
    VisDrone2019-DET-val benchmark.
    """
    if output_path is None:
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir / "performance_comparison.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. System Configurations & Authentic Benchmark Metrics
    # ---------------------------------------------------------
    systems = [
        "Baseline YOLO-World\n(Zero-Shot)",
        "YOLO-World + SAHI\n(Sliced Inference)",
        "AgentUAV\n(Multi-Agent Pipeline)"
    ]
    
    metrics = [
        "Precision (%)",
        "Recall (%)",
        "mAP@0.5 (%)",
        "F1-Score (%)"
    ]

    # Real calculated values from VisDrone2019-DET-val benchmark (tuned operating points)
    data = {
        "Precision (%)": [29.8, 67.8, 65.4],
        "Recall (%)":    [22.3, 61.6, 61.1],
        "mAP@0.5 (%)":   [7.1,  37.2, 33.6],
        "F1-Score (%)":  [25.5, 64.6, 63.2]
    }

    # High-contrast IEEE Paper Color Palette
    colors = ['#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

    # ---------------------------------------------------------
    # 2. Figure & Axis Geometry
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    
    # Typography
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'

    x = np.arange(len(systems))
    num_metrics = len(metrics)
    bar_width = 0.18
    
    # Calculate offsets for 4 bars per group
    offsets = [-1.5 * bar_width, -0.5 * bar_width, 0.5 * bar_width, 1.5 * bar_width]

    # ---------------------------------------------------------
    # 3. Plotting Bars & Value Annotations
    # ---------------------------------------------------------
    for i, (metric_name, color) in enumerate(zip(metrics, colors)):
        values = data[metric_name]
        bar_positions = x + offsets[i]
        
        bars = ax.bar(
            bar_positions, 
            values, 
            width=bar_width, 
            label=metric_name, 
            color=color, 
            edgecolor='#222222',
            linewidth=0.8,
            zorder=3
        )
        
        # Display values above each bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8.5,
                fontweight='bold',
                color='#111111'
            )

    # ---------------------------------------------------------
    # 4. Chart Title, Ticks & IEEE Styling
    # ---------------------------------------------------------
    ax.set_title("Performance Comparison of UAV Target Detection Systems\n(Evaluated on VisDrone2019-DET-val with Real Benchmark Metrics)", 
                 fontsize=13.5, fontweight='bold', pad=18, color='#1A237E')
    
    ax.set_ylabel("Evaluation Score (%)", fontsize=11.5, fontweight='bold', labelpad=10, color='#111111')
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10.5, fontweight='bold', color='#111111')
    
    ax.set_ylim(0, 85)
    ax.set_yticks([0, 15, 30, 45, 60, 75])
    ax.set_yticklabels(['0%', '15%', '30%', '45%', '60%', '75%'], fontsize=10.0, color='#222222')

    # Gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#B0BEC5', zorder=0)
    ax.xaxis.grid(False)

    # Border spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#37474F')
    ax.spines['bottom'].set_color('#37474F')

    # Legend
    legend = ax.legend(
        loc='upper left',
        frameon=True,
        facecolor='white',
        edgecolor='#CFD8DC',
        framealpha=0.95,
        fontsize=10.0,
        handlelength=1.4,
        handletextpad=0.6,
        borderpad=0.7
    )
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Chart successfully saved to: {output_path}")

if __name__ == "__main__":
    generate_performance_comparison()
