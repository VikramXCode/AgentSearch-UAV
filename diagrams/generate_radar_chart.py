import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def generate_radar_chart(output_path=None):
    """
    Generates a high-resolution IEEE research-paper styled radar (spider) chart
    comparing the 3 primary experimental detection systems using 100% genuine calculated values
    from the VisDrone2019-DET-val benchmark across 5 evaluation metrics:
    - Precision
    - Recall
    - mAP@50
    - F1-Score
    - Small-Target AP@50
    """
    if output_path is None:
        script_dir = Path(__file__).resolve().parent
        output_path = script_dir / "agentuav_performance_radar.png"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Categories & Real Benchmark Metrics
    # -------------------------------------------------------------------------
    categories = [
        'Precision',
        'Recall',
        'mAP@0.5',
        'F1-Score',
        'Small-Target\nAP@0.5'
    ]
    num_vars = len(categories)

    # Genuine calculated metrics for the 3 requested systems
    systems_data = {
        'Baseline YOLO-World': {
            'scores': [0.6136, 0.0944, 0.0395, 0.1636, 0.0058],
            'color': '#1E88E5',   # Cobalt Blue
            'linewidth': 2.2,
            'linestyle': '--',
        },
        'YOLO-World + SAHI': {
            'scores': [0.6008, 0.6545, 0.3860, 0.6265, 0.3903],
            'color': '#FB8C00',   # Warm Amber / Orange
            'linewidth': 2.4,
            'linestyle': '-.',
        },
        'AgentUAV Multi-Agent System': {
            'scores': [0.4833, 0.6965, 0.3638, 0.5706, 0.4287],
            'color': '#2E7D32',   # Forest Emerald Green
            'linewidth': 2.8,
            'linestyle': '-',
        }
    }

    # -------------------------------------------------------------------------
    # 2. Polar Geometry Setup
    # -------------------------------------------------------------------------
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete circular loop

    # Create figure with ample horizontal width for radar + side legend
    fig, ax = plt.subplots(figsize=(10.5, 8.2), subplot_kw=dict(polar=True), dpi=300)

    # Rotate so 0 is North and goes clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Font styling
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'

    # Perimeter labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11.5, fontweight='bold', color='#111111')
    ax.tick_params(axis='x', pad=18)

    # -------------------------------------------------------------------------
    # 3. Radial Gridlines
    # -------------------------------------------------------------------------
    radial_ticks = [0.15, 0.30, 0.45, 0.60, 0.75]
    ax.set_ylim(0.0, 0.75)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f'{tick:.2f}' for tick in radial_ticks], fontsize=9.5, color='#555555')
    ax.set_rlabel_position(45)

    ax.yaxis.grid(True, linestyle='--', color='#B0BEC5', alpha=0.7, linewidth=0.9)
    ax.xaxis.grid(True, linestyle='--', color='#B0BEC5', alpha=0.7, linewidth=0.9)
    ax.spines['polar'].set_color('#37474F')
    ax.spines['polar'].set_linewidth(1.3)

    # -------------------------------------------------------------------------
    # 4. Plot Polygons & Fill Areas
    # -------------------------------------------------------------------------
    for system_name, data in systems_data.items():
        values = data['scores']
        values_closed = values + values[:1]
        color = data['color']
        lw = data.get('linewidth', 2.0)
        ls = data.get('linestyle', '-')

        ax.plot(
            angles,
            values_closed,
            label=system_name,
            color=color,
            linewidth=lw,
            linestyle=ls,
            marker='o',
            markersize=7.0,
            zorder=4
        )
        ax.fill(
            angles,
            values_closed,
            color=color,
            alpha=0.12,
            zorder=2
        )

    # -------------------------------------------------------------------------
    # 5. Legend & Title
    # -------------------------------------------------------------------------
    plt.title(
        "AgentSearch-UAV Performance Radar\n(Evaluated on VisDrone2019-DET-val with Real Benchmark Metrics)",
        fontsize=12.5,
        fontweight='bold',
        pad=28,
        color='#1A237E'
    )

    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.10, 0.95),
        frameon=True,
        facecolor='white',
        edgecolor='#CFD8DC',
        framealpha=0.95,
        fontsize=10.5,
        handlelength=2.2,
        borderpad=0.9
    )
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Radar chart successfully generated and saved to: {output_path}")

if __name__ == "__main__":
    generate_radar_chart()
