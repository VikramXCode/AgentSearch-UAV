import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_flowchart(output_path=None):
    """
    Generates a high-resolution IEEE research-paper style flowchart for the AgentUAV project,
    reflecting the authentic VisDrone2019 aerial dataset curation, sanitization, and benchmark split pipeline.
    """
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "agentuav_dataset_flowchart.png")
    else:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    # Figure dimensions for clean IEEE portrait layout
    fig = plt.figure(figsize=(7.5, 9.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Typography settings
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'

    # Box dimensions and styling
    box_w = 64
    box_h = 8.5
    center_x = 50
    box_x = center_x - box_w / 2

    # Step 1 to 5 vertical positions
    y_positions = [87, 72, 57, 42, 27]

    # Data stages and processing descriptions (100% authentic to AgentUAV & VisDrone2019)
    steps = [
        ("Raw VisDrone2019 Aerial Dataset\n7,019 Images", 11.5, True),
        ("Initial Filtering & Quality Control\n(Coordinate Sanitization & Out-of-Bounds Checks)", 10.5, False),
        ("Sanitized Aerial Annotations\nNormalized Bounding Boxes [x, y, w, h]", 10.5, True),
        ("Standardization & YOLO-World Formatting\n10 Aerial Target Categories", 10.5, False),
        ("Final Benchmark Dataset (VisDrone2019-YOLO)\n7,019 Images", 11.0, True),
    ]

    # ---------------------------------------------------------
    # Draw Vertical Boxes (Steps 1 to 5)
    # ---------------------------------------------------------
    for i, ((text, font_size, is_stage), y) in enumerate(zip(steps, y_positions)):
        # Crisp rectangular box matching reference style
        rect = patches.Rectangle(
            (box_x, y), box_w, box_h,
            facecolor='white',
            edgecolor='#111111',
            linewidth=1.4,
            zorder=2
        )
        ax.add_patch(rect)

        # Centered bold text
        ax.text(
            center_x, y + box_h / 2, text,
            ha='center', va='center',
            fontsize=font_size, fontweight='bold',
            color='#111111', linespacing=1.3, zorder=3
        )

        # Arrow down to next step (for steps 1 to 4)
        if i < len(steps) - 1:
            next_y = y_positions[i + 1]
            arrow_start_y = y
            arrow_end_y = next_y + box_h
            
            ax.annotate(
                '',
                xy=(center_x, arrow_end_y),
                xytext=(center_x, arrow_start_y),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color='black',
                    lw=1.5,
                    mutation_scale=12
                ),
                zorder=4
            )

    # ---------------------------------------------------------
    # Draw Split / Fork Branches (Step 5 to Training & Validation Sets)
    # ---------------------------------------------------------
    bottom_y = 6.5
    branch_w = 34
    branch_h = 10.5
    left_x = 12
    right_x = 54

    # Left Split Box: Training Set
    left_rect = patches.Rectangle(
        (left_x, bottom_y), branch_w, branch_h,
        facecolor='white',
        edgecolor='#111111',
        linewidth=1.4,
        zorder=2
    )
    ax.add_patch(left_rect)

    ax.text(
        left_x + branch_w / 2, bottom_y + branch_h / 2,
        "Training Set (Train Corpus)\n6,471 Images\n92.2%",
        ha='center', va='center',
        fontsize=10.5, fontweight='bold',
        color='#111111', linespacing=1.3, zorder=3
    )

    # Right Split Box: Validation Set
    right_rect = patches.Rectangle(
        (right_x, bottom_y), branch_w, branch_h,
        facecolor='white',
        edgecolor='#111111',
        linewidth=1.4,
        zorder=2
    )
    ax.add_patch(right_rect)

    ax.text(
        right_x + branch_w / 2, bottom_y + branch_h / 2,
        "Validation Set (Held-Out Eval)\n548 Images\n7.8%",
        ha='center', va='center',
        fontsize=10.5, fontweight='bold',
        color='#111111', linespacing=1.3, zorder=3
    )

    # Fork connectors: Stem down from Step 5, branch left and right, then arrow down
    step5_bottom_y = y_positions[-1]
    fork_y = step5_bottom_y - 4.5
    left_target_x = left_x + branch_w / 2
    right_target_x = right_x + branch_w / 2
    branch_top_y = bottom_y + branch_h

    # Vertical stem from Step 5 to fork point
    ax.plot([center_x, center_x], [step5_bottom_y, fork_y], color='black', lw=1.5, zorder=4)

    # Horizontal connector across the fork
    ax.plot([left_target_x, right_target_x], [fork_y, fork_y], color='black', lw=1.5, zorder=4)

    # Left downward arrow
    ax.annotate(
        '',
        xy=(left_target_x, branch_top_y),
        xytext=(left_target_x, fork_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color='black',
            lw=1.5,
            mutation_scale=12
        ),
        zorder=4
    )

    # Right downward arrow
    ax.annotate(
        '',
        xy=(right_target_x, branch_top_y),
        xytext=(right_target_x, fork_y),
        arrowprops=dict(
            arrowstyle="-|>",
            color='black',
            lw=1.5,
            mutation_scale=12
        ),
        zorder=4
    )

    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Flowchart saved successfully to: {output_path}")

if __name__ == "__main__":
    generate_flowchart()
