import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def generate_agentuav_pipeline(output_path=None):
    """
    Generates a high-resolution IEEE research-paper style architecture diagram
    for the AgentUAV project, featuring:
    - 100% Project-Authentic VisDrone2019-DET Dataset Split (6,471 train vs 548 val)
    - Full AgentUAV Multi-Agent Workflow:
      Query Agent -> Knowledge & Strategy Agent -> Tool Agent -> Detection Agent -> Verification Agent -> Explanation Agent
    """
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "agentuav_multi_agent_pipeline.png")
    else:
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    # Create figure with high DPI and 16:7 landscape aspect ratio
    fig = plt.figure(figsize=(16, 7), dpi=300)
    
    # Configure typography
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.family'] = 'sans-serif'
    
    # =========================================================================
    # Left Subplot: Authentic VisDrone2019-DET Dataset Split
    # =========================================================================
    ax1 = fig.add_axes([0.02, 0.08, 0.36, 0.82])
    
    # Title matching project dataset
    ax1.text(0.5, 1.08, "Dataset Split:\nVisDrone2019-DET (7,019 Images)", 
             ha='center', va='center', fontsize=13.5, fontweight='bold', 
             transform=ax1.transAxes, linespacing=1.3, color='#111111')
    
    # Authentic counts from VisDrone2019 benchmark configuration:
    # 6,471 train images (92.2%), 548 validation images (7.8%)
    sizes = [7.8, 92.2]
    colors = ['#4DB748', '#F17235']  # Green for Val/Test, Orange for Train Corpus
    
    # Pie chart clockwise from 90°
    wedges, texts = ax1.pie(
        sizes, 
        colors=colors, 
        startangle=90, 
        counterclock=False,
        wedgeprops=dict(edgecolor='white', linewidth=3.0, antialiased=True)
    )
    
    # Percentage text
    ax1.text(0.20, 0.68, "7.8%", ha='center', va='center', fontsize=11, fontweight='bold', color='black')
    ax1.text(-0.25, -0.35, "92.2%", ha='center', va='center', fontsize=12.5, fontweight='bold', color='black')
    
    # External callout labels matching exact project files (configs/visdrone.yaml & experiments/baseline_visdrone.json)
    ax1.text(0.68, 0.92, "Validation Set\n(Held-Out Eval)\n(548 images, 7.8%)", 
             ha='left', va='center', fontsize=10.5, fontweight='bold', color='#111111', linespacing=1.25)
    
    ax1.text(-0.85, -0.96, "Training Set (Train Corpus)\n(6,471 images, 92.2%)", 
             ha='center', va='center', fontsize=10.5, fontweight='bold', color='#111111', linespacing=1.25)
    
    ax1.axis('equal')
    
    # =========================================================================
    # Right Subplot: AgentUAV Multi-Agent Pipeline
    # =========================================================================
    ax2 = fig.add_axes([0.40, 0.08, 0.58, 0.82])
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.axis('off')
    
    # Main Section Title
    ax2.text(50, 108, "AgentUAV Multi-Agent Adaptive Pipeline", 
             ha='center', va='center', fontsize=14.5, fontweight='bold', color='black')
    
    # Top Output Label
    ax2.text(50, 93, "Output: Detections + Bounding Boxes + Confidence + Explanation Report", 
             ha='center', va='center', fontsize=10.5, fontweight='normal', color='#222222')
    
    # Bottom Input Label
    ax2.text(50, 7, "Input: UAV Aerial Image / Video + Natural Language Target Query", 
             ha='center', va='center', fontsize=11, fontweight='normal', color='#222222')
    
    # The 6 Agents in AgentUAV execution sequence:
    agents = [
        ("Query\nAgent", "#4282DC"),                  # 1. Query parsing & visual attributes (Blue)
        ("Knowledge &\nStrategy\nAgent", "#F07B37"),  # 2. Domain rules & adaptive planning (Orange)
        ("Tool\nAgent", "#55BA49"),                   # 3. Super-Resolution & SAHI tiling (Green)
        ("Detection\nAgent", "#9665B3"),              # 4. YOLO-World open-vocabulary detection (Purple)
        ("Verification\nAgent", "#2A9D8F"),           # 5. False positive & CLIP filtering (Teal)
        ("Explanation\nAgent", "#DA5E58")             # 6. Structured rationale & reporting (Coral Red)
    ]
    
    total_agents = len(agents)
    start_x = 0.5
    gap = 2.8
    box_w = (100.0 - 2 * start_x - (total_agents - 1) * gap) / total_agents
    box_h = 70
    box_y = 15
    
    for i, (name, col) in enumerate(agents):
        x = start_x + i * (box_w + gap)
        
        # Rounded card box with subtle dark border
        rect = FancyBboxPatch(
            (x, box_y), box_w, box_h,
            boxstyle="round,pad=0,rounding_size=1.5",
            facecolor=col,
            edgecolor='#222222',
            linewidth=1.2,
            zorder=2
        )
        ax2.add_patch(rect)
        
        # Agent Name inside box
        ax2.text(
            x + box_w / 2, box_y + box_h / 2, name,
            ha='center', va='center',
            fontsize=9.8, fontweight='bold', color='white',
            linespacing=1.25, zorder=3
        )
        
        # Connecting directional arrows between agents
        if i < total_agents - 1:
            arrow_start_x = x + box_w
            arrow_end_x = arrow_start_x + gap
            arrow_y = box_y + box_h / 2
            
            ax2.annotate(
                '', 
                xy=(arrow_end_x, arrow_y), 
                xytext=(arrow_start_x, arrow_y),
                arrowprops=dict(
                    arrowstyle="-|>", 
                    color='black', 
                    lw=1.5, 
                    mutation_scale=11
                ),
                zorder=4
            )
            
    # Save the figure with tight bounding box
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Diagram saved successfully to: {output_path}")

if __name__ == "__main__":
    generate_agentuav_pipeline()
