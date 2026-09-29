#!/usr/bin/env python3
import os
import sys
import argparse
import json
import csv
import subprocess
import time
from pathlib import Path

_current = Path(__file__).resolve().parent
PROJECT_ROOT = _current.parent

REGISTRY_PATH = PROJECT_ROOT / "experiments" / "research" / "experiment_registry.json"
MASTER_MD = PROJECT_ROOT / "experiments" / "research" / "MASTER_RESULTS.md"
MASTER_CSV = PROJECT_ROOT / "experiments" / "research" / "FINAL_COMPARISON.csv"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_research.py"

def load_registry():
    if not REGISTRY_PATH.exists():
        print(f"Registry not found at {REGISTRY_PATH}")
        sys.exit(1)
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def save_registry(registry):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def update_master_results(registry):
    # Write MD
    md_content = "# Master Research Results\n\n"
    md_content += "| Experiment | Hypothesis | mAP50 | mAP50:95 | Precision | Recall | Delta vs E3 (mAP50) | Status |\n"
    md_content += "|---|---|---|---|---|---|---|---|\n"
    
    csv_rows = []
    
    for exp in registry:
        eid = exp["experiment_id"]
        hyp = exp["hypothesis"]
        status = exp["status"]
        
        map50 = f"{exp.get('map50', 0)*100:.2f}%" if exp.get('map50') else "N/A"
        map50_95 = f"{exp.get('map50_95', 0)*100:.2f}%" if exp.get('map50_95') else "N/A"
        precision = f"{exp.get('precision', 0)*100:.2f}%" if exp.get('precision') else "N/A"
        recall = f"{exp.get('recall', 0)*100:.2f}%" if exp.get('recall') else "N/A"
        
        delta = "N/A"
        diff = None
        baseline_map50 = 0.6210
        if exp.get('map50'):
            diff = (exp['map50'] - baseline_map50) * 100
            delta = f"{diff:+.2f}%"
            if eid == "EXP01":
                delta = "Reference"
                
        md_content += f"| {eid} | {hyp} | {map50} | {map50_95} | {precision} | {recall} | {delta} | {status} |\n"
        
        csv_rows.append({
            "Experiment": eid,
            "Name": exp["experiment_name"],
            "Status": status,
            "mAP50": exp.get('map50'),
            "mAP50_95": exp.get('map50_95'),
            "Precision": exp.get('precision'),
            "Recall": exp.get('recall'),
            "Delta_mAP50": diff if delta not in ["N/A", "Reference"] else None
        })
        
    with open(MASTER_MD, "w") as f:
        f.write(md_content)
        
    # Write CSV
    if csv_rows:
        with open(MASTER_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

def run_experiment(exp, registry, dry_run=False):
    print(f"\n{'='*50}")
    print(f"STARTING {exp['experiment_id']}: {exp['experiment_name']}")
    print(f"{'='*50}")
    
    if not dry_run:
        exp["status"] = "RUNNING"
        exp["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_registry(registry)
    
    out_dir = PROJECT_ROOT / exp["output_directory"]
    
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--exp-dir", str(out_dir)
    ]
    if exp.get("enable_sahi"): cmd.append("--sahi")
    if exp.get("enable_super_resolution"): cmd.append("--sr")
    if exp.get("enable_clip_verification"): cmd.append("--clip")
    if exp.get("is_adaptive"): cmd.append("--adaptive")
    if exp.get("is_open_vocab"): cmd.append("--open-vocab")
    
    print(f"Executing: {' '.join(cmd)}")
    
    if dry_run:
        print("[DRY-RUN] Would execute evaluator, skipping...")
        return
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] Experiment {exp['experiment_id']} FAILED during execution.")
        exp["status"] = "FAILED"
        save_registry(registry)
        sys.exit(1)
        
    # Load metrics
    met_path = out_dir / "metrics.json"
    if not met_path.exists():
        print(f"\n[ERROR] Evaluation metrics not found at {met_path}. Experiment FAILED.")
        exp["status"] = "FAILED"
        save_registry(registry)
        sys.exit(1)
        
    with open(met_path, "r") as f:
        metrics = json.load(f)
        
    exp["precision"] = metrics["overall"]["precision"]
    exp["recall"] = metrics["overall"]["recall"]
    exp["map50"] = metrics["overall"]["map50"]
    exp["map50_95"] = metrics["overall"]["map50_95"]
    
    exp["status"] = "COMPLETE"
    exp["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    save_registry(registry)
    update_master_results(registry)
    print(f"Experiment {exp['experiment_id']} COMPLETE. Registry and Master files updated.")

def main():
    parser = argparse.ArgumentParser(description="AgentSearch-UAV Research Experiment Runner")
    parser.add_argument("--status", action="store_true", help="Print registry status and exit")
    parser.add_argument("--resume", action="store_true", help="Resume pending/running experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be executed without running")
    parser.add_argument("--experiment", type=str, help="Run a specific experiment ID (e.g. EXP01)")
    args = parser.parse_args()

    registry = load_registry()
    
    if args.status:
        print("Experiment Status:")
        for exp in registry:
            print(f"  {exp['experiment_id']}: {exp['status']} - {exp['experiment_name']}")
        sys.exit(0)
        
    for exp in registry:
        if args.experiment and exp["experiment_id"] != args.experiment:
            continue
            
        if exp["status"] == "COMPLETE" and not args.experiment:
            print(f"Skipping {exp['experiment_id']}: Already COMPLETE.")
            continue
        if exp["status"] == "REQUIRES_MANUAL_INTERVENTION" and not args.experiment:
            print(f"Skipping {exp['experiment_id']}: REQUIRES MANUAL INTERVENTION.")
            continue
        if exp["status"] == "FAILED" and not args.resume and not args.experiment:
            print(f"[FATAL] Sequence halted because {exp['experiment_id']} previously FAILED. Manual intervention required. Use --resume to override.")
            sys.exit(1)
            
        if exp["status"] in ["PLANNED", "RUNNING", "FAILED"] or args.experiment:
            run_experiment(exp, registry, dry_run=args.dry_run)
            
    if not args.dry_run and not args.status:
        print("\nAll planned executable experiments have finished.")
    
if __name__ == "__main__":
    main()
