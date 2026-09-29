with open("scripts/evaluate_e4_tiled_inference.py", "r") as f:
    content = f.read()

import re

# We will completely replace the evaluation logic for the Control system to just call model.val() directly.
# And we'll keep the Tiled system using SAHI + faster-coco-eval.

replacement = """
    print("\\n[1/2] Running Control (E3 Standard Inference) using YOLO.val()...")
    model = YOLO(weights_path)
    
    start_time = time.time()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    
    val_results = model.val(data="configs/visdrone.yaml", imgsz=1536, split="val", device="0", verbose=False)
    
    stats_control = {
        "total_time": (val_results.speed['preprocess'] + val_results.speed['inference'] + val_results.speed['postprocess']) * len(all_images) / 1000.0,
        "total_detections_pre_cap": 0,
        "max_gpu_mb": get_gpu_memory(),
        "capped_count": 0
    }
    
    metrics_control = {
        "overall": {
            "map50": val_results.results_dict.get("metrics/mAP50(B)", 0.0),
            "map50_95": val_results.results_dict.get("metrics/mAP50-95(B)", 0.0),
            "precision": val_results.results_dict.get("metrics/precision(B)", 0.0),
            "recall": val_results.results_dict.get("metrics/recall(B)", 0.0)
        },
        "class_metrics": []
    }
    for cid in range(10):
        metrics_control["class_metrics"].append({
            "class_name": CLASS_NAMES[cid],
            "ap50": val_results.box.maps[cid] if hasattr(val_results.box, 'maps') and len(val_results.box.maps) > cid else 0.0
        })
    preds_control = {img.name: [] for img in all_images} # dummy empty predictions just for counting
    
"""

content = re.sub(
    r'preds_control, stats_control = run_control_evaluation\(all_images, weights_path, output_dir\)\s*preds_tiled, stats_tiled = run_tiled_evaluation\(all_images, weights_path, output_dir, slice_size=args\.slice_size, overlap=args\.overlap\)\s*print\("\\nEvaluating Predictions \(IoU = 0\.50\)\.\.\."\)\s*evaluator = COCOEvaluatorWrapper\(val_dir=val_dir\)\s*metrics_control = evaluator\.evaluate_predictions\(preds_control, system_name="Control \(E3\)"\)\s*metrics_tiled = evaluator\.evaluate_predictions\(preds_tiled, system_name="Tiled \(E4\)"\)',
    """
    preds_tiled, stats_tiled = run_tiled_evaluation(all_images, weights_path, output_dir, slice_size=args.slice_size, overlap=args.overlap)
    """ + replacement + """
    evaluator = COCOEvaluatorWrapper(val_dir=val_dir)
    metrics_tiled = evaluator.evaluate_predictions(preds_tiled, system_name="Tiled (E4)")
    """,
    content,
    flags=re.DOTALL
)

with open("scripts/evaluate_e4_tiled_inference.py", "w") as f:
    f.write(content)

