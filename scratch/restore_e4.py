with open("scripts/evaluate_e4_tiled_inference.py", "r") as f:
    content = f.read()

import re

# We will remove the replacement from the patch and restore predict()
content = re.sub(
    r'preds_tiled, stats_tiled = run_tiled_evaluation\(all_images, weights_path, output_dir, slice_size=args\.slice_size, overlap=args\.overlap\)\n\s+print\("\\n\[1/2\] Running Control.*?preds_control = {img\.name: \[\] for img in all_images} # dummy empty predictions just for counting\n\s+',
    """
    preds_control, stats_control = run_control_evaluation(all_images, weights_path, output_dir)
    preds_tiled, stats_tiled = run_tiled_evaluation(all_images, weights_path, output_dir, slice_size=args.slice_size, overlap=args.overlap)
    
    print("\\nEvaluating Predictions (IoU = 0.50)...")
    evaluator = COCOEvaluatorWrapper(val_dir=val_dir)
    
    metrics_control = evaluator.evaluate_predictions(preds_control, system_name="Control (E3)")
    """,
    content,
    flags=re.DOTALL
)

with open("scripts/evaluate_e4_tiled_inference.py", "w") as f:
    f.write(content)
