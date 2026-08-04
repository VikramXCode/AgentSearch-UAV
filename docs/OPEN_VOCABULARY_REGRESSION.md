# Open-Vocabulary Regression Plan

The fine-tuned model must preserve the current text-query capability, not just improve VisDrone validation scores.

## Closed-set VisDrone checks

Run after training:

```bash
python scripts/evaluate_visdrone.py --checkpoint runs/visdrone_yoloworld/<run_name>/weights/best.pt
python scripts/compare_visdrone_results.py --fine-tuned experiments/<run_name>_visdrone_eval.json
```

Compare:

- Precision
- Recall
- mAP50
- mAP50-95
- per-class mAP where available

## Open-vocabulary regression checks

Keep a small manual regression set of text queries that are outside the VisDrone 10-class vocabulary or that stress semantic matching:

- `find a dog` on `sample_images/world.png`
- `find a car` on existing sample UAV images
- any future out-of-domain text query you care about preserving

Current baseline reminder:

- `find a dog` on `sample_images/world.png` currently returns about 3 dogs with average confidence around 0.914.

Do not invent expected results for new samples until you have actually tested them.

## Promotion rule

Only adopt a fine-tuned checkpoint if both of these are acceptable:

1. VisDrone metrics improve materially versus `experiments/baseline_visdrone.json`.
2. Open-vocabulary regression behavior remains acceptable on the baseline sample queries.