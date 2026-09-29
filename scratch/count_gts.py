import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from evaluation.comprehensive_evaluator import ComprehensiveEvaluator
evaluator = ComprehensiveEvaluator()
total = 0
for img, boxes in evaluator.ground_truths.items():
    total += len(boxes)
print("ComprehensiveEvaluator total GT:", total)
