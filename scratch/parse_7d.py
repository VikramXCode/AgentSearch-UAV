import json
with open("scratch/calibration_7d.json", "r") as f:
    res = json.load(f)
ag = res["aggregate"]

for cond in ["base", "sahi_b", "sahi_c"]:
    tp, fp, fn, rt = ag[cond]["tp"], ag[cond]["fp"], ag[cond]["fn"], ag[cond]["runtime"]
    p = tp/(tp+fp) if tp+fp>0 else 0
    r = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    print(f"{cond.upper()}: TP={tp} FP={fp} FN={fn} P={p:.3f} R={r:.3f} F1={f1:.3f} Time={rt:.3f}s")

# Changes relative to E3
btp, bfp, bfn, brt = ag["base"]["tp"], ag["base"]["fp"], ag["base"]["fn"], ag["base"]["runtime"]
bp = btp/(btp+bfp) if btp+bfp>0 else 0
br = btp/(btp+bfn) if btp+bfn>0 else 0
bf1 = 2*bp*br/(bp+br) if bp+br>0 else 0

for cond in ["sahi_b", "sahi_c"]:
    tp, fp, fn, rt = ag[cond]["tp"], ag[cond]["fp"], ag[cond]["fn"], ag[cond]["runtime"]
    p = tp/(tp+fp) if tp+fp>0 else 0
    r = tp/(tp+fn) if tp+fn>0 else 0
    f1 = 2*p*r/(p+r) if p+r>0 else 0
    print(f"Diff {cond}: dTP={tp-btp} dFP={fp-bfp} dFN={fn-bfn} dP={p-bp:.3f} dR={r-br:.3f} dF1={f1-bf1:.3f} TimeMult={rt/brt if brt>0 else 0:.2f}x")

sm = res["small_objects"]
print("SMALL OBJECTS:")
for cond in ["base", "sahi_b", "sahi_c"]:
    print(f"{cond}: TP={sm[cond]['tp']} FN={sm[cond]['fn']}")
