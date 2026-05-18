"""Check SNR-ready baseline results."""
import json
from pathlib import Path

root = Path("results/updated_dataset_baseline_6_4_seed44_snr_ready_20260514/6_4")
models = ["mlp", "cnn1d", "transformer", "itransformer", "logreg", "svm", "random_forest"]

for m in models:
    p = root / m / "metrics.json"
    if not p.exists():
        print(f"{m}: NOT FOUND")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    test = d.get("test", d)
    acc = test.get("accuracy", 0)
    mf1 = test.get("macro_f1", 0)
    param = d.get("parameter_count", 0)
    preds = test.get("predictions", [])
    import collections
    pred_dist = dict(collections.Counter(preds))
    print(f"{m}: acc={acc:.4f}, macro_f1={mf1:.4f}, params={param}, pred_dist={pred_dist}")
