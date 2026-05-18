"""Quick check: verify noise experiment metrics."""
import json
from pathlib import Path

root = Path("results/current_snr_noise_6_4_seed44_artifacts")
models = ["mlp", "cnn1d", "transformer", "itransformer"]
snrs = ["clean", "snr_40dB", "snr_35dB", "snr_30dB", "snr_25dB", "snr_20dB", "snr_15dB", "snr_10dB"]

for model in models:
    print(f"\n=== {model} ===")
    for snr in snrs:
        p = root / model / snr / "metrics.json"
        if not p.exists():
            print(f"  {snr}: NOT FOUND")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        test = d.get("test", d)
        acc = test.get("accuracy", 0)
        mf1 = test.get("macro_f1", 0)
        print(f"  {snr}: acc={acc:.4f}, macro_f1={mf1:.4f}")
