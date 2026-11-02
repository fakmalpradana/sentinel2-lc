"""Score checkpoints on the held-out test cells and write runs/eval.md + runs/eval.json.

    uv run python scripts/evaluate.py runs/lgbm.pt runs/unet.pt
"""

import json
import sys
from pathlib import Path

import numpy as np
from train import confusion, load_chips, miou

from s2lc import imagery, model
from s2lc.classes import CLASSES


def evaluate(path: Path, x: np.ndarray, y: np.ndarray) -> dict:
    ck = model.load(path)
    conf = np.zeros((9, 9), np.int64)
    for xi, yi in zip(x, y, strict=True):
        f = imagery.features(xi.astype(np.float32))
        conf += confusion(model.predict(f, ck).astype(np.int64), yi, np.isnan(f[0]))
    tp = np.diag(conf).astype(float)
    prec = tp / np.maximum(conf.sum(0), 1)
    rec = tp / np.maximum(conf.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return {"model": path.name, "kind": ck["kind"], "oa": float(tp.sum() / conf.sum()),
            "miou": miou(conf), "pixels": int(conf.sum()),
            "per_class": {CLASSES[i + 1][1]: {"precision": round(float(prec[i]), 4),
                                              "recall": round(float(rec[i]), 4),
                                              "f1": round(float(f1[i]), 4),
                                              "support": int(conf[i].sum())}
                          for i in range(9)},
            "confusion": conf.tolist()}


def markdown(r: dict) -> str:
    names = [CLASSES[i][1] for i in range(1, 10)]
    out = [f"### {r['model']} ({r['kind']})", "",
           f"Overall accuracy **{r['oa']:.4f}**, mIoU **{r['miou']:.4f}**, "
           f"{r['pixels']:,} test pixels.", "",
           "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for n in names:
        c = r["per_class"][n]
        out.append(f"| {n} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | "
                   f"{c['support']:,} |")
    out += ["", "Confusion matrix (rows = reference, columns = predicted):", "",
            "| | " + " | ".join(names) + " |", "|---" * 10 + "|"]
    for n, row in zip(names, r["confusion"], strict=True):
        out.append(f"| **{n}** | " + " | ".join(f"{v:,}" for v in row) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    x, y = load_chips("test")
    print("test chips", len(x))
    results = [evaluate(Path(p), x, y) for p in sys.argv[1:]]
    Path("runs/eval.json").write_text(json.dumps(results, indent=1))
    Path("runs/eval.md").write_text("\n".join(markdown(r) for r in results))
    for r in results:
        print(f"{r['model']}: OA {r['oa']:.4f} mIoU {r['miou']:.4f}")
