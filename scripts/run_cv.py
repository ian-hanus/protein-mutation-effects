"""Evaluate all models on the selected assays with position-held-out CV. Resumable."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pme import assays, esm, fit  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--esm", default=esm.DEFAULT_MODEL)
parser.add_argument("--models", nargs="+", default=fit.MODELS)
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--weight-decay", type=float, default=1e-3)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--out", default="results/cv.csv")
args = parser.parse_args()

torch.set_num_threads(args.threads)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
done = set(pd.read_csv(out).dms_id) if out.exists() else set()

cfg = fit.FitConfig(dim=args.dim, dropout=args.dropout, weight_decay=args.weight_decay, lr=args.lr)
tok, model = esm.load_model(args.esm)
for dms_id in assays.select_assays(seed=0):
    if dms_id in done:
        continue
    ctx = fit.make_context(assays.load_assay(dms_id), tok, model)
    rows = fit.evaluate_assay(ctx, args.models, seeds=args.seeds, cfg=cfg)
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    best = pd.DataFrame(rows).groupby("model").spearman.mean().sort_values(ascending=False)
    print(f"{dms_id:44} best={best.index[0]} ({best.iloc[0]:+.3f})  zeroshot={best.get('zeroshot', float('nan')):+.3f}", flush=True)
