"""Aggregate results/cv.csv into results/summary.md and figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
ORDER = ["zeroshot", "ridge", "gnn-identity", "gnn-window", "gnn-contact",
         "gnn-identity+llr", "gnn-window+llr", "gnn-contact+llr"]
# (structure model, control, label) pairs that isolate what 3D contacts add
COMPARISONS = [
    ("gnn-contact", "gnn-identity", "contact vs identity (no zero-shot input)"),
    ("gnn-contact", "gnn-window", "contact vs sequence window (no zero-shot input)"),
    ("gnn-contact+llr", "gnn-identity+llr", "contact vs identity (+ zero-shot input)"),
    ("gnn-contact+llr", "gnn-window+llr", "contact vs sequence window (+ zero-shot input)"),
    ("gnn-contact+llr", "gnn-contact", "zero-shot input added to the contact GNN"),
    ("gnn-contact+llr", "ridge", "contact GNN + zero-shot vs ridge"),
]


def per_assay(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Spearman over split seeds: one row per assay, one column per model."""
    return df.groupby(["dms_id", "model"]).spearman.mean().unstack()[ORDER]


def bootstrap_ci(diff: np.ndarray, n: int = 5000, seed: int = 0):
    rng = np.random.RandomState(seed)
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def main() -> None:
    df = pd.read_csv(RES / "cv.csv")
    pa = per_assay(df)
    types = df.drop_duplicates("dms_id").set_index("dms_id").selection_type
    n = len(pa)

    out = ["# Results", "",
           f"Spearman correlation, out-of-fold with position-held-out 5-fold CV, "
           f"mean over {df.seed.nunique()} split seeds, across {n} ProteinGym assays.", "",
           "## Average over assays", "", "| model | mean Spearman | median |", "|---|---|---|"]
    for m in ORDER:
        out.append(f"| {m} | {pa[m].mean():.3f} | {pa[m].median():.3f} |")

    out += ["", "## By selection type (mean Spearman)", "",
            "| selection type | n assays | " + " | ".join(ORDER) + " |", "|---|---|" + "---|" * len(ORDER)]
    for t, grp in pa.groupby(types.reindex(pa.index)):
        out.append(f"| {t} | {len(grp)} | " + " | ".join(f"{grp[m].mean():.3f}" for m in ORDER) + " |")

    out += ["", "## Does 3D structure help?", "",
            "Paired differences across assays (first model minus second). The interval is a",
            "95% bootstrap over assays; \"wins\" counts assays where the first model is better.", "",
            "| comparison | mean diff | 95% CI | wins |", "|---|---|---|---|"]
    for a, b, label in COMPARISONS:
        d = (pa[a] - pa[b]).dropna().to_numpy()
        lo, hi = bootstrap_ci(d)
        out.append(f"| {label} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {(d > 0).sum()}/{len(d)} |")

    out += ["", "## Per assay", "", "| assay | type | len | " + " | ".join(ORDER) + " |",
            "|---|---|---|" + "---|" * len(ORDER)]
    meta = df.drop_duplicates("dms_id").set_index("dms_id")
    for dms_id, row in pa.iterrows():
        out.append(f"| {dms_id} | {meta.loc[dms_id, 'selection_type']} | {meta.loc[dms_id, 'seq_len']} | "
                   + " | ".join(f"{row[m]:.2f}" for m in ORDER) + " |")
    (RES / "summary.md").write_text("\n".join(out) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.bar(range(len(ORDER)), [pa[m].mean() for m in ORDER], color=["#9aa5b1"] * 2 + ["#4c78a8"] * 3 + ["#d1495b"] * 3)
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=40, ha="right")
    ax.set_ylabel("mean Spearman")
    ax.set_title(f"All {n} assays")
    ax = axes[1]
    d = (pa["gnn-contact+llr"] - pa["gnn-identity+llr"]).sort_values()
    ax.barh(range(len(d)), d.to_numpy(), color=np.where(d > 0, "#d1495b", "#4c78a8"))
    ax.set_yticks(range(len(d)), [i.split("_Tsuboyama")[0].split("_20")[0][:22] for i in d.index], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Spearman: contact GNN − identity GNN (both + zero-shot)")
    ax.set_title("Per-assay effect of adding 3D contacts")
    fig.tight_layout()
    fig.savefig(RES / "fig_structure_effect.png", dpi=150)
    print("\n".join(out))


if __name__ == "__main__":
    main()
