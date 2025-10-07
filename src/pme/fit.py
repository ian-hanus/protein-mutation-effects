"""Per-assay fitting: ridge baseline and GNN variants, evaluated out-of-fold."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

from . import esm
from .assays import AF2_DIR, Assay
from .cv import position_folds, spearman
from .models import ADJACENCY_KINDS, MutationGNN, build_adjacency
from .structure import parse_ca


@dataclass
class FitConfig:
    dim: int = 64
    depth: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-3
    epochs: int = 300
    patience: int = 30
    val_frac: float = 0.1


@dataclass
class Context:
    """Everything about one assay that models need, computed once."""

    assay: Assay
    node_x: torch.Tensor           # [L, D+1]: ESM embedding + pLDDT/100
    coords: np.ndarray
    llr: np.ndarray                # zero-shot score per mutation row
    pos: torch.Tensor
    wt: torch.Tensor
    mt: torch.Tensor
    y: np.ndarray
    adjacency: dict = field(default_factory=dict)

    def adj(self, kind: str) -> torch.Tensor:
        if kind not in self.adjacency:
            self.adjacency[kind] = build_adjacency(kind, self.coords)
        return self.adjacency[kind]


def make_context(assay: Assay, tok=None, model=None) -> Context:
    emb, logp = esm.get_features(assay, tok, model)
    _, coords, plddt = parse_ca(AF2_DIR / assay.pdb_file)
    t = assay.table
    node_x = torch.tensor(np.concatenate([emb, plddt[:, None] / 100.0], axis=1), dtype=torch.float32)
    return Context(
        assay, node_x, coords, esm.zero_shot_scores(t, logp),
        torch.tensor(t.pos.to_numpy()), torch.tensor(t.wt.map(esm.AA_INDEX).to_numpy()),
        torch.tensor(t.mt.map(esm.AA_INDEX).to_numpy()), t.score.to_numpy(dtype=np.float64),
    )


def _inner_split(ctx: Context, train_idx: np.ndarray, frac: float, seed: int):
    """Hold out a fraction of the *training positions* for early stopping."""
    positions = ctx.pos.numpy()[train_idx]
    unique = np.unique(positions)
    rng = np.random.RandomState(seed)
    val_pos = set(rng.choice(unique, size=max(1, int(frac * len(unique))), replace=False))
    is_val = np.array([p in val_pos for p in positions])
    return train_idx[~is_val], train_idx[is_val]


def fit_gnn(ctx: Context, train_idx, test_idx, kind: str, use_llr: bool, seed: int, cfg: FitConfig) -> np.ndarray:
    torch.manual_seed(seed)
    tr, va = _inner_split(ctx, train_idx, cfg.val_frac, seed)
    mean, std = ctx.y[tr].mean(), ctx.y[tr].std() + 1e-8
    y = torch.tensor((ctx.y - mean) / std, dtype=torch.float32)
    llr = torch.tensor(ctx.llr, dtype=torch.float32)
    a = ctx.adj(kind)

    model = MutationGNN(ctx.node_x.shape[1], cfg.dim, cfg.depth, cfg.dropout, use_llr)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def forward(idx):
        i = torch.as_tensor(idx)
        return model(ctx.node_x, a, ctx.pos[i], ctx.wt[i], ctx.mt[i], llr[i] if use_llr else None)

    best, best_state, stale = float("inf"), None, 0
    for _ in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        loss = ((forward(tr) - y[tr]) ** 2).mean()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val = ((forward(va) - y[va]) ** 2).mean().item()
        if val < best - 1e-5:
            best, best_state, stale = val, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return forward(test_idx).numpy()


def fit_ridge(ctx: Context, train_idx, test_idx) -> np.ndarray:
    """Linear baseline on [ESM embedding of the site, wt one-hot, mutant one-hot, zero-shot score]."""
    emb = ctx.node_x.numpy()[ctx.pos.numpy()]
    eye = np.eye(len(esm.AA_INDEX))
    X = np.concatenate([emb, eye[ctx.wt.numpy()], eye[ctx.mt.numpy()], ctx.llr[:, None]], axis=1)
    model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X[train_idx], ctx.y[train_idx])
    return model.predict(X[test_idx])


MODELS = ["zeroshot", "ridge"] + [f"gnn-{k}{'+llr' if u else ''}" for u in (False, True) for k in ADJACENCY_KINDS]


def predict(ctx: Context, model_name: str, train_idx, test_idx, seed: int, cfg: FitConfig) -> np.ndarray:
    if model_name == "zeroshot":
        return ctx.llr[test_idx]
    if model_name == "ridge":
        return fit_ridge(ctx, train_idx, test_idx)
    kind = model_name.split("-")[1].split("+")[0]
    return fit_gnn(ctx, train_idx, test_idx, kind, model_name.endswith("+llr"), seed, cfg)


def evaluate_assay(ctx: Context, models: list[str], seeds: int = 3, n_folds: int = 5, cfg: FitConfig | None = None):
    """Out-of-fold Spearman per (model, split seed): predictions are pooled across folds."""
    cfg = cfg or FitConfig()
    rows = []
    for seed in range(seeds):
        oof = {m: np.zeros(len(ctx.y)) for m in models}
        for tr, te in position_folds(ctx.pos.numpy(), n_folds, seed):
            for m in models:
                oof[m][te] = predict(ctx, m, tr, te, seed, cfg)
        for m in models:
            rows.append({
                "dms_id": ctx.assay.dms_id, "selection_type": ctx.assay.selection_type,
                "seq_len": len(ctx.assay.wt_seq), "n_mutations": len(ctx.y),
                "model": m, "seed": seed, "spearman": spearman(ctx.y, oof[m]),
            })
    return rows
