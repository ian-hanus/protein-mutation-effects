"""Mutation-effect models: one GNN architecture, three adjacency controls.

The architecture is held fixed and only the graph changes:
  identity - every residue sees only itself (no neighbours)
  window   - sequence neighbours within +/-4 residues (context, but no 3D)
  contact  - 3D residue contacts from the AlphaFold structure
so any gain of ``contact`` over ``identity``/``window`` is due to structure,
not to model capacity.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .esm import AA_INDEX
from .structure import contact_adjacency, normalized_adjacency

ADJACENCY_KINDS = ("identity", "window", "contact")


def build_adjacency(kind: str, coords: np.ndarray, window: int = 4, cutoff: float = 8.0) -> torch.Tensor:
    n = len(coords)
    if kind == "identity":
        adj = np.zeros((n, n), dtype=bool)
    elif kind == "window":
        idx = np.arange(n)
        adj = (np.abs(idx[:, None] - idx[None, :]) <= window) & (idx[:, None] != idx[None, :])
    elif kind == "contact":
        adj = contact_adjacency(coords, cutoff)
    else:
        raise ValueError(f"unknown adjacency: {kind}")
    return torch.tensor(normalized_adjacency(adj))


class MutationGNN(nn.Module):
    def __init__(self, in_dim: int, dim: int = 64, depth: int = 2, dropout: float = 0.2, use_llr: bool = False):
        super().__init__()
        self.use_llr = use_llr
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        head_in = dim + 2 * len(AA_INDEX) + int(use_llr)
        self.head = nn.Sequential(nn.Linear(head_in, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def encode(self, node_x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(self.norm(node_x)))
        for lin in self.layers:
            h = h + self.dropout(torch.relu(a_norm @ lin(h)))
        return h

    def forward(self, node_x, a_norm, pos, wt, mt, llr=None) -> torch.Tensor:
        h = self.encode(node_x, a_norm)[pos]
        feats = [h, nn.functional.one_hot(wt, len(AA_INDEX)).float(), nn.functional.one_hot(mt, len(AA_INDEX)).float()]
        if self.use_llr:
            feats.append(llr.unsqueeze(1))
        return self.head(torch.cat(feats, dim=1)).squeeze(-1)
