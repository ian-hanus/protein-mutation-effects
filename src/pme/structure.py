"""AlphaFold structure parsing and residue-contact graphs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_ca(pdb_path: Path | str):
    """Return (sequence, CA coordinates [L, 3], pLDDT [L]) from a PDB file.

    AlphaFold writes per-residue pLDDT confidence into the B-factor column.
    """
    seq, coords, plddt = [], [], []
    for line in open(pdb_path):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            seq.append(THREE_TO_ONE.get(line[17:20], "X"))
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            plddt.append(float(line[60:66]))
    return "".join(seq), np.array(coords, dtype=np.float32), np.array(plddt, dtype=np.float32)


def contact_adjacency(coords: np.ndarray, cutoff: float = 8.0) -> np.ndarray:
    """Boolean residue-contact matrix: CA-CA distance below ``cutoff`` angstroms, no self edges."""
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    adj = dist < cutoff
    np.fill_diagonal(adj, False)
    return adj


def normalized_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric GCN normalisation D^-1/2 (A + I) D^-1/2."""
    a = adj.astype(np.float32) + np.eye(len(adj), dtype=np.float32)
    d = a.sum(1) ** -0.5
    return a * d[:, None] * d[None, :]
