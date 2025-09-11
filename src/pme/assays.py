"""ProteinGym assay selection and loading (single substitutions only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DMS_DIR = RAW / "dms" / "DMS_ProteinGym_substitutions"
AF2_DIR = RAW / "af2" / "ProteinGym_AF2_structures"

# Assays drawn per ProteinGym selection type. Fixed up front so the subset is a
# seeded random sample rather than hand-picked.
PER_TYPE = {"Stability": 5, "Activity": 5, "Binding": 4, "OrganismalFitness": 4, "Expression": 2}


def reference() -> pd.DataFrame:
    return pd.read_csv(RAW / "DMS_substitutions.csv")


def eligible(ref: pd.DataFrame, max_len: int = 350, min_mut: int = 800, max_mut: int = 6000) -> pd.DataFrame:
    """Assays that are small enough for CPU work and whose structure spans the whole sequence."""
    full_structure = ref.pdb_range == ref.seq_len.map(lambda n: f"1-{n}")
    ok = (
        full_structure
        & (ref.seq_len <= max_len)
        & ref.DMS_number_single_mutants.between(min_mut, max_mut)
        & ref.pdb_file.notna()
    )
    return ref[ok]


def select_assays(seed: int = 0, per_type: dict[str, int] = PER_TYPE, **kwargs) -> list[str]:
    ref = eligible(reference(), **kwargs)
    rng = np.random.RandomState(seed)
    chosen = []
    for sel_type, n in per_type.items():
        ids = sorted(ref[ref.coarse_selection_type == sel_type].DMS_id)
        take = min(n, len(ids))
        chosen += list(rng.choice(ids, size=take, replace=False))
    return chosen


def single_mutation(wt_seq: str, mut_seq: str):
    """Return (position0, wt, mt) if the sequences differ at exactly one site, else None."""
    if len(wt_seq) != len(mut_seq):
        return None
    diff = [i for i, (a, b) in enumerate(zip(wt_seq, mut_seq)) if a != b]
    if len(diff) != 1:
        return None
    i = diff[0]
    return i, wt_seq[i], mut_seq[i]


@dataclass
class Assay:
    dms_id: str
    uniprot_id: str
    selection_type: str
    wt_seq: str
    pdb_file: str
    table: pd.DataFrame  # columns: pos, wt, mt, score  (pos is 0-based into wt_seq)


def load_assay(dms_id: str) -> Assay:
    ref = reference()
    row = ref[ref.DMS_id == dms_id].iloc[0]
    df = pd.read_csv(DMS_DIR / row.DMS_filename)
    df = df[~df.mutant.str.contains(":")]  # singles only

    parsed = [single_mutation(row.target_seq, s) for s in df.mutated_sequence]
    keep = [p is not None for p in parsed]
    df, parsed = df[keep], [p for p in parsed if p is not None]
    table = pd.DataFrame(parsed, columns=["pos", "wt", "mt"], index=df.index)
    table["score"] = df.DMS_score.to_numpy()
    table = table.dropna().reset_index(drop=True)
    return Assay(dms_id, row.UniProt_ID, row.coarse_selection_type, row.target_seq, row.pdb_file, table)
