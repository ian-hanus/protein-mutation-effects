"""ESM-2 wild-type embeddings and zero-shot (wild-type marginal) mutation scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .assays import Assay

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA)}
DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"


def load_model(name: str = DEFAULT_MODEL):
    from transformers import AutoTokenizer, EsmForMaskedLM

    tok = AutoTokenizer.from_pretrained(name)
    model = EsmForMaskedLM.from_pretrained(name).eval()
    return tok, model


@torch.no_grad()
def embed_wild_type(seq: str, tok, model) -> tuple[np.ndarray, np.ndarray]:
    """One forward pass over the unmasked wild type.

    Returns per-residue embeddings [L, D] and log-probabilities over the 20 amino
    acids [L, 20]. The special start/end tokens are stripped.
    """
    enc = tok(seq, return_tensors="pt")
    out = model(**enc, output_hidden_states=True)
    emb = out.hidden_states[-1][0, 1:-1]
    logp = torch.log_softmax(out.logits[0, 1:-1], dim=-1)
    aa_ids = [tok.convert_tokens_to_ids(a) for a in AA]
    return emb.numpy(), logp[:, aa_ids].numpy()


def get_features(assay: Assay, tok=None, model=None, model_name: str = DEFAULT_MODEL):
    """Cached (embeddings, log-probs) for an assay's wild-type sequence."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{model_name.split('/')[-1]}__{assay.dms_id}.npz"
    if path.exists():
        z = np.load(path)
        return z["emb"], z["logp"]
    if tok is None:
        tok, model = load_model(model_name)
    emb, logp = embed_wild_type(assay.wt_seq, tok, model)
    np.savez_compressed(path, emb=emb, logp=logp)
    return emb, logp


def zero_shot_scores(table, logp: np.ndarray) -> np.ndarray:
    """Wild-type marginal log-likelihood ratio: log p(mutant) - log p(wild type) at the site."""
    pos = table.pos.to_numpy()
    mt = table.mt.map(AA_INDEX).to_numpy()
    wt = table.wt.map(AA_INDEX).to_numpy()
    return logp[pos, mt] - logp[pos, wt]
