import numpy as np
import pytest

from pme import assays
from pme.structure import contact_adjacency, normalized_adjacency, parse_ca


def test_contacts_on_a_line():
    coords = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [30, 0, 0]], dtype=np.float32)
    adj = contact_adjacency(coords, cutoff=8.0)
    assert adj[0, 1] and adj[1, 2] and not adj[0, 2] and not adj[2, 3]
    assert (adj == adj.T).all() and not adj.diagonal().any()


def test_normalized_adjacency_is_symmetric_with_self_loops():
    adj = np.array([[0, 1], [1, 0]], dtype=bool)
    norm = normalized_adjacency(adj)
    assert np.allclose(norm, norm.T)
    assert np.allclose(norm, 0.5)  # both nodes degree 2 after the self loop


needs_data = pytest.mark.skipif(not assays.AF2_DIR.exists(), reason="AF2 structures not downloaded")


@needs_data
def test_every_selected_structure_matches_its_sequence():
    for dms_id in assays.select_assays(seed=0):
        a = assays.load_assay(dms_id)
        seq, coords, plddt = parse_ca(assays.AF2_DIR / a.pdb_file)
        assert seq == a.wt_seq, dms_id
        assert coords.shape == (len(seq), 3) and plddt.shape == (len(seq),)
