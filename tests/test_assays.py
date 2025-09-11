import pytest

from pme import assays
from pme.assays import single_mutation


def test_single_mutation_found():
    assert single_mutation("ACDEF", "ACGEF") == (2, "D", "G")


def test_no_or_multiple_differences_rejected():
    assert single_mutation("ACDEF", "ACDEF") is None
    assert single_mutation("ACDEF", "GCGEF") is None
    assert single_mutation("ACDEF", "ACDE") is None


needs_data = pytest.mark.skipif(not (assays.RAW / "DMS_substitutions.csv").exists(), reason="ProteinGym not downloaded")


@needs_data
def test_selection_is_deterministic_and_stratified():
    a, b = assays.select_assays(seed=0), assays.select_assays(seed=0)
    assert a == b
    ref = assays.reference().set_index("DMS_id")
    types = [ref.loc[i, "coarse_selection_type"] for i in a]
    for sel_type, n in assays.PER_TYPE.items():
        assert types.count(sel_type) <= n


@needs_data
def test_loaded_assay_positions_match_wild_type():
    assay = assays.load_assay(assays.select_assays(seed=0)[0])
    assert len(assay.table) > 500
    for r in assay.table.head(200).itertuples():
        assert assay.wt_seq[r.pos] == r.wt and r.wt != r.mt
