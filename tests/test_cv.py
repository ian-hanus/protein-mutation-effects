import numpy as np

from pme.cv import position_folds, spearman

POSITIONS = np.repeat(np.arange(40), 19)  # 40 sites x 19 substitutions


def test_folds_partition_rows():
    tests = [te for _, te in position_folds(POSITIONS, n_folds=5, seed=0)]
    all_test = np.concatenate(tests)
    assert sorted(all_test) == list(range(len(POSITIONS)))


def test_no_position_shared_between_train_and_test():
    for tr, te in position_folds(POSITIONS, n_folds=5, seed=1):
        assert not (set(POSITIONS[tr]) & set(POSITIONS[te]))


def test_seed_changes_assignment():
    a = [te.tolist() for _, te in position_folds(POSITIONS, seed=0)]
    b = [te.tolist() for _, te in position_folds(POSITIONS, seed=1)]
    assert a != b


def test_spearman_known_and_degenerate():
    assert np.isclose(spearman(np.arange(10), np.arange(10) ** 2), 1.0)
    assert np.isnan(spearman(np.arange(10), np.ones(10)))
