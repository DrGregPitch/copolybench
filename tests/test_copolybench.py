"""Tests for copolybench, organised by the invariant they protect.

Several pin the *physics* of the controlled benchmark -- that Johnston collapses
onto Fox when the junction is neutral, that the naive representation is provably
blind to sequence -- because those are the claims the whole project rests on.
"""

from __future__ import annotations

import numpy as np
import pytest

from copolybench import (
    blockiness_index,
    dyad_fractions,
    generate_dataset,
    mean_run_lengths,
    sequence_effect,
    tg_fox,
    tg_johnston,
)
from copolybench.experiment import run_ablation, sequence_effect_scale
from copolybench.represent import build_representation

# --------------------------------------------------------------------------
# sequences
# --------------------------------------------------------------------------

def test_dyads_sum_to_one_across_grid():
    for f in np.linspace(0, 1, 11):
        for chi in np.linspace(-1, 1, 11):
            d = dyad_fractions(float(f), float(chi))
            assert d.F_AA + d.F_AB + d.F_BB == pytest.approx(1.0)
            assert min(d.F_AA, d.F_AB, d.F_BB) >= -1e-9


def test_random_limit_is_bernoulli():
    f = 0.3
    d = dyad_fractions(f, 0.0)
    assert d.F_AB == pytest.approx(2 * f * (1 - f))


def test_alternating_and_blocky_limits():
    alt = dyad_fractions(0.5, -1.0)
    assert alt.F_AB == pytest.approx(1.0) and alt.F_AA == pytest.approx(0.0)
    blocky = dyad_fractions(0.5, 1.0)
    assert blocky.F_AB == pytest.approx(0.0)
    assert blocky.F_AA == pytest.approx(0.5) and blocky.F_BB == pytest.approx(0.5)


def test_run_lengths_and_blockiness_directions():
    assert mean_run_lengths(0.5, 0.0)[0] == pytest.approx(2.0)   # random
    assert mean_run_lengths(0.5, 0.8)[0] > 5                      # blocky: long runs
    assert mean_run_lengths(0.5, -1.0)[0] == pytest.approx(1.0)   # alternating
    assert blockiness_index(0.5, 0.0) == pytest.approx(1.0)
    assert blockiness_index(0.5, 1.0) == pytest.approx(0.0)
    assert blockiness_index(0.5, -1.0) == pytest.approx(2.0)


@pytest.mark.parametrize("f,chi", [(-0.1, 0), (1.1, 0), (0.5, 2.0), (0.5, -2.0)])
def test_invalid_inputs_raise(f, chi):
    with pytest.raises(ValueError):
        dyad_fractions(f, chi)


# --------------------------------------------------------------------------
# oracle
# --------------------------------------------------------------------------

def test_fox_endpoints_and_betweenness():
    assert tg_fox(100.0, -125.0, 1.0) == pytest.approx(100.0)
    assert tg_fox(100.0, -125.0, 0.0) == pytest.approx(-125.0)
    mid = tg_fox(100.0, -125.0, 0.5)
    assert -125.0 < mid < 100.0


def test_johnston_reduces_to_fox_when_neutral():
    """delta=0 makes the AB junction the harmonic mean -> sequence has no effect."""
    for chi in (-1.0, -0.3, 0.0, 0.5, 1.0):
        d = dyad_fractions(0.5, chi)
        j = tg_johnston(100.0, -125.0, d, delta=0.0)
        assert j == pytest.approx(tg_fox(100.0, -125.0, 0.5), abs=1e-9)


def test_sequence_effect_scales_with_ab_fraction():
    """With delta != 0, the effect grows with the AB dyad fraction; zero when blocky."""
    blocky = sequence_effect(100.0, -125.0, 0.5, chi=1.0, delta=0.3)
    rand = sequence_effect(100.0, -125.0, 0.5, chi=0.0, delta=0.3)
    alt = sequence_effect(100.0, -125.0, 0.5, chi=-1.0, delta=0.3)
    assert blocky == pytest.approx(0.0, abs=1e-9)
    assert abs(rand) > 1.0
    assert abs(alt) > abs(rand)


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_pairs=40, points_per_pair=15, noise_c=2.0, seed=0)


def test_dataset_shape_and_columns(dataset):
    assert len(dataset) == 40 * 15
    assert dataset.pair_id.nunique() == 40
    for col in ("psmiles_a", "psmiles_b", "f", "chi", "F_AB", "tg", "tg_fox", "delta"):
        assert col in dataset.columns


def test_dataset_has_a_real_sequence_signal(dataset):
    """The planted sequence effect must be a meaningful fraction of Tg variance."""
    scale = sequence_effect_scale(dataset)
    assert scale > 10.0                       # sequence matters
    assert scale < dataset.tg.std()           # but is not the whole story


def test_delta_varies_with_pair_chemistry(dataset):
    """delta is a function of the comonomers, so it differs across pairs."""
    per_pair = dataset.groupby("pair_id").delta.first()
    assert per_pair.std() > 0.02


# --------------------------------------------------------------------------
# representations  (the crux)
# --------------------------------------------------------------------------

def test_composition_weighted_is_blind_to_sequence(dataset):
    """Two copolymers, same monomers and composition, different blockiness, must map
    to the *identical* naive feature vector. This is the failure the project measures."""
    rep = build_representation("composition_weighted")
    row = dataset.iloc[[0]].copy()
    a = row.copy()
    a[["chi", "F_AB", "blockiness"]] = [-0.9, 0.9, 1.8]
    b = row.copy()
    b[["chi", "F_AB", "blockiness"]] = [0.9, 0.05, 0.1]
    np.testing.assert_allclose(rep.transform(a), rep.transform(b))


def test_plus_sequence_distinguishes_blockiness(dataset):
    """The sequence-aware representation must NOT be invariant to blockiness."""
    rep = build_representation("plus_sequence")
    row = dataset.iloc[[0]].copy()
    a = row.copy()
    a[["F_AB", "blockiness"]] = [0.9, 1.8]
    b = row.copy()
    b[["F_AB", "blockiness"]] = [0.05, 0.1]
    assert not np.allclose(rep.transform(a), rep.transform(b))


def test_representation_shapes(dataset):
    n = len(dataset)
    assert build_representation("composition_weighted").transform(dataset).shape[0] == n
    assert build_representation("plus_sequence").transform(dataset).shape[0] == n


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------

def test_sequence_representation_beats_naive_on_random_split():
    """The headline: adding sequence statistics lowers random-split RMSE.

    Uses a larger dataset than the module fixture -- the effect is real but only
    crisp above ~100 comonomer pairs, below which small-sample noise can hide it.
    """
    df = generate_dataset(n_pairs=100, points_per_pair=15, noise_c=2.0, seed=0)
    res = run_ablation(df, seed=0, n_estimators=200)
    rnd = res[res.split == "random"].set_index("representation")["rmse"]
    assert rnd["plus_sequence"] < rnd["monomers_composition"]
    assert rnd["plus_sequence"] < rnd["composition_weighted"]
