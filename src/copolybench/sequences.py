"""Copolymer sequence statistics: composition + blockiness -> dyad fractions.

A copolymer is not a molecule; it is an *ensemble* of chains described by
statistics. The two that matter most for properties are **composition** (how much
of each comonomer) and **sequence** (how the comonomers are arranged along the
chain -- random, alternating, or blocky). Composition alone is what almost every
published copolymer-property model uses. Sequence is the part they drop, and this
project is about when dropping it is a mistake.

The first-order (nearest-neighbour) sequence statistic is the set of **dyad
fractions** -- the fraction of adjacent pairs that are AA, AB (either order), or
BB. For a two-comonomer system these three numbers, together with the composition,
fully specify the first-order sequence, and they are exactly what the Johnston Tg
oracle in :mod:`copolybench.oracle` consumes.

Parameterisation
----------------
We describe a copolymer by its A-composition ``f`` in [0, 1] and a single
**blockiness** ``chi`` in [-1, 1]:

* ``chi = 0``  -> random (Bernoulli): AB junctions as chance alone would give.
* ``chi > 0``  -> blocky: fewer AB junctions; ``chi = 1`` is fully phase-separated
  (two long blocks, essentially no AB junctions).
* ``chi < 0``  -> alternating: more AB junctions; ``chi = -1`` is maximally
  alternating (every minority unit flanked by the majority).

This is a normalised, interpretable index. The dyad relations it must respect are
the standard ones: the A-centred dyads account for all A units, so
``F_AA + F_AB/2 = f`` and ``F_BB + F_AB/2 = 1 - f``, and all three sum to one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DyadFractions", "dyad_fractions", "mean_run_lengths", "blockiness_index"]


@dataclass(frozen=True)
class DyadFractions:
    """Nearest-neighbour pair fractions for a two-comonomer copolymer.

    ``f_AB`` counts both AB and BA junctions. The three fractions sum to 1.
    """

    F_AA: float
    F_AB: float
    F_BB: float

    def as_array(self) -> np.ndarray:
        return np.array([self.F_AA, self.F_AB, self.F_BB], dtype=float)

    def __post_init__(self) -> None:
        total = self.F_AA + self.F_AB + self.F_BB
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(f"dyad fractions must sum to 1, got {total}")
        if min(self.F_AA, self.F_AB, self.F_BB) < -1e-9:
            raise ValueError("dyad fractions must be non-negative")


def dyad_fractions(f: float, chi: float = 0.0) -> DyadFractions:
    """Dyad fractions for A-composition ``f`` and blockiness ``chi`` in [-1, 1].

    At ``chi = 0`` the AB fraction is the Bernoulli value ``2 f (1 - f)``. Positive
    ``chi`` interpolates linearly toward the fully blocky limit (``F_AB = 0``);
    negative ``chi`` toward the maximally alternating limit
    (``F_AB = 2 min(f, 1-f)``). The composition constraint then fixes ``F_AA`` and
    ``F_BB``.
    """
    if not 0.0 <= f <= 1.0:
        raise ValueError(f"composition f must be in [0, 1], got {f}")
    if not -1.0 <= chi <= 1.0:
        raise ValueError(f"blockiness chi must be in [-1, 1], got {chi}")

    f_ab_random = 2.0 * f * (1.0 - f)
    if chi >= 0:  # toward blocky: F_AB shrinks to 0
        f_ab = f_ab_random * (1.0 - chi)
    else:  # toward alternating: F_AB grows to its maximum
        f_ab_max = 2.0 * min(f, 1.0 - f)
        f_ab = f_ab_random + (f_ab_max - f_ab_random) * (-chi)

    f_aa = f - f_ab / 2.0
    f_bb = (1.0 - f) - f_ab / 2.0
    # clamp tiny negative round-off at the extremes
    f_aa = max(f_aa, 0.0)
    f_bb = max(f_bb, 0.0)
    # renormalise to kill any 1e-16 drift from the clamp
    total = f_aa + f_ab + f_bb
    return DyadFractions(f_aa / total, f_ab / total, f_bb / total)


def mean_run_lengths(f: float, chi: float = 0.0) -> tuple[float, float]:
    """Number-average run lengths of A and B blocks.

    A run is a maximal stretch of identical units. The mean A-run length is the
    number of A units divided by the number of AB (A-to-B) junctions per A unit;
    in dyad terms ``L_A = 2 F_A / F_AB`` where ``F_A = f``. Random gives the
    familiar ``L_A = 1/(1-f)``; blocky drives both run lengths up; alternating
    drives them toward 1.
    """
    d = dyad_fractions(f, chi)
    if d.F_AB <= 0:
        return (float("inf"), float("inf"))
    return (2.0 * f / d.F_AB, 2.0 * (1.0 - f) / d.F_AB)


def blockiness_index(f: float, chi: float = 0.0) -> float:
    """Ratio of actual AB fraction to the random (Bernoulli) AB fraction.

    1.0 is random, <1 blocky, >1 alternating. A representation that is handed this
    single number already knows more about the chain than a composition-only model
    ever can -- which is the crux of the whole comparison.
    """
    f_ab_random = 2.0 * f * (1.0 - f)
    if f_ab_random <= 0:  # homopolymer limit
        return 1.0
    return dyad_fractions(f, chi).F_AB / f_ab_random
