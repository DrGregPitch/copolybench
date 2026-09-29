r"""A controlled copolymer-Tg oracle where sequence matters by a tunable amount.

To study *representations* we need ground truth we fully control. This module is
that ground truth: a physically motivated copolymer glass-transition model in
which the sequence contribution can be dialled from zero to large, so we can prove
which representations capture it and which structurally cannot.

Two literature models, and the exact relationship between them
-------------------------------------------------------------
* **Fox** (composition only): ``1/Tg = w_A/Tg_A + w_B/Tg_B``. Depends on how *much*
  of each comonomer is present, not how it is arranged. This is the ceiling a
  composition-only representation can reach.
* **Johnston** (sequence aware): ``1/Tg = F_AA/Tg_AA + F_AB/Tg_AB + F_BB/Tg_BB``,
  a sum over *dyad* fractions with a dyad-specific Tg for the AB junction.

Substituting the composition constraint ``F_AA = f - F_AB/2``,
``F_BB = (1-f) - F_AB/2`` into Johnston gives, exactly:

    1/Tg = [f/Tg_A + (1-f)/Tg_B]  +  F_AB * (1/Tg_AB - 1/(2 Tg_A) - 1/(2 Tg_B))
           \_________ Fox _________/    \_______ the entire sequence effect _______/

So the whole sequence dependence lives in the ``F_AB`` coefficient. When the AB
junction Tg equals the harmonic mean of the two homopolymer Tgs,
``Tg_AB = 2 Tg_A Tg_B / (Tg_A + Tg_B)``, that coefficient is **exactly zero** and
Johnston collapses onto Fox -- sequence stops mattering. We parameterise the AB
junction as ``Tg_AB = harmonic_mean * (1 + delta)``: ``delta = 0`` is a
sequence-neutral pair, and ``|delta|`` is a clean dial for how much sequence
matters. Physically, ``delta > 0`` is a favourable comonomer interaction (say
cross-pair hydrogen bonding) stiffening junctions; ``delta < 0`` an antagonistic
one.

All temperatures in the equations are absolute (Kelvin) -- Fox and Johnston are
reciprocal-temperature relations and are simply wrong in Celsius. Inputs and
outputs here are in Celsius for convenience; the conversion happens internally.
"""

from __future__ import annotations

from .sequences import DyadFractions, dyad_fractions

__all__ = [
    "C_TO_K",
    "harmonic_mean_tg",
    "tg_fox",
    "tg_johnston",
    "sequence_effect",
]

C_TO_K = 273.15


def harmonic_mean_tg(tg_a_c: float, tg_b_c: float) -> float:
    """Harmonic mean of two Tgs (in C), computed in Kelvin, returned in C.

    This is the AB-junction Tg that makes the sequence contribution vanish, i.e.
    the value at which Johnston equals Fox.
    """
    a, b = tg_a_c + C_TO_K, tg_b_c + C_TO_K
    hm_k = 2.0 * a * b / (a + b)
    return hm_k - C_TO_K


def tg_fox(tg_a_c: float, tg_b_c: float, f: float) -> float:
    """Composition-only Fox Tg (C). ``f`` is the mole fraction of A.

    Uses mole fractions as weights -- the equal-unit-mass simplification
    appropriate to this controlled benchmark; real applications weight by mass.
    """
    a, b = tg_a_c + C_TO_K, tg_b_c + C_TO_K
    inv = f / a + (1.0 - f) / b
    return 1.0 / inv - C_TO_K


def tg_johnston(
    tg_a_c: float,
    tg_b_c: float,
    dyads: DyadFractions,
    delta: float = 0.0,
) -> float:
    """Sequence-aware Johnston Tg (C).

    ``delta`` sets the AB-junction Tg as ``harmonic_mean * (1 + delta)`` in Kelvin.
    ``delta = 0`` recovers Fox exactly (for the same composition); non-zero
    ``delta`` introduces a sequence effect proportional to the AB dyad fraction.
    """
    a, b = tg_a_c + C_TO_K, tg_b_c + C_TO_K
    hm_k = 2.0 * a * b / (a + b)
    tg_ab_k = hm_k * (1.0 + delta)
    if tg_ab_k <= 0:
        raise ValueError(
            f"delta={delta} makes the AB-junction Tg non-positive "
            f"({tg_ab_k:.1f} K); an absolute temperature must be > 0. "
            "Keep delta > -1 (physically, well inside (-1, 1))."
        )

    inv = dyads.F_AA / a + dyads.F_AB / tg_ab_k + dyads.F_BB / b
    return 1.0 / inv - C_TO_K


def sequence_effect(
    tg_a_c: float,
    tg_b_c: float,
    f: float,
    chi: float,
    delta: float,
) -> float:
    """Tg shift attributable purely to sequence: Johnston(f, chi) - Fox(f).

    This is the number a composition-only representation cannot resolve per-sample
    (it cannot tell which sequence a given copolymer has). Its spread sets the
    SCALE of the sequence contribution -- a yardstick, not a strict floor, since a
    monomer-aware model can still anticipate the composition-conditional mean.
    """
    dyads = dyad_fractions(f, chi)
    return tg_johnston(tg_a_c, tg_b_c, dyads, delta) - tg_fox(tg_a_c, tg_b_c, f)
