"""Run the representation ablation on shared splits.

Same model, same splits, four representations. Two splits, because they ask
different questions:

* ``random`` -- copolymers of a comonomer pair appear in both train and test.
  Measures whether a representation can fit the sequence effect at all.
* ``leave_pair_out`` -- entire comonomer pairs are held out (a ``group_split`` on
  ``pair_id``). Measures whether it *generalises* the sequence effect to comonomer
  combinations it has never seen. This is the honest test, and the one where a
  representation that merely memorised per-pair behaviour is exposed.

The scale of the planted sequence effect (std of ``tg - tg_fox``) is reported
alongside as a yardstick -- NOT a lower bound. Because each pair's sequence
sensitivity is learnable from monomer structure, a sequence-blind model that knows
the monomers can score below it (see ``sequence_effect_scale``); what no
sequence-blind model can resolve is which sequence a given sample has.
"""

from __future__ import annotations

import pandas as pd
from polytools import GBMRegressor, group_split, random_split, regression_metrics

from .represent import REPRESENTATIONS, build_representation

__all__ = ["run_ablation", "sequence_effect_scale"]


def sequence_effect_scale(df: pd.DataFrame) -> float:
    """Standard deviation of the planted sequence effect, ``std(tg - tg_fox)``.

    This is the *scale* of what sequence contributes -- a yardstick for the RMSE
    table, **not** a lower bound for sequence-blind models. It is not a bound
    because the pair sensitivity ``delta`` is (by design) learnable from monomer
    structure: a sequence-blind model that knows the monomers can anticipate the
    composition-conditional *mean* of the sequence effect and score below this
    number. What no sequence-blind model can do is resolve *which* sequence a
    given sample has -- that irreducible part is what separates the sequence-blind
    representations from ``plus_sequence`` in the ablation.
    """
    return float((df["tg"] - df["tg_fox"]).std())


def _splits(df: pd.DataFrame, seed: int) -> dict:
    n = len(df)
    return {
        "random": random_split(n, seed=seed),
        "leave_pair_out": group_split(df["pair_id"].to_numpy(), seed=seed),
    }


def run_ablation(
    df: pd.DataFrame,
    seed: int = 0,
    n_estimators: int = 500,
) -> pd.DataFrame:
    """Train each representation under each split; return a tidy results frame."""
    y = df["tg"].to_numpy(dtype=float)
    splits = _splits(df, seed)
    scale = sequence_effect_scale(df)

    rows = []
    for rep_name in REPRESENTATIONS:
        X = build_representation(rep_name).transform(df)
        for split_name, sp in splits.items():
            model = GBMRegressor(
                n_estimators=n_estimators, backend="lightgbm", random_state=seed
            ).fit(X[sp.train], y[sp.train])
            pred = model.predict(X[sp.test])
            m = regression_metrics(y[sp.test], pred)
            rows.append({
                "representation": rep_name,
                "split": split_name,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "sequence_effect_scale": scale,
            })
    return pd.DataFrame(rows)
