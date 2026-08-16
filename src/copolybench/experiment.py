"""Run the representation ablation on shared splits.

Same model, same splits, four representations. Two splits, because they ask
different questions:

* ``random`` -- copolymers of a comonomer pair appear in both train and test.
  Measures whether a representation can fit the sequence effect at all.
* ``leave_pair_out`` -- entire comonomer pairs are held out (a ``group_split`` on
  ``pair_id``). Measures whether it *generalises* the sequence effect to comonomer
  combinations it has never seen. This is the honest test, and the one where a
  representation that merely memorised per-pair behaviour is exposed.

The composition-only floor is reported alongside: the standard deviation of the
planted sequence effect (``tg - tg_fox``) is, by construction, a lower bound on the
RMSE any sequence-blind representation can achieve.
"""

from __future__ import annotations

import pandas as pd
from polytools import GBMRegressor, group_split, random_split, regression_metrics

from .represent import REPRESENTATIONS, build_representation

__all__ = ["run_ablation", "composition_only_floor"]


def composition_only_floor(df: pd.DataFrame) -> float:
    """Lower bound on RMSE for any sequence-blind representation.

    A representation that is a function of composition only can, at best, predict
    the Fox Tg for each composition. The residual it cannot touch is the sequence
    effect ``tg - tg_fox``; its standard deviation is the achievable floor.
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
    floor = composition_only_floor(df)

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
                "sequence_blind_floor": floor,
            })
    return pd.DataFrame(rows)
