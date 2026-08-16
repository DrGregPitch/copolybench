"""The representation ladder: four ways to encode a copolymer, increasingly honest.

All four feed the *same* model and the *same* splits, so any difference in error is
attributable to the representation alone -- the clean ablation the whole project is
built to run.

1. ``composition_weighted`` -- the naive baseline nearly every published copolymer
   model uses: a composition-weighted average of the two homopolymer feature
   vectors, ``f * v_A + (1-f) * v_B``. By construction it is a function of
   composition only; a block and a random copolymer of the same composition map to
   the *identical* vector. It cannot represent sequence, full stop.
2. ``monomers_composition`` -- concatenate both monomers' features and the
   composition, ``[v_A, v_B, f]``. Strictly more expressive than the weighted
   average (it keeps the two monomers distinct), but still a function of
   composition only. Also sequence-blind.
3. ``plus_sequence`` -- add first-order sequence statistics: the AB dyad fraction,
   the blockiness index, and the mean run lengths. This is the first representation
   that can *see* whether a chain is blocky or alternating.
4. ``sequence_stats_only`` -- composition and sequence statistics with no monomer
   structure. A diagnostic: it isolates how far you get from the statistics alone,
   without knowing what the comonomers are.

The prediction the benchmark tests: 1 and 2 are floored at the variance of the
planted sequence effect; 3 breaks through it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from polytools.featurize import DescriptorFeaturizer

__all__ = ["Representation", "REPRESENTATIONS", "build_representation"]


class Representation:
    """Turns a copolymer dataframe into a feature matrix, with a cached monomer encoder."""

    def __init__(self, name: str, kind: str, mode: str = "cap") -> None:
        self.name = name
        self.kind = kind
        self._desc = DescriptorFeaturizer(mode=mode)
        self._cache: dict[str, np.ndarray] = {}

    def _monomer(self, psmiles: str) -> np.ndarray:
        if psmiles not in self._cache:
            self._cache[psmiles] = self._desc.transform([psmiles])[0]
        return self._cache[psmiles]

    def _monomers(self, smis: Sequence[str]) -> np.ndarray:
        # warm the cache for all unique monomers in one featurizer call
        uniq = [s for s in dict.fromkeys(smis) if s not in self._cache]
        if uniq:
            vecs = self._desc.transform(uniq)
            for s, v in zip(uniq, vecs):
                self._cache[s] = v
        return np.vstack([self._cache[s] for s in smis])

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        vA = self._monomers(df["psmiles_a"].tolist())
        vB = self._monomers(df["psmiles_b"].tolist())
        f = df["f"].to_numpy(dtype=float)[:, None]

        if self.kind == "composition_weighted":
            return f * vA + (1.0 - f) * vB

        if self.kind == "monomers_composition":
            return np.hstack([vA, vB, f])

        if self.kind == "plus_sequence":
            seq = df[["F_AB", "blockiness", "run_a", "run_b"]].to_numpy(dtype=float)
            return np.hstack([vA, vB, f, seq])

        if self.kind == "sequence_stats_only":
            seq = df[["F_AB", "blockiness", "run_a", "run_b"]].to_numpy(dtype=float)
            return np.hstack([f, seq])

        raise ValueError(f"unknown representation kind {self.kind!r}")


#: The ladder, in reporting order.
REPRESENTATIONS = {
    "composition_weighted": "composition-weighted homopolymer features (naive)",
    "monomers_composition": "both monomers + composition (sequence-blind)",
    "sequence_stats_only": "composition + sequence stats, no monomer identity",
    "plus_sequence": "both monomers + composition + sequence stats",
}


def build_representation(kind: str, mode: str = "cap") -> Representation:
    if kind not in REPRESENTATIONS:
        raise ValueError(f"unknown representation {kind!r}; choose from {list(REPRESENTATIONS)}")
    return Representation(kind, kind, mode=mode)
