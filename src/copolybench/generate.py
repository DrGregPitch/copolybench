"""Generate a controlled copolymer dataset with a known sequence dependence.

We take the bundled homopolymers from :mod:`polytools` as the comonomer library,
form pairs, and for each pair sweep composition ``f`` and blockiness ``chi``. The
target Tg is the Johnston oracle (:mod:`copolybench.oracle`), so every label's
composition and sequence contributions are known exactly.

Each comonomer *pair* is assigned a sequence-sensitivity ``delta`` at generation
time -- some pairs are sequence-neutral (delta 0), others strongly sequence
dependent -- standing in for the unmodelled junction chemistry (specific
comonomer-comonomer interactions) that makes sequence matter for some pairs and
not others. That per-pair structure is what makes the leave-pair-out split a real
test: a model must generalise the *idea* that sequence matters to pairs it has
never seen.

This is a controlled benchmark, labelled as such -- its purpose is to measure
which representations recover a sequence effect we planted, not to be a Tg
dataset. Swap in real copolymer data through the same record structure when you
have it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from polytools import load_toy_tg
from polytools.featurize import DescriptorFeaturizer

from .oracle import tg_fox, tg_johnston
from .sequences import blockiness_index, dyad_fractions, mean_run_lengths

__all__ = ["CopolymerRecord", "generate_dataset", "records_from_frame"]


@dataclass
class CopolymerRecord:
    """One copolymer sample: two comonomers, a composition, a sequence, a Tg."""

    pair_id: int
    name_a: str
    name_b: str
    psmiles_a: str
    psmiles_b: str
    tg_a: float
    tg_b: float
    f: float          # mole fraction of A
    chi: float        # blockiness in [-1, 1]
    delta: float      # pair sequence-sensitivity (hidden from the model)
    F_AA: float
    F_AB: float
    F_BB: float
    blockiness: float
    run_a: float
    run_b: float
    tg_fox: float     # composition-only reference
    tg: float         # Johnston ground truth (the target)


def generate_dataset(
    n_pairs: int = 60,
    points_per_pair: int = 25,
    delta_scale: float = 0.35,
    min_tg_gap: float = 40.0,
    noise_c: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a copolymer dataset from the homopolymer library.

    Parameters
    ----------
    n_pairs
        Number of distinct comonomer pairs to draw.
    points_per_pair
        Copolymer samples per pair, each a random ``(f, chi)``.
    delta_scale
        Sets the magnitude of the sequence effect. Each pair's ``delta`` is a
        function of the comonomers' **polarity mismatch** (TPSA per unit):
        chemically similar comonomers mix near-ideally (delta ~ 0, sequence
        barely matters), dissimilar ones have non-ideal junctions (larger
        ``|delta|``, sequence matters). This makes ``delta`` *predictable from
        monomer structure* -- essential, because otherwise no representation could
        generalise the sequence effect to comonomer pairs held out in the
        leave-pair-out split.
    min_tg_gap
        Only pair comonomers whose homopolymer Tgs differ by at least this, so
        composition has real leverage and the problem is not trivial.
    noise_c
        Optional Gaussian label noise (deg C) added to the Tg, for realism.
    """
    homo = load_toy_tg()
    rng = np.random.default_rng(seed)
    n_homo = len(homo)

    # per-homopolymer polarity (TPSA per repeat unit) drives junction non-ideality
    desc = DescriptorFeaturizer(mode="cap")
    Xd = desc.transform(homo.psmiles)
    polarity = Xd[:, desc.feature_names.index("TPSA_per_unit")]
    pol_spread = float(np.std(polarity)) or 1.0

    def pair_delta(i: int, j: int, jitter: float) -> float:
        """Sequence sensitivity from polarity mismatch, centred so median ~ 0."""
        mismatch = abs(polarity[i] - polarity[j]) / pol_spread
        # tanh centred at ~1 spread: similar pairs -> slightly negative, dissimilar
        # -> positive, bounded by delta_scale; small structural jitter caps accuracy
        return float(delta_scale * np.tanh(mismatch - 1.0) + jitter)

    # draw distinct, sufficiently-different comonomer pairs
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < n_pairs and attempts < 200 * n_pairs:
        attempts += 1
        i, j = rng.integers(0, n_homo, 2)
        key = (min(i, j), max(i, j))
        if i == j or key in seen:
            continue
        if abs(homo.y[i] - homo.y[j]) < min_tg_gap:
            continue
        seen.add(key)
        pairs.append((int(i), int(j)))

    records: list[CopolymerRecord] = []
    for pair_id, (i, j) in enumerate(pairs):
        delta = pair_delta(i, j, jitter=float(rng.normal(0.0, 0.03)))

        for _ in range(points_per_pair):
            f = float(rng.uniform(0.1, 0.9))
            chi = float(rng.uniform(-1.0, 1.0))
            dy = dyad_fractions(f, chi)
            run_a, run_b = mean_run_lengths(f, chi)
            tg = tg_johnston(homo.y[i], homo.y[j], dy, delta)
            if noise_c > 0:
                tg += float(rng.normal(0.0, noise_c))
            records.append(
                CopolymerRecord(
                    pair_id=pair_id,
                    name_a=str(homo.names[i]), name_b=str(homo.names[j]),
                    psmiles_a=str(homo.psmiles[i]), psmiles_b=str(homo.psmiles[j]),
                    tg_a=float(homo.y[i]), tg_b=float(homo.y[j]),
                    f=f, chi=chi, delta=delta,
                    F_AA=dy.F_AA, F_AB=dy.F_AB, F_BB=dy.F_BB,
                    blockiness=blockiness_index(f, chi),
                    run_a=min(run_a, 1e6), run_b=min(run_b, 1e6),
                    tg_fox=tg_fox(homo.y[i], homo.y[j], f),
                    tg=tg,
                )
            )

    return pd.DataFrame([asdict(r) for r in records])


def records_from_frame(df: pd.DataFrame) -> list[CopolymerRecord]:
    """Reconstruct typed records from a generated frame (or a real-data CSV)."""
    return [CopolymerRecord(**{k: row[k] for k in CopolymerRecord.__annotations__})
            for _, row in df.iterrows()]
