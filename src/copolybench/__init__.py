"""copolybench -- when does copolymer *sequence* matter for property prediction?

A controlled benchmark that plants a known, tunable sequence effect in copolymer
Tg (via the Johnston dyad model) and measures which representations recover it.
The naive composition-weighted encoding that most published copolymer work uses is
structurally blind to sequence; this quantifies exactly what that costs and when.

Built on the ``polytools`` evaluation harness.
"""

from .generate import CopolymerRecord, generate_dataset, records_from_frame
from .oracle import (
    C_TO_K,
    harmonic_mean_tg,
    sequence_effect,
    tg_fox,
    tg_johnston,
)
from .sequences import (
    DyadFractions,
    blockiness_index,
    dyad_fractions,
    mean_run_lengths,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # sequences
    "DyadFractions", "dyad_fractions", "mean_run_lengths", "blockiness_index",
    # oracle
    "C_TO_K", "harmonic_mean_tg", "tg_fox", "tg_johnston", "sequence_effect",
    # generate
    "CopolymerRecord", "generate_dataset", "records_from_frame",
]
