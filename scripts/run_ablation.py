#!/usr/bin/env python3
"""Generate the controlled copolymer set and run the representation ablation.

    python scripts/run_ablation.py --outdir results

Writes the results table and two figures:

* ``ablation_rmse.png`` -- test RMSE per representation and split, with the
  composition-only floor drawn in. The gap between the naive and sequence-aware
  bars on the random split is the headline.
* ``blindness.png`` -- the naive representation's residuals plotted against the
  planted sequence effect. They line up, which is the mechanistic proof that what
  a composition-only encoding gets wrong *is* the sequence it cannot see.

Controlled benchmark -- labelled as such. Swap in real copolymer data through the
same record structure when you have it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from polytools import GBMRegressor, random_split, silence_rdkit

from copolybench import generate_dataset
from copolybench.experiment import composition_only_floor, run_ablation
from copolybench.represent import build_representation

ORDER = ["composition_weighted", "monomers_composition",
         "sequence_stats_only", "plus_sequence"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results", type=Path)
    p.add_argument("--n-pairs", default=300, type=int)
    p.add_argument("--points-per-pair", default=18, type=int)
    p.add_argument("--noise", default=2.0, type=float)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    silence_rdkit()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(
        n_pairs=args.n_pairs, points_per_pair=args.points_per_pair,
        noise_c=args.noise, seed=args.seed,
    )
    floor = composition_only_floor(df)
    print(f"Dataset: {len(df)} copolymers from {df.pair_id.nunique()} comonomer pairs")
    print(f"Tg std {df.tg.std():.1f} C | sequence-effect std (composition-blind floor) "
          f"{floor:.1f} C | label noise {args.noise} C")

    res = run_ablation(df, seed=args.seed)
    res.to_csv(args.outdir / "ablation_results.csv", index=False)
    piv = res.pivot(index="representation", columns="split", values="rmse").reindex(ORDER)

    print("\nTest RMSE (C) by representation and split:")
    print(piv.round(1).to_string())

    with open(args.outdir / "ablation_rmse.md", "w") as fh:
        cols = list(piv.columns)
        fh.write("| representation | " + " | ".join(cols) + " |\n")
        fh.write("|:---|" + "---:|" * len(cols) + "\n")
        for name, row in piv.round(1).iterrows():
            fh.write(f"| {name} | " + " | ".join(f"{v:.1f}" for v in row) + " |\n")

    # headline reduction
    rnd = piv["random"]
    print(f"\nOn the random split, sequence statistics cut RMSE from "
          f"{rnd['monomers_composition']:.1f} to {rnd['plus_sequence']:.1f} C "
          f"({100 * (1 - rnd['plus_sequence'] / rnd['monomers_composition']):.0f}% lower); "
          f"the naive composition-weighted encoding ({rnd['composition_weighted']:.1f}) "
          f"sits at the sequence-blind floor ({floor:.1f}).")

    if args.no_figures:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- figure 1: RMSE bars by representation x split ---
    fig, ax = plt.subplots(figsize=(8, 4.6))
    splits = list(piv.columns)
    x = np.arange(len(ORDER))
    w = 0.8 / len(splits)
    cmap = plt.get_cmap("tab10")
    for k, s in enumerate(splits):
        ax.bar(x + k * w - 0.4 + w / 2, piv[s].values, w, label=s,
               color=cmap(k), edgecolor="white")
    ax.axhline(floor, color="crimson", ls="--", lw=1.2,
               label=f"composition-blind floor ({floor:.0f} C)")
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in ORDER], fontsize=8)
    ax.set_ylabel("Test RMSE (C)")
    ax.set_title("Copolymer Tg: what each representation can and cannot see")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.outdir / "ablation_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure 2: blockiness sweep for one pair -- the naive encoding is flat ---
    from copolybench.oracle import tg_johnston
    from copolybench.sequences import (
        blockiness_index,
        dyad_fractions,
        mean_run_lengths,
    )

    y = df["tg"].to_numpy(float)
    sp = random_split(len(df), seed=args.seed)
    trained = {}
    for rep in ("composition_weighted", "plus_sequence"):
        Xr = build_representation(rep).transform(df)
        trained[rep] = (build_representation(rep),
                        GBMRegressor(n_estimators=500, backend="lightgbm")
                        .fit(Xr[sp.train], y[sp.train]))

    # most sequence-sensitive pair, held at equimolar composition
    pair = df.loc[df["delta"].abs().idxmax()]
    f0 = 0.5
    chis = np.linspace(-1, 1, 41)
    sweep = []
    for chi in chis:
        d = dyad_fractions(f0, chi)
        ra, rb = mean_run_lengths(f0, chi)
        sweep.append({
            "psmiles_a": pair["psmiles_a"], "psmiles_b": pair["psmiles_b"],
            "f": f0, "chi": chi, "F_AA": d.F_AA, "F_AB": d.F_AB, "F_BB": d.F_BB,
            "blockiness": blockiness_index(f0, chi),
            "run_a": min(ra, 1e6), "run_b": min(rb, 1e6),
            "tg_true": tg_johnston(pair["tg_a"], pair["tg_b"], d, pair["delta"]),
        })
    sdf = pd.DataFrame(sweep)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(chis, sdf["tg_true"], "k-", lw=2, label="true Tg (Johnston)")
    for rep, style in (("composition_weighted", "--"), ("plus_sequence", "-")):
        r, model = trained[rep]
        ax.plot(chis, model.predict(r.transform(sdf)), style, lw=1.8,
                label=rep.replace("_", " "))
    ax.set_xlabel("blockiness  chi   (alternating <-  0 random  -> blocky)")
    ax.set_ylabel("Tg (C)")
    ax.set_title(f"Same composition (f=0.5), varying sequence\n"
                 f"{pair['name_a']} / {pair['name_b']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.outdir / "blindness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigures written to {args.outdir}/")
    print("\nNOTE: controlled synthetic benchmark, not a Tg dataset. The sequence "
          "effect is\nplanted with a known model; this measures which representations "
          "recover it.")


if __name__ == "__main__":
    main()
