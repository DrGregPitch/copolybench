# Sequence is not composition: measuring what copolymer representations miss

*A controlled benchmark for copolymer representation learning.*

## The assumption

Machine learning on polymers rests on an assumption that is rarely stated. SMILES — and every fingerprint, descriptor, and graph built on it — was designed for discrete molecules: a fixed set of atoms in a fixed arrangement. A copolymer is not that. It is a *statistical ensemble* of chains, and two statistics describe it: **composition** (the mole fractions of each comonomer) and **sequence** (how those comonomers are arranged along the backbone — random, alternating, or blocky).

Most copolymer property-prediction papers make the same move: a copolymer is encoded as a composition-weighted average of its comonomers' feature vectors, `f·v_A + (1−f)·v_B`. It is convenient, it is common, and it is sequence-blind by construction. A block copolymer and a random copolymer of identical composition produce the identical vector, and therefore the identical prediction. The model cannot tell them apart even in principle.

Whether that matters depends on the system, and it is rarely measured. This project builds an instrument to measure it.

## Why a controlled benchmark

To ask "which representation recovers the sequence effect?" you need data where the sequence effect is *known*. Real experimental copolymer datasets with both composition and sequence characterization are scarce, noisy, and confounded — a bad place to isolate a representational question. So the labels here are generated from a known model, and the whole design is oriented around one property: how much of each label is composition and how much is sequence is known exactly.

The generator draws comonomer pairs from the 60 homopolymers bundled in [`polytools`](https://github.com/DrGregPitch/polytools), each with a literature glass-transition temperature (Tg). For each copolymer it computes Tg from the **Johnston dyad equation**, a real sequence-aware model:

```
1/Tg = F_AA/Tg_AA + F_AB/Tg_AB + F_BB/Tg_BB
```

summed over the nearest-neighbour *dyad* fractions (`F_AA`, `F_AB`, `F_BB`), with a dyad-specific Tg for the AB junction. It has an exact relationship to the composition-only Fox equation. Substituting the composition constraints `F_AA = f − F_AB/2` and `F_BB = (1−f) − F_AB/2` gives, with no approximation:

```
1/Tg = [ f/Tg_A + (1−f)/Tg_B ]  +  F_AB · ( 1/Tg_AB − 1/(2·Tg_A) − 1/(2·Tg_B) )
        └────────── Fox ──────────┘    └──────── the entire sequence effect ────────┘
```

Every bit of sequence dependence lives in the `F_AB` coefficient. When the AB-junction Tg equals the harmonic mean of the two homopolymer Tgs, that coefficient is *exactly zero* and Johnston collapses onto Fox — sequence stops mattering. A single per-pair parameter, `delta`, sets how far the junction departs from neutral, and it is tied to the comonomers' polarity mismatch: chemically similar comonomers mix near-ideally (`delta ≈ 0`), dissimilar ones have non-ideal junctions (larger `|delta|`). Because `delta` is a function of monomer structure, a model can in principle *learn* it from the comonomers and generalise to pairs it has never seen — which is what makes the out-of-distribution test meaningful.

The result is a dataset of 5,400 copolymers across 300 comonomer pairs in which the sequence contribution has a standard deviation of about 27 °C — the scale of what sequence contributes, and the yardstick every representation gets measured against. (It is a yardstick rather than a strict floor: because each pair's sequence *sensitivity* is learnable from monomer structure, a composition-only model that knows the monomers can anticipate part of the average effect — what it can never do is resolve which sequence a given sample has.)

## The experiment

Four representations, one gradient-boosting model, two splits. The representations range from sequence-blind to sequence-aware: composition-weighted average; both monomers concatenated with composition; sequence statistics with no monomer identity; and the full encoding — both monomers, composition, and first-order sequence statistics (the AB dyad fraction, a blockiness index, and mean run lengths). The splits ask different questions: a **random** split (copolymers of a pair appear in train and test) measures whether a representation can fit the sequence effect at all; a **leave-pair-out** split (whole comonomer pairs held out) measures whether it *generalises* to new chemistry.

| representation | random | leave-pair-out |
|:---|---:|---:|
| composition-weighted (naive) | 25.3 | 44.8 |
| both monomers + composition | 15.8 | 39.8 |
| sequence stats only | 84.5 | 82.2 |
| **both monomers + composition + sequence** | **9.4** | **38.0** |

*Test RMSE, °C.*

Taking the random column first: the naive composition-weighted encoding leaves 25.3 °C of error, essentially the full 27 °C scale of the planted sequence effect, so it resolves almost none of what sequence contributes. This is not a matter of model capacity or data volume but of information the representation never contained. Adding first-order sequence statistics drops RMSE to 9.4 °C. Measured against the matched baseline that already distinguishes the two monomers (15.8 °C), that is a 40 % reduction attributable to sequence; the larger fall from the naive 25.3 °C also folds in simply keeping the comonomers distinct, which is not a sequence effect. On the held-out `leave_pair_out` split the sequence gain narrows to ~5 %: the per-pair sensitivity has to be learned from structure, and generalising it to unseen pairs is the hard part.

A slice through the table shows the same thing more directly. Fix a sequence-sensitive comonomer pair at 50/50 composition and sweep the blockiness axis. The true Tg swings roughly 130 °C from alternating to blocky. The composition-weighted model returns a flat horizontal line across the entire sweep — one answer for a property that moves by 130 °C — while the sequence-aware model tracks the real curve.

## Two secondary findings

**Sequence statistics alone are not enough.** Strip out monomer identity and keep only composition and sequence numbers, and error rises to 84 °C. Sequence describes how the pieces are arranged; it says nothing about what the pieces are. Composition, sequence and monomer identity are needed together. This rules out "just add blockiness as a feature" — blockiness is only meaningful alongside the chemistry it modulates.

**The advantage narrows out of distribution.** On leave-pair-out, every representation struggles (38–45 °C), because generalising the sequence *sensitivity* — which depends on the specific comonomer chemistry — to unseen pairs is genuinely hard. Sequence features still help, but the gap over the naive baseline shrinks from 16 °C to 7 °C. A composition-weighted average is a reasonable inductive bias precisely when least is known about the pair — which a random-split-only study would not surface.

## Scope and limitations

This is a measurement instrument, not a Tg oracle, and it is labelled that way throughout. It plants a first-order (dyad) sequence effect and asks which representations recover it; it does not claim agreement with experiment, and it does not model triad or longer-range order, microphase separation, or the two-Tg behaviour real block copolymers can show. The Fox/Johnston framework is a real but simplified picture. The equal-unit-mass simplification in the mixing weights is a known corner cut.

The core claim is structural rather than empirical, so it survives those caveats: a representation that is a function of composition alone cannot express a property that depends on sequence, and for a first-order sequence effect the dyad fraction is the minimal statistic that repairs it. The open question it contributes to — how to encode a stochastic macromolecular ensemble for property prediction — is a live one, and what this adds is a reproducible way to ask of any proposed copolymer representation what it cannot see.

The code, the oracle, and every figure reproduce in about a minute from a clean clone. Point the same record structure at real copolymer data and the question sharpens from "which representation recovers a planted effect" to "which representation recovers the real one." That is the next experiment.

---

*Built on [`polytools`](https://github.com/DrGregPitch/polytools) — the shared evaluation harness (structured splits, calibrated metrics, polymer-aware featurizers). Code: [`copolybench`](https://github.com/DrGregPitch/copolybench).*
