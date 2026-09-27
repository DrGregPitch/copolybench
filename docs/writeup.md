# What most ML gets wrong about copolymers: sequence is not composition

*A controlled benchmark for copolymer representation learning.*

## The quiet assumption

Machine learning on polymers has a foundation problem that almost nobody states out loud. SMILES — and every fingerprint, descriptor, and graph built on it — was designed for discrete molecules: a fixed set of atoms in a fixed arrangement. A copolymer is not that. It is a *statistical ensemble* of chains, and two statistics describe it: **composition** (the mole fractions of each comonomer) and **sequence** (how those comonomers are arranged along the backbone — random, alternating, or blocky).

Open almost any copolymer property-prediction paper and you find the same move: a copolymer is encoded as a composition-weighted average of its comonomers' feature vectors, `f·v_A + (1−f)·v_B`. It is convenient, it is everywhere, and it is *blind to sequence by construction*. A block copolymer and a random copolymer of identical composition produce the identical vector, and therefore the identical prediction. The model cannot tell them apart even in principle.

Does that blindness matter? The honest answer is "it depends, and rarely is anyone measuring how much." This project builds the instrument to measure it.

## Why a controlled benchmark

To ask "which representation recovers the sequence effect?" you need data where the sequence effect is *known*. Real experimental copolymer datasets with both composition and sequence characterization are scarce, noisy, and confounded — a bad place to isolate a representational question. So the labels here are generated from a known model, and the whole design is oriented around one property: **we know exactly how much of each label is composition and how much is sequence.**

The generator draws comonomer pairs from the 60 homopolymers bundled in [`polytools`](https://github.com/DrGregPitch/polytools), each with a literature glass-transition temperature (Tg). For each copolymer it computes Tg from the **Johnston dyad equation**, a real sequence-aware model:

```
1/Tg = F_AA/Tg_AA + F_AB/Tg_AB + F_BB/Tg_BB
```

summed over the nearest-neighbour *dyad* fractions (`F_AA`, `F_AB`, `F_BB`), with a dyad-specific Tg for the AB junction. The elegant part is its exact relationship to the composition-only **Fox equation**. Substituting the composition constraints `F_AA = f − F_AB/2` and `F_BB = (1−f) − F_AB/2` gives, with no approximation:

```
1/Tg = [ f/Tg_A + (1−f)/Tg_B ]  +  F_AB · ( 1/Tg_AB − 1/(2·Tg_A) − 1/(2·Tg_B) )
        └────────── Fox ──────────┘    └──────── the entire sequence effect ────────┘
```

Every bit of sequence dependence lives in the `F_AB` coefficient. When the AB-junction Tg equals the harmonic mean of the two homopolymer Tgs, that coefficient is *exactly zero* and Johnston collapses onto Fox — sequence stops mattering. A single per-pair parameter, `delta`, sets how far the junction departs from neutral, and it is tied to the comonomers' polarity mismatch: chemically similar comonomers mix near-ideally (`delta ≈ 0`), dissimilar ones have non-ideal junctions (larger `|delta|`). Because `delta` is a function of monomer structure, a model can in principle *learn* it from the comonomers and generalise to pairs it has never seen — which is what makes the out-of-distribution test meaningful rather than hopeless.

The result is a dataset of 5,400 copolymers across 300 comonomer pairs in which the sequence contribution has a standard deviation of about 27 °C — the scale of what sequence contributes, and the yardstick every representation gets measured against. (It is a yardstick rather than a strict floor: because each pair's sequence *sensitivity* is learnable from monomer structure, a composition-only model that knows the monomers can anticipate part of the average effect — what it can never do is resolve which sequence a given sample has.)

## The experiment

Four representations, one gradient-boosting model, two splits. The representations climb from blind to sighted: composition-weighted average; both monomers concatenated with composition; sequence statistics with no monomer identity; and the full encoding — both monomers, composition, and first-order sequence statistics (the AB dyad fraction, a blockiness index, and mean run lengths). The splits ask different questions: a **random** split (copolymers of a pair appear in train and test) measures whether a representation can fit the sequence effect at all; a **leave-pair-out** split (whole comonomer pairs held out) measures whether it *generalises* to new chemistry.

| representation | random | leave-pair-out |
|:---|---:|---:|
| composition-weighted (naive) | 25.3 | 44.8 |
| both monomers + composition | 15.8 | 39.8 |
| sequence stats only | 84.5 | 82.2 |
| **both monomers + composition + sequence** | **9.4** | **38.0** |

*Test RMSE, °C.*

The random column is the headline. The naive composition-weighted encoding leaves 25.3 °C of error — essentially the full 27 °C scale of the planted sequence effect, meaning it resolves almost none of what sequence contributes. That is not about model capacity or data volume; it is about information the representation never contained. Adding first-order sequence statistics drops RMSE to 9.4 °C, a 60 % reduction, down toward the label-noise floor.

The single most convincing picture is not the table but a slice through it. Fix a sequence-sensitive comonomer pair at 50/50 composition and sweep the blockiness axis. The true Tg swings roughly 130 °C from alternating to blocky. The composition-weighted model draws a **flat horizontal line** across the entire sweep — one answer for a property that moves by 130 °C — while the sequence-aware model tracks the real curve. That flat line is the thesis.

## The nuances that matter more than the headline

Two secondary findings are, to a working scientist, the interesting part.

**Sequence statistics alone are not enough.** Strip out monomer identity and keep only composition and sequence numbers, and error explodes to 84 °C. Sequence tells you how the pieces are arranged; it says nothing about what the pieces *are*. You need composition, sequence, and monomer identity together. This rules out the lazy conclusion "just add blockiness as a feature" — blockiness is only meaningful alongside the chemistry it modulates.

**The advantage narrows out of distribution.** On leave-pair-out, every representation struggles (38–45 °C), because generalising the sequence *sensitivity* — which depends on the specific comonomer chemistry — to unseen pairs is genuinely hard. Sequence features still help, but the gap over the naive baseline shrinks from 16 °C to 7 °C. A composition-weighted average turns out to be a decent inductive bias precisely when you know least about the pair. That is not a comfortable result to report, and it is exactly the kind a random-split-only study would never surface.

## What this contributes, and where it is thin

This is a measurement instrument, not a Tg oracle, and it is labelled that way throughout. It plants a first-order (dyad) sequence effect and asks which representations recover it; it does not claim agreement with experiment, and it does not model triad or longer-range order, microphase separation, or the two-Tg behaviour real block copolymers can show. The Fox/Johnston framework is a real but simplified picture. The equal-unit-mass simplification in the mixing weights is a known corner cut.

But the core claim survives all of those caveats, because it is structural rather than empirical: **a representation that is a function of composition alone cannot express a property that depends on sequence, and for a first-order sequence effect, the dyad fraction is the minimal statistic that repairs it.** The open question this contributes to is a live one — how to encode a stochastic macromolecular ensemble for property prediction — and the contribution is a clean, reproducible way to ask, of any proposed copolymer representation, the question the field usually skips: *what can it not see?*

The code, the oracle, and every figure reproduce in about a minute from a clean clone. Point the same record structure at real copolymer data and the question sharpens from "which representation recovers a planted effect" to "which representation recovers the real one." That is the next experiment.

---

*Built on [`polytools`](https://github.com/DrGregPitch/polytools) — the shared, honest evaluation harness (structured splits, calibrated metrics, polymer-aware featurizers). Code: [`copolybench`](https://github.com/DrGregPitch/copolybench).*
