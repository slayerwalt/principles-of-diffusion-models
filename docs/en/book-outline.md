---
title: Book contents
description: Chapter structure and conversion status for the English web edition.
---

# Book contents

This outline follows the order of the book. Chapters 1–12 are available in English. The front matter, the unnumbered Part II opening, and the appendices are awaiting conversion.

## Front matter

- Preface — awaiting conversion
- Notation — awaiting conversion

## Part I · Introduction to Deep Generative Modeling

1. [Deep Generative Modeling](chapters/01-deep-generative-modeling/index.md) — available

    - [1.1 What is Deep Generative Modeling?](chapters/01-deep-generative-modeling/01-what-is-deep-generative-modeling.md)
    - [1.2 Prominent Deep Generative Models](chapters/01-deep-generative-modeling/02-examples.md)
    - [1.3 Taxonomy of Modeling](chapters/01-deep-generative-modeling/03-taxonomy.md)
    - [1.4 Closing Remarks](chapters/01-deep-generative-modeling/04-summary.md)
    - [Chapter references](chapters/01-deep-generative-modeling/references.md)

## Part II · Foundations and Perspectives of Diffusion Models

The unnumbered opening section, [A First Operational View of Diffusion Model Training and Sampling](foundations/operational-view.md), is awaiting conversion.

2. [Variational Perspective: From VAEs to DDPMs](chapters/02-variational-perspective/index.md) — available

    - [2.1 Variational Autoencoder](chapters/02-variational-perspective/01-variational-autoencoder.md)
    - [2.2 Variational Perspective: DDPM](chapters/02-variational-perspective/02-ddpm.md)
    - [2.3 Closing Remarks](chapters/02-variational-perspective/03-closing-remarks.md)
    - [Chapter references](chapters/02-variational-perspective/references.md)

3. [Score-Based Perspective: From EBMs to NCSN](chapters/03-score-based-perspective/index.md) — available

    - [3.1 Energy-Based Models](chapters/03-score-based-perspective/01-energy-based-models.md)
    - [3.2 From Energy-Based to Score-Based Generative Models](chapters/03-score-based-perspective/02-score-based-models.md)
    - [3.3 Denoising Score Matching](chapters/03-score-based-perspective/03-denoising-score-matching.md)
    - [3.4 Multi-Noise Levels of Denoising Score Matching (NCSN)](chapters/03-score-based-perspective/04-noise-conditional-score-networks.md)
    - [3.5 Summary: A Comparative View of NCSN and DDPM](chapters/03-score-based-perspective/05-comparison.md)
    - [3.6 Closing Remarks](chapters/03-score-based-perspective/06-closing-remarks.md)
    - [Chapter references](chapters/03-score-based-perspective/references.md)

4. [Diffusion Models Today: Score SDE Framework](chapters/04-score-sde-framework/index.md) — available

    - [4.1 Score SDE: Its Principles](chapters/04-score-sde-framework/01-score-sde-principles.md)
    - [4.2 Matching Marginals in Forward/Reverse-Time SDEs and PF-ODE](chapters/04-score-sde-framework/02-matching-marginals.md)
    - [4.3 Score SDE: Its Training and Sampling](chapters/04-score-sde-framework/03-training-and-sampling.md)
    - [4.4 Instantiations of SDEs](chapters/04-score-sde-framework/04-sde-instantiations.md)
    - [4.5 A Direct View Through Accumulated Perturbation Kernels](chapters/04-score-sde-framework/05-accumulated-perturbation-kernels.md)
    - [4.6 (Optional) Fokker–Planck Equation and Reverse-Time SDEs via Marginalization and Bayes’ Rule](chapters/04-score-sde-framework/06-fokker-planck-and-reverse-sde.md)
    - [4.7 Closing Remarks](chapters/04-score-sde-framework/07-closing-remarks.md)
    - [Chapter references](chapters/04-score-sde-framework/references.md)

5. [Flow-Based Perspective: From NFs to Flow Matching](chapters/05-flow-based-perspective/index.md) — available

    - [5.1 Flow-Based Models: Normalizing Flows and Neural ODEs](chapters/05-flow-based-perspective/01-normalizing-flows-and-neural-odes.md)
    - [5.2 Flow Matching Framework](chapters/05-flow-based-perspective/02-flow-matching-framework.md)
    - [5.3 Constructing Probability Paths and Velocities Between Distributions](chapters/05-flow-based-perspective/03-probability-paths-and-velocities.md)
    - [5.4 (Optional) Properties of the Canonical Affine Flow](chapters/05-flow-based-perspective/04-canonical-affine-flow.md)
    - [5.5 Closing Remarks](chapters/05-flow-based-perspective/05-closing-remarks.md)
    - [Chapter references](chapters/05-flow-based-perspective/references.md)

6. [A Unified and Systematic Lens on Diffusion Models](chapters/06-unified-perspective/index.md) — available

    - [6.1 Conditional Tricks: The Secret Sauce of Diffusion Models](chapters/06-unified-perspective/01-conditioning-tricks.md)
    - [6.2 A Roadmap for Elucidating Training Losses in Diffusion Models](chapters/06-unified-perspective/02-training-objectives.md)
    - [6.3 Equivalence in Diffusion Models](chapters/06-unified-perspective/03-equivalent-parameterizations.md)
    - [6.4 Beneath It All: The Fokker–Planck Equation](chapters/06-unified-perspective/04-fokker-planck-connections.md)
    - [6.5 Closing Remarks](chapters/06-unified-perspective/05-closing-remarks.md)
    - [Chapter references](chapters/06-unified-perspective/references.md)

7. [(Optional) Diffusion Models and Optimal Transport](chapters/07-optimal-transport/index.md) — available

    - [7.1 Prologue of Distribution-to-Distribution Translation](chapters/07-optimal-transport/01-distribution-translation.md)
    - [7.2 Taxonomy of the Problem Setups](chapters/07-optimal-transport/02-transport-problem-setups.md)
    - [7.3 Relationship of Variant Optimal Transport Formulations](chapters/07-optimal-transport/03-relationships-between-formulations.md)
    - [7.4 Is Diffusion Model's SDE Optimal Solution to SB Problem?](chapters/07-optimal-transport/04-diffusion-sdes-and-schrodinger-bridges.md)
    - [7.5 Is Diffusion Model's ODE an Optimal Map to OT Problem?](chapters/07-optimal-transport/05-probability-flow-odes-and-optimal-transport.md)
    - [7.6 Closing Remarks](chapters/07-optimal-transport/06-closing-remarks.md)
    - [Chapter references](chapters/07-optimal-transport/references.md)

## Part III · Sampling of Diffusion Models

8. [Guidance and Controllable Generation](chapters/08-guidance/index.md) — available

    - [8.1 Prologue](chapters/08-guidance/01-prologue.md)
    - [8.2 Classifier Guidance](chapters/08-guidance/02-classifier-guidance.md)
    - [8.3 Classifier-Free Guidance](chapters/08-guidance/03-classifier-free-guidance.md)
    - [8.4 Training-Free Guidance](chapters/08-guidance/04-training-free-guidance.md)
    - [8.5 From Reinforcement Learning to Direct Preference Optimization for Model Alignment](chapters/08-guidance/05-preference-alignment.md)
    - [8.6 Closing Remarks](chapters/08-guidance/06-closing-remarks.md)
    - [Chapter references](chapters/08-guidance/references.md)

9. [Numerical Solvers for Fast Diffusion Sampling](chapters/09-numerical-solvers/index.md) — available

    - [9.1 Prologue](chapters/09-numerical-solvers/01-prologue.md)
    - [9.2 DDIM](chapters/09-numerical-solvers/02-ddim.md)
    - [9.3 DEIS](chapters/09-numerical-solvers/03-deis.md)
    - [9.4 DPM-Solver](chapters/09-numerical-solvers/04-dpm-solver.md)
    - [9.5 DPM-Solver++](chapters/09-numerical-solvers/05-dpm-solver-plus-plus.md)
    - [9.6 (Optional) ParaDiGMs](chapters/09-numerical-solvers/06-paradigms.md)
    - [9.7 Closing Remarks](chapters/09-numerical-solvers/07-closing-remarks.md)
    - [Chapter references](chapters/09-numerical-solvers/references.md)

## Part IV · Toward Learning Fast Diffusion-Based Generators

10. [Distillation-Based Methods for Fast Sampling](chapters/10-distillation/index.md) — available

    - [10.1 Prologue](chapters/10-distillation/01-prologue.md)
    - [10.2 Distribution-Based Distillation](chapters/10-distillation/02-distribution-based-distillation.md)
    - [10.3 Progressive Distillation](chapters/10-distillation/03-progressive-distillation.md)
    - [10.4 Closing Remarks](chapters/10-distillation/04-closing-remarks.md)
    - [Chapter references](chapters/10-distillation/references.md)

11. [Learning Fast Generators from Scratch](chapters/11-fast-generators/index.md) — available

    - [11.1 Prologue](chapters/11-fast-generators/01-prologue.md)
    - [11.2 Special Flow Map: Consistency Model in Discrete Time](chapters/11-fast-generators/02-discrete-consistency-models.md)
    - [11.3 Special Flow Map: Consistency Model in Continuous Time](chapters/11-fast-generators/03-continuous-consistency-models.md)
    - [11.4 General Flow Map: Consistency Trajectory Model](chapters/11-fast-generators/04-consistency-trajectory-models.md)
    - [11.5 General Flow Map: Mean Flow](chapters/11-fast-generators/05-mean-flow.md)
    - [11.6 Closing Remarks](chapters/11-fast-generators/06-closing-remarks.md)
    - [Chapter references](chapters/11-fast-generators/references.md)

## Epilogue and Outlook

12. [A Unifying Principle and the Road Ahead](chapters/12-epilogue/index.md) — available

    - [12.1 The Density-Transport Backbone](chapters/12-epilogue/01-density-transport-backbone.md)
    - [12.2 Beyond Continuous States: Probability Evolution on Discrete Spaces](chapters/12-epilogue/02-discrete-spaces.md)
    - [12.3 Closing Remarks of the Book](chapters/12-epilogue/03-closing-remarks.md)
    - [Chapter references](chapters/12-epilogue/references.md)

## Appendices

- **A.** [Crash Course on Differential Equations](appendices/a-differential-equations/index.md) — not yet available
- **B.** [Density Evolution: From Change of Variable to Fokker–Planck](appendices/b-density-evolution/index.md) — not yet available
- **C.** [Behind the Scenes of Diffusion Models: Itô’s Calculus and Girsanov's Theorem](appendices/c-ito-calculus/index.md) — not yet available
- **D.** [Supplementary Materials and Proofs](appendices/d-supplementary-proofs/index.md) — not yet available

The complete bibliography is awaiting conversion. References cited in each available chapter are included with that chapter.
