# C. Behind the Scenes of Diffusion Models: Itô’s Calculus and Girsanov’s Theorem {#app-ito}

(Score-based) Diffusion models are built on SDEs: a drift that pushes states and a Brownian term that jitters them. Unlike ODE paths, Brownian paths are nowhere differentiable, so the ordinary chain rule fails. In this section, we introduce two fundamental tools that make the math precise:

-   **Itô’s Formula** is the correct chain rule for stochastic trajectories. It tells us how a function ${\mathbf{h}}({\mathbf{x}}_t,t)$ evolves when ${\mathbf{x}}_t$ follows an SDE. It enables derivations of the Fokker–Planck equation, moment dynamics, the Itô product rule, and the identities used in score-based training.

-   **Girsanov’s Theorem** is a change-of-measure result on *path probabilities*. It quantifies how likelihoods change when the noise is fixed but the drift is altered. This links likelihood-weighted score matching to path-space KL divergence and, through a variational bound, to data likelihood.

With these tools, the standard diffusion model derivations (Fokker–Planck, reverse time SDE, training objectives, and likelihood relations) follow cleanly and without hand waving.

**After this chapter, you will be able to...**

-   understand why ordinary calculus is not enough for SDEs, and how Itô’s formula gives the correct chain rule for stochastic trajectories;

-   see how Itô’s formula works behind the scenes in important diffusion-model derivations, including the Fokker–Planck equation and linear SDEs;

-   understand Girsanov’s theorem as a way to compare the probabilities of stochastic paths when their drifts are different but their diffusion is the same;

-   distinguish matching distributions at each time from matching the full stochastic path, and see why likelihood-weighted score matching is related to path-space KL divergence and data likelihood.

**Reader route.** This appendix develops some of the mathematical tools that operate behind the scenes of diffusion models. Itô calculus makes stochastic differentiation precise, while Girsanov’s theorem explains how changing the drift changes the probability of entire stochastic paths. These tools underlie several standard diffusion-model derivations that are often used without being shown explicitly.

Continue with the sections below and the [references](../../references.md#references).

[C.1 Itô’s Formula: The Chain Rule for Random Processes](01-ito-formula.md#sec-ito-formula)

[C.2 Change-of-Variable For Measures: Girsanov’s Theorem in Diffusion Models](02-girsanov-theorem.md#sec-girsanov-theorem)

<span id="subsec-rigorous-proof-fpe"></span>

[Section C.1.4](01-ito-formula.md#subsec-rigorous-proof-fpe)

<span id="subsec-rigorous-proof-linear-sde"></span>

[Section C.1.5](01-ito-formula.md#subsec-rigorous-proof-linear-sde)

<span id="subsec-girsanov"></span>

[Section C.2.1](02-girsanov-theorem.md#subsec-girsanov)
