# 11. Learning Fast Generators from Scratch {#ch-fast-scratch}

> *Truth is ever to be found in simplicity, and not in the multiplicity and confusion of things.*
>
> — Isaac Newton

In [Chapter 10](../10-distillation/index.md#ch-distillation), we saw that slow iterative samplers in diffusion models can be compressed into few-step generators through distillation. From an engineering perspective, two-stage pipelines are practical because they divide a complex generative training task into clear, independent objectives. The first stage learns the data distribution, while the second accelerates sampling or enhances quality. This separation allows each stage to be optimized independently, making the overall system easier to manage, more stable, and more reliable.

In this chapter, however, the focus shifts to a central question driving the progress of deep generative modeling:

<span id="question-11-0-1"></span>

**Question 11.0.1**

Can we design a standalone generative principle that trains in a stable and efficient way, fast sampling, and allows users to easily guide or control what is produced?

In this chapter we pursue this direction and discuss an alternative approach: training few-step diffusion-based generators *without* relying on a pre-trained model. Our focus is the *flow-map model*. Here, a *flow map* is the classical solution map of an ODE, a notion rooted in differential equations and dynamical systems and closely connected, in continuum and fluid mechanics, to following the motion of material particles. Specifically, ${\boldsymbol{\Psi}}_{s\to t}$ takes a state at time $s$ to the state reached at time $t$ under the prescribed dynamics. A flow-map model learns to approximate this map directly, allowing a long trajectory to be traversed in one or a few network evaluations [^1].

This formulation provides a principled way to transport probability mass from the prior distribution $p_{\mathrm{prior}}$ to the data distribution $p_{\mathrm{data}}$, while preserving the marginal distributions $p_t$ specified by the forward diffusion process at each intermediate time.

**After this chapter, you will be able to...**

-   distinguish an instantaneous velocity field from its ODE solution map (flow map);

-   understand the semigroup property of exact ODE flows and how it motivates consistency-based learning objectives;

-   understand Consistency Model, Consistency Trajectory Model, and Mean Flow as representative ways to learn special or general flow maps;

-   distinguish oracle flow-map identities from their practical approximations, and understand how suitable parametrizations enable few-step generation while retaining access to local velocity information.

**Reader route.** The central object in this chapter is the ODE flow map. The individual methods should be viewed as different parametrizations and training strategies for learning this same underlying object.

[^1]: See [Equation (4.2.2)](../04-score-sde-framework/02-matching-marginals.md#eq-flow-map-def) and the surrounding discussion for the mathematical and historical background of flow maps.

Continue with the sections below and the [chapter references](references.md#chapter-11-references).

[11.1 Prologue](01-prologue.md#sec-fast-scratch-prologue)

[11.2 Special Flow Map: Consistency Model in Discrete Time](02-discrete-consistency-models.md#sec-discrete-cm)

[11.3 Special Flow Map: Consistency Model in Continuous Time](03-continuous-consistency-models.md#sec-conti-cm)

[11.4 General Flow Map: Consistency Trajectory Model](04-consistency-trajectory-models.md#sec-ctm)

[11.5 General Flow Map: Mean Flow](05-mean-flow.md#sec-mf)

[11.6 Closing Remarks](06-closing-remarks.md#sec-ch11-cr)
