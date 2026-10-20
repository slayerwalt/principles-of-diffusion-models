# 6. A Unified and Systematic Lens on Diffusion Models {#ch-unified-perspective}

> *Mathematics is the art of giving the same name to different things.*
>
> — Henri Poincaré

This chapter presents a systematic viewpoint that connects the variational, score-based, and flow-based perspectives within a coherent picture. Although motivated by different intuitions, these approaches converge on the same core mechanism underlying modern diffusion models: define a forward corruption process that traces a path of marginal distributions, then learn the local quantities needed to construct generative dynamics that follow this path in reverse. Building on [Chapter 2](../02-variational-perspective/index.md#ch-variational) and [Chapter 3](../03-score-based-perspective/index.md#ch-score-based) and [Chapter 4](../04-score-sde-framework/index.md#ch-score-sde) and [Chapter 5](../05-flow-based-perspective/index.md#ch-flow-based), we observe a common recipe: define a forward corruption process that traces a path of marginals, then learn a time varying vector field that transports a simple prior to the data distribution along this path.

A key ingredient across all perspectives is the conditioning trick introduced in [Section 6.1](01-conditioning-tricks.md#sec-conditional-trick), which transforms an intractable marginal objective into a tractable conditional one, leading to stable and efficient training.

In [Section 6.2](02-training-objectives.md#sec-elucidating) we analyze the training objective in a systematic way, identifying its essential components and clarifying how loss functions are formulated in the variational, score-based, and flow-based viewpoints.

[Section 6.3](03-equivalent-parameterizations.md#sec-equivalent-parametrizations) shows that any affine forward noise injection of the form ${\mathbf{x}}_t = \alpha_t {\mathbf{x}}_0 + \sigma_t {\boldsymbol{\epsilon}}$ can be equivalently transformed into the standard linear schedule ${\mathbf{x}}_t = (1 - t){\mathbf{x}}_0 + t{\boldsymbol{\epsilon}}$. Moreover, common parameterizations such as noise prediction, clean data prediction, score prediction, and velocity prediction are interchangeable at the level of gradients. Thus, the choices of noise schedulers and parameterizations both adhere to the same modeling principle.

Finally, [Section 6.4](04-fokker-planck-connections.md#sec-comparison-connection) brings the discussion together and identifies the governing rule: the Fokker–Planck equation. Whether viewed as a variational scheme (discrete time denoising), a score-based method (SDE formulation), or a flow-based method (ODE formulation), each constructs a generator whose marginals follow the same density evolution. The Fokker–Planck equation thus serves as the universal constraint respected by all three viewpoints, with differences arising only in parameterization and training objectives.

**After this chapter, you will be able to...**

-   recognize the common conditioning principle behind variational, score-based, and flow-based training: an intractable marginal target can be learned through tractable conditional targets;

-   understand the four common prediction types—clean data, noise, score, and velocity—and convert between their oracle predictions when the forward schedule is nondegenerate;

-   disentangle the main design choices in diffusion training: the forward noise schedule, prediction target, time-sampling distribution, and time weighting, and understand which changes are mathematical reparameterizations and which can still affect practical optimization;

-   distinguish different notions of “equivalence”: representing the same oracle information, having equivalent population objectives up to weighting or additive constants, tracing the same marginal density path, and behaving identically as a practical training or sampling algorithm;

-   understand how different affine probability paths are related through time reparameterization and spatial rescaling, while keeping track of what these transformations do and do not preserve;

-   understand how the continuity and Fokker–Planck equations provide the density-level principle connecting variational, deterministic ODE, and stochastic SDE descriptions of diffusion models.

**Reader route.** This chapter is the conceptual keystone of the book. For a first pass, focus on the conditioning principle, the relationships among the four prediction types, and the continuity/Fokker–Planck view that connects the three perspectives. The central lesson is not to memorize every conversion formula, but to ask what notion of equivalence is being used and what that equivalence actually preserves. The detailed analysis of weighting, schedules, and parameterization choices can then be revisited to understand why mathematically equivalent formulations may still behave differently in practice.

Continue with the sections below and the [chapter references](references.md#chapter-06-references).

[6.1 Conditional Tricks: The Secret Sauce of Diffusion Models](01-conditioning-tricks.md#sec-conditional-trick)

[6.2 A Roadmap for Elucidating Training Losses in Diffusion Models](02-training-objectives.md#sec-elucidating)

[6.3 Equivalence in Diffusion Models](03-equivalent-parameterizations.md#sec-equivalent-parametrizations)

[6.4 Beneath It All: The Fokker–Planck Equation](04-fokker-planck-connections.md#sec-comparison-connection)

[6.5 Closing Remarks](05-closing-remarks.md#subsec-takeaway-dm-gaussianfm)

<span id="equi-para"></span>

[Proposition 6.3.1](03-equivalent-parameterizations.md#equi-para)
