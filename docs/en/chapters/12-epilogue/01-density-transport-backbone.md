# 12.1 The Density-Transport Backbone {#sec-epilogue-backbone}

Throughout this book, we have followed two closely related lines of development. The first is *diffusion models themselves*, including variational, score-based, and flow-based formulations, which offer different mathematical descriptions of how probability laws evolve and how such evolution may be reversed for generation. The second is *diffusion-motivated fast generators*, most notably flow map models, which are built on the same transport viewpoint but aim to learn more direct and efficient generators. Despite their differences in form, all of these methods are rooted in a single organizing idea: generative modeling is fundamentally a problem of *transporting probability mass*. Starting from a simple reference distribution, such as Gaussian noise, we seek to construct a process that gradually connects it to a complex data distribution.

This viewpoint is useful because it shifts attention away from individual model families and toward a more fundamental question: *how does a probability law evolve when the underlying samples are moved by a prescribed dynamics?* The mathematical language for answering this question is the change-of-variable formula and its continuous-time extensions. In this sense, the change-of-variable principle is not merely a technical tool; it is the conceptual backbone underlying both diffusion models and the fast generators they inspire.

**A Common Three-Step Structure.** At a high level, the methods studied in this book all follow the same simple pattern:

1.  Define a forward process that gradually moves data toward a simple reference distribution.

2.  Understand how the probability law evolves under that process through the appropriate law-evolution equation.

3.  Learn a generative mechanism that reverses, approximates, or otherwise exploits this evolution in order to produce samples.

What varies across formulations is the choice of forward process, the mathematical form of the law-evolution equation, and the way the generative mechanism is parameterized and trained.

**The Change-of-Variable Hierarchy.** In [Appendix B](../../appendices/b-density-evolution/index.md#app-continuity), we developed a systematic progression of change-of-variable formulas, from deterministic bijections to stochastic differential equations. That progression is worth recalling here, because it reveals a single transport principle appearing in several forms.

<span id="fig-flow-3x3rows"></span>

![Vector field velocities deform the underlying grid](../../assets/chapter-12/vector-field.png)

**(a)** Vector field illustrations. The arrows represent velocities that move particles through space, deforming the underlying grid accordingly.

![A vector field transports a cloud of particles](../../assets/chapter-12/particle-cloud.png)

**(b)** Particle-cloud dynamics. A predefined vector field generates a flow that transports particles from their initial state.

![Density contours deform under the vector field](../../assets/chapter-12/density-evolution.png)

**(c)** Density evolution. As particles are advected by the vector field, the density contours deform accordingly, reflecting how the flow reshapes the underlying distribution.

**Figure 12.1:** **Illustrations of particle and density dynamics under a vector field.** Each column shows successive time snapshots from left to right.

Source: Adapted from [Lipman et al. (2024)](../../references.md#ref-lipman2024flow).

1.  **Single Bijection.** A smooth invertible map ${\boldsymbol{\Psi}}:{\mathbb{R}}^D\to{\mathbb{R}}^D$ transforms a density $p_0$ into $p_1$ via

    $$
    p_1({\mathbf{x}}_1)
        =
        p_0\bigl({\boldsymbol{\Psi}}^{-1}({\mathbf{x}}_1)\bigr)
        \left|\det \partial_{{\mathbf{x}}_1}{\boldsymbol{\Psi}}^{-1}({\mathbf{x}}_1)\right|.
    $$

    The Jacobian determinant records the local change of volume: when the map stretches space, the density decreases; when it compresses space, the density increases.

2.  **Composed Bijections.** Chaining many bijections, with ${\mathbf{x}}_k={\boldsymbol{\Psi}}_k({\mathbf{x}}_{k-1})$, yields

    $$
    \log p_L({\mathbf{x}}_L)
        =
        \log p_0({\mathbf{x}}_0)
        -
        \sum_{k=1}^L
        \log\left|\det \partial_{{\mathbf{x}}_{k-1}}{\boldsymbol{\Psi}}_k({\mathbf{x}}_{k-1})\right|.
    $$

    This is the basic principle behind normalizing flows. More importantly, it already illustrates the central message: once particle motion is specified, the evolution of the density follows accordingly.

3.  **Continuous-Time Deterministic Flow (ODE).** Taking the infinitesimal-step limit leads to the continuity equation,

    $$
    \frac{\partial p_t({\mathbf{x}})}{\partial t}
        +
        \nabla\cdot\bigl(p_t({\mathbf{x}})\,{\mathbf{v}}({\mathbf{x}},t)\bigr)
        =
        0,
    $$

    where particles evolve according to

    $$
    \frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = {\mathbf{v}}({\mathbf{x}}(t),t).
    $$

    This is the continuous-time expression of mass conservation. Probability mass is not created or destroyed; it simply flows through space under the velocity field ${\mathbf{v}}$.

4.  **Continuous-Time Stochastic Flow (SDE).** Adding Brownian noise yields the Fokker–Planck equation,

    $$
    \frac{\partial p_t({\mathbf{x}})}{\partial t}
        =
        -\nabla\cdot\bigl({\mathbf{v}}({\mathbf{x}},t)\,p_t({\mathbf{x}})\bigr)
        +
        \frac{1}{2}g^2(t)\,\Delta p_t({\mathbf{x}}),
    $$

    which corresponds to the stochastic dynamics

    $$
    \mathop{}\!\mathrm{d}{\mathbf{x}}(t) = {\mathbf{v}}({\mathbf{x}}(t),t)\mathop{}\!\mathrm{d} t + g(t)\mathop{}\!\mathrm{d}{\mathbf{w}}(t).
    $$

    The first term describes directed transport by the drift, while the second captures the random spreading induced by noise.

These are not unrelated formulas. The multi-layer change-of-variable rule is the discrete-time version of transport under repeated deterministic transformations. Its infinitesimal deterministic limit gives the continuity equation. Adding stochastic perturbation leads to the Fokker–Planck equation. Each level extends the previous one, while preserving the same underlying principle: probability mass is conserved, and the probability law adjusts to reflect how the underlying dynamics moves samples through space.

From this perspective, the main formulations in this book are easier to place in context. Variational methods describe generation through denoising transitions on a time grid. Score-based methods work with stochastic dynamics and learn the score field governing the reverse-time SDE. Flow-based methods work with deterministic transport and learn a velocity field whose induced marginal distributions follow a prescribed family of intermediate laws. Fast generators, in turn, build on this same backbone, but seek more direct ways to exploit the learned transport for efficient sampling.

The central takeaway is therefore simple. Diffusion modeling is not best understood as a single algorithm tied to one architecture or one noise schedule. It is a *design principle*: specify a forward transport process, understand the induced evolution of probability laws, and then construct a generator on top of that structure. Once this viewpoint is in place, many seemingly different methods become natural variations of the same idea.

A natural question then arises: does this principle extend beyond continuous state spaces? In the next section, we show that it does.
