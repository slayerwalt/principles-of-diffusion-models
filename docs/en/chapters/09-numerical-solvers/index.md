# 9. Numerical Solvers for Fast Diffusion Sampling {#ch-numerical-solvers}

The generation process of a diffusion model, which maps noise to data samples, can be formulated through a reverse-time SDE or its associated PF-ODE. This procedure is inherently slow, since it relies on numerical solvers that approximate solution trajectories with many small integration steps (see [Appendix A](../../appendices/a-differential-equations/index.md#app-de) for a brief introduction). Accelerating inference has therefore become a central research objective. Broadly, existing approaches fall into two categories:

-   **Training-Free Approaches:** The focus of this chapter. These methods develop advanced numerical solvers to improve the efficiency of diffusion sampling without additional training.

-   **Training-Based Approaches:** Covered in [Chapter 10](../10-distillation/index.md#ch-distillation) and [Chapter 11](../11-fast-generators/index.md#ch-fast-scratch). These techniques either distill a pre-trained diffusion model into a fast generator, or directly learn the ODE flow map (solution) so that only a few sampling steps are required.

SDE-based samplers (e.g., Euler–Maruyama) may yield more diverse samples due to stochasticity but typically require more steps ([Xu et al., 2023](../../references.md#ref-xu2023restart)). Here we focus on ODE-based generation, whose principles extend naturally to the SDE setting.

**After this chapter, you will be able to...**

-   view diffusion sampling as a numerical integration problem for the PF-ODE;

-   distinguish model approximation error from numerical discretization error;

-   understand the main numerical ideas behind DDIM, DEIS, DPM-Solver, and DPM-Solver`++`;

-   connect modern diffusion samplers to classical numerical methods, including Euler, Runge-Kutta (RK), Adams–Bashforth (AB), and exponential integrators.

**Reader route.** Readers unfamiliar with differential equations or numerical solvers may first refer to [Appendix A](../../appendices/a-differential-equations/index.md#app-de). Throughout this chapter, keep one question in mind: what does each solver integrate exactly, and what does it approximate numerically?

Continue with the sections below and the [references](../../references.md#references).

[9.1 Prologue](01-prologue.md#sec-prologue-solvers)

<span id="sec-revisit-ddim"></span>

[9.2 DDIM](02-ddim.md#sec-revisit-ddim)

[9.3 DEIS](03-deis.md#sec-exponential)

[9.4 DPM-Solver](04-dpm-solver.md#sec-dpm)

[9.5 DPM-Solver`++`](05-dpm-solver-plus-plus.md#sec-dpm)

[9.6 (Optional) ParaDiGMs](06-paradigms.md#sec-picard)

[9.7 Closing Remarks](07-closing-remarks.md#sec-ch9-cr)

<span id="eq-ddim-update-all"></span>

[Equation (9.2.2): Equivalent forms of the deterministic DDIM update](02-ddim.md#eq-ddim-update-all).

<span id="subsec-ddim"></span>

[9.2.3 (Optional) The Original Variational Perspective on DDIM](02-ddim.md#subsec-ddim).
