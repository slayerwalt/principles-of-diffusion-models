# 4. Diffusion Models Today: Score SDE Framework {#ch-score-sde}

> *There is only one precise way of presenting the laws, and that is by means of differential equations. They have the advantage of being fundamental and, so far as we know, precise.*
>
> — Richard P. Feynman

So far, we have studied diffusion models from two perspectives: the variational view and the score-based view, the latter naturally emerging from the EBM formulation. We now take the next step and move to the *continuous-time framework*. At its core lies the *Score SDE* ([Song et al., 2020](references.md#ref-song2020score)), which provides a continuous-time view that brings DDPM and NCSN under the same framework.

Instead of working with only a finite sequence of noise levels, we imagine inserting more and more intermediate levels between data and noise. In the continuous-time limit, the noise level varies smoothly with time, giving the basic picture

$$
p_{\mathrm{data}}
    \;\xrightarrow{\text{forward SDE}}\;
    p_T \approx p_{\mathrm{prior}},
    \qquad
    p_{\mathrm{prior}}
    \;\xrightarrow{\text{reverse dynamics}}\;
    p_{\mathrm{data}}.
$$

The forward SDE describes how data is gradually corrupted into noise, playing the continuous-time role of the noise schedule in DDPM or the sequence of noise levels in NCSN. Once this forward process is specified, the main challenge is generation in the opposite direction: starting from a simple noise distribution, how can we recover the data distribution?

The key quantity is the time-dependent score

$$
\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}}),
$$

which describes the local geometry of the noisy distribution at every time $t$. Once this score is known, or approximated by a neural network, it determines the reverse-time dynamics used for generation, transforming noise into data. This leads to two complementary descriptions: a stochastic *reverse-time SDE* and a deterministic *probability flow ODE* (PF-ODE). Although their sample trajectories are different, they can follow the same evolution of marginal distributions.

This continuous-time view turns generation into the problem of solving differential equations, opening the door to a broad set of tools from stochastic processes, differential equations, and numerical analysis. In particular, once the generative dynamics are written as an SDE or ODE, standard numerical methods can be used to simulate them, while more advanced solvers can improve sampling efficiency and accuracy. This perspective therefore provides not only a unified mathematical description of diffusion models, but also a practical foundation for understanding and designing faster sampling methods developed later in this book.

**After this chapter, you will be able to...**

-   understand how the discrete noise levels of DDPM and NCSN lead to a continuous-time forward SDE that gradually transforms data into noise;

-   understand how generation from noise back to data can be described either by a stochastic reverse-time SDE or by a deterministic probability flow ODE (PF-ODE);

-   explain how the reverse-time SDE and PF-ODE can follow the same sequence of marginal distributions even though their individual sample trajectories are different, and understand how the Fokker–Planck equation describes the continuous-time evolution of probability densities;

-   understand how the time-dependent score is learned by denoising score matching and then used for sampling through the reverse-time SDE or PF-ODE;

-   understand how the PF-ODE can also be run forward for inversion and track changes in probability density through the continuous change-of-variables formula, enabling likelihood evaluation;

-   recognize VE and VP SDEs as two important instances of the same continuous-time diffusion framework, connecting back to NCSN and DDPM.

**Reader route.** Readers new to stochastic differential equations may consult the differential-equation appendices in [Appendix A](../../appendices/a-differential-equations/index.md#app-de) as needed. For a first pass, focus on the roles of the forward SDE, reverse-time SDE, and PF-ODE, together with their training and sampling procedures. The Gaussian example and optional derivations can be revisited after the main picture is clear.

Continue with the sections below and the [chapter references](references.md#chapter-04-references).

[4.1 Score SDE: Its Principles](01-score-sde-principles.md#sec-dm-today)

[4.2 Matching Marginals in Forward/Reverse-Time SDEs and PF-ODE](02-matching-marginals.md#subsec-fpe-sde-ode)

[4.3 Score SDE: Its Training and Sampling](03-training-and-sampling.md#sec-scoresde-training-sampling)

[4.4 Instantiations of SDEs](04-sde-instantiations.md#sec-instances)

[4.5 A Direct View Through Accumulated Perturbation Kernels](05-accumulated-perturbation-kernels.md#sec-variational-perspective)

[4.6 (Optional) Fokker–Planck Equation and Reverse-Time SDEs via Marginalization and Bayes’ Rule](06-fokker-planck-and-reverse-sde.md#eq-heuristic-fpe-sde)

[4.7 Closing Remarks](07-closing-remarks.md#sec-ch4-cr)

<span id="dsm-minimizer"></span>

[Proposition 4.3.1](03-training-and-sampling.md#dsm-minimizer)

<span id="sec-variational-perspective"></span>

[Section 4.5](05-accumulated-perturbation-kernels.md#sec-variational-perspective)

<span id="reverse-kernel"></span>

[Lemma 4.5.2](05-accumulated-perturbation-kernels.md#reverse-kernel)
