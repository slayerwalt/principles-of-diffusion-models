# 3.5 Summary: A Comparative View of NCSN and DDPM {#sec-comparison-ncsn-ddpm}

We begin by comparing the graphical models of NCSN and DDPM in [Figure 3.7](04-noise-conditional-score-networks.md#fig-smld-ddpm-graphical-model), with key differences and similarities summarized in [Table 3.1](05-comparison.md#tb-comparison-smld-ddpm).

<span id="tb-comparison-smld-ddpm"></span>

**Table 3.1: Comparisons of NCSN and DDPM**

|  | **NCSN** | **DDPM** |
| --- | --- | --- |
| ${\mathbf{x}}_{i+1}\vert {\mathbf{x}}_{i}$ | Derive as ${\mathbf{x}}_{i+1} = {\mathbf{x}}_i + \sqrt{\sigma^2_{i+1}-\sigma^2_{i}}\boldsymbol{\epsilon}$ | Define as ${\mathbf{x}}_{i+1} = \sqrt{1-\beta_{i}} {\mathbf{x}}_{i} + \sqrt{\beta_{i}}\boldsymbol{\epsilon}$ |
| ${\mathbf{x}}_{i}\vert {\mathbf{x}}$ | Define as ${\mathbf{x}}_{i} = {\mathbf{x}} + \sigma_{i}\boldsymbol{\epsilon}$ | Derive as ${\mathbf{x}}_i = \bar{\alpha}_i{\mathbf{x}} + \sqrt{1 - \bar{\alpha}_i^2}\boldsymbol{\epsilon}$ |
| $p_{\text{prior}}$ | $\mathcal{N}(\boldsymbol{0}, \sigma_L^2{\mathbf{I}})$ | $\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$ |
| Loss | $\mathbb{E}_{i} \mathbb{E}_{p_{\text{data}}({\mathbf{x}})} \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}\left(\mathbf{0}, {\mathbf{I}}\right)} \left[ \left\lVert{\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_i, \sigma_i) + \frac{\boldsymbol{\epsilon}}{\sigma_i}\right\rVert_2^2 \right]$ | $\mathbb{E}_{i}\mathbb{E}_{p_{\text{data}}({\mathbf{x}})} \mathbb{E}_{\boldsymbol{\epsilon}\sim \mathcal{N}(\boldsymbol{0},{\mathbf{I}})}\left[ \left\lVert\boldsymbol{\epsilon}_{\boldsymbol{\phi}}({\mathbf{x}}_i, i) - \boldsymbol{\epsilon}\right\rVert_2^2 \right]$ |
| Sampling | Apply Langevin per layer;<br> use output to initialize the next | Traversing the Markovian chain with <br> $p_{\boldsymbol{\phi}^\times}({\mathbf{x}}_{i-1} \vert  {\mathbf{x}}_i)$ |

**A Shared Bottleneck.** Despite their different formulations, both NCSN and DDPM rely on dense time discretization. This leads to a critical limitation: sampling often requires hundreds or even thousands of iterations, making generation slow and computationally intensive.

<span id="ques-dm-sampling-slow"></span>

**Question 3.5.1**

How can we accelerate sampling in diffusion models?

We return to this challenge in [Chapter 9](../09-numerical-solvers/index.md#ch-numerical-solvers), which develops more efficient samplers; [Chapter 10](../10-distillation/index.md#ch-distillation), which compresses pretrained diffusion models requiring hundreds of sampling steps into fewer-step student generators; and [Chapter 11](../11-fast-generators/index.md#ch-fast-scratch), which studies how to learn fast few-step generators directly from scratch under diffusion-inspired principles.
