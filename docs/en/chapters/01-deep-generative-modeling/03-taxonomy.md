# 1.3 Taxonomy of Modeling {#sec-taxonomy}

As we have seen, DGMs span a wide spectrum of modeling strategies. A fundamental distinction lies in how these models *parameterize* the underlying data distribution, that is, whether they specify $p_{\boldsymbol{\phi}}(\mathbf{x})$ *explicitly* or only *implicitly*, irrespective of the training objective.

-   **Explicit Models:** These models directly parameterize a probability distribution $p_{\boldsymbol{\phi}}(\mathbf{x})$ via a tractable or approximately tractable density or mass function. Examples include ARs, NFs, VAEs, and DMs, all of which define $p_{\boldsymbol{\phi}}(\mathbf{x})$ either exactly or through a tractable bound.

-   **Implicit Models:** These models specify a distribution only through a sampling procedure, typically of the form $\mathbf{x} = {\mathbf{G}}_{\boldsymbol{\phi}}(\mathbf{z})$ for some noise variable $\mathbf{z} \sim p_{\mathrm{prior}}$. In this case, $p_{\boldsymbol{\phi}}(\mathbf{x})$ is not available in closed form and may not be defined at all.

The table in [Table 1.1](03-taxonomy.md#tb-comparison-explicit-implicit) offers a concise summary of these contrasting approaches.

<span id="tb-comparison-explicit-implicit"></span>

**Table 1.1: Comparison of Explicit and Implicit Generative Models**

| | Explicit: Exact Likelihood | Explicit: Approx. Likelihood | Implicit |
| --- | --- | --- | --- |
| **Likelihood** | Tractable | Bound/Approx. | Not Directly Modeled/Intractable |
| **Objective** | MLE | ELBO | Adversarial |
| **Examples** | NFs, ARs | VAEs, DMs | GANs |

**Connection to Diffusion Models.** Taken together, these classical families of DGMs illustrate complementary strategies for modeling complex distributions. Beyond their standalone importance, they also provide guiding principles for understanding diffusion models. Diffusion methods inherit ideas from several of these perspectives: they connect to VAEs through variational training objectives, to EBMs through score-matching approaches that learn gradients of the log-density (closely tied to energy functions), and to NFs through continuous-time transformations.

To lay the groundwork for the diffusion methods discussed in later chapters, we will focus on three central paradigms: VAEs ([Section 2.1](../02-variational-perspective/01-variational-autoencoder.md#sec-vae)), EBMs ([Section 3.1](../03-score-based-perspective/01-energy-based-models.md#sec-ebm)), and NFs ([Section 5.1](../05-flow-based-perspective/index.md#sec-flow-based-method)). This exploration provides a foundation for the core principles that underlie modern diffusion-based generative modeling, which will be developed further in the chapters that follow.
