# 1. Deep Generative Modeling {#ch-overview-dgm}

> English source conversion of arXiv:2510.21890v3, Chapter 1. References to chapters not yet migrated open the original PDF.

> *What I cannot create, I do not understand.*
>
> — Richard P. Feynman

Deep generative models (DGMs) are neural networks that learn a probability distribution over high-dimensional data (e.g., images, text, audio) so they can generate new examples that resemble the dataset. We denote the model distribution by $p_{{\boldsymbol{\phi}}}$ and the data distribution by $p_{\mathrm{data}}$. Given a finite dataset, we fit ${\boldsymbol{\phi}}$ by minimizing a loss that measures how far $p_{{\boldsymbol{\phi}}}$ is from $p_{\mathrm{data}}$. After training, generation amounts to running the model’s sampling procedure to draw ${\mathbf{x}} \sim p_{{\boldsymbol{\phi}}}$ (the density $p_{{\boldsymbol{\phi}}}({\mathbf{x}})$ may or may not be directly computable, depending on the model class). Model quality is judged by how well generated samples and their summary statistics match those of $p_{\mathrm{data}}$, together with task-specific or perceptual metrics.

This chapter builds the mathematical and conceptual foundations behind these ideas. We formalize the problem in [Section 1.1](01-what-is-deep-generative-modeling.md#sec-formulation-dgm), present representative model classes in [Section 1.2](02-examples.md#sec-examples-dgm), and summarize a practical taxonomy in [Section 1.3](03-taxonomy.md#sec-taxonomy).

**After this chapter, you will be able to...**

-   explain, in plain language, what a generative model is trying to learn from a dataset and what it means to model the data distribution;

-   understand the core idea behind autoregressive models, VAEs, energy-based models, normalizing flows, GANs, and diffusion models, and see how they differ in the way they learn and generate new samples;

**Reader route.** If you are already familiar with the basic ideas behind modern generative models, you may skim this chapter and use it later as a quick reference.
