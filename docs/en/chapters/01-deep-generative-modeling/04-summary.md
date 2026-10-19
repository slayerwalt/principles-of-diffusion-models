# 1.4 Closing Remarks {#sec-ch1-cr}

This chapter has established the foundational concepts of deep generative modeling. We begin by defining the primary objective: to learn a tractable model distribution $p_{\text{model}}$ (parametrized by $\boldsymbol{\phi}$) that approximates an unknown, complex data distribution $p_{\text{data}}$. A central challenge is the computational intractability of the normalizing constant, or partition function $Z(\boldsymbol{\phi})$, which is required to define a valid probability density.

To circumvent this problem, various families of deep generative models have been developed, each employing a distinct strategy. We surveyed several prominent approaches, including Energy-Based Models (EBMs), Autoregressive Models (ARs), Variational Autoencoders (VAEs), Normalizing Flows (NFs), and Generative Adversarial Networks (GANs). These models can be broadly categorized into explicit models, which define a tractable density, and implicit models, which define a distribution only through a sampling procedure.

While each of these classical frameworks is significant, three in particular serve as the conceptual origins for the diffusion models that are the focus of this monograph: VAEs, EBMs, and NFs. In the chapters that follow, we will trace the evolution of diffusion models from these three foundational paradigms:

1.  Part B will begin by exploring the variational perspective ([Chapter 2](../02-variational-perspective/index.md#ch-variational)), showing how (the hierarchical latent variable structure of) VAEs leads naturally to the formulation of Denoising Diffusion Probabilistic Models (DDPMs).

2.  Next, we will examine the score-based perspective ([Chapter 3](../03-score-based-perspective/index.md#ch-score-based)), which originates from EBMs and score matching, and develops into Noise Conditional Score Networks (NCSN) and the more general Score SDE framework ([Chapter 4](../04-score-sde-framework/index.md#ch-score-sde)).

3.  Finally, we will investigate the flow-based perspective ([Chapter 5](../05-flow-based-perspective/index.md#ch-flow-based)), which builds upon the principles of Normalizing Flows to frame generation as a continuous transformation, generalized by the concept of Flow Matching.

By understanding these origins, we will build a coherent framework for interpreting the diverse formulations of diffusion models and uncovering the deep principles that unify them.
