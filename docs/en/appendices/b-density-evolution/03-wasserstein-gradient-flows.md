# B.3 (Optional) Wasserstein Gradient Flows as Distribution-Level Training {#sec-wgf-distribution-level-training}

We now step back from diffusion-specific dynamics and take a broader distribution-level view of deep generative modeling. We have discussed that the continuity equation describes how a probability distribution moves when its particles move. The same language can also be used to understand the training of deep generative models: a training algorithm updates the model parameters, the updated generator moves generated particles, and the movement of these particles changes the induced model distribution $p_{{\boldsymbol{\phi}}}$.

From this viewpoint, different generative modeling methods can be organized according to how the model distribution $p_{{\boldsymbol{\phi}}}$ is driven toward the data distribution $p_{\mathrm{data}}$. Wasserstein gradient flow gives one particularly clean way to describe this movement: it treats training as an idealized flow of probability mass in distribution space, which is then approximated by finite-dimensional neural network updates.

A neural generator ${\mathbf{G}}_{{\boldsymbol{\phi}}}$ defines both a function and a probability distribution. If

$$
{\mathbf{z}}\sim p_{\mathrm{prior}},
    \qquad
    {\mathbf{x}}={\mathbf{G}}_{{\boldsymbol{\phi}}}({\mathbf{z}}),
$$

then the generated distribution is the pushforward

$$
p_{{\boldsymbol{\phi}}}:=({\mathbf{G}}_{{\boldsymbol{\phi}}})_\# p_{\mathrm{prior}}.
$$

From the generative modeling viewpoint, the object we ultimately care about is how $p_{{\boldsymbol{\phi}}}$ approaches the data distribution $p_{\mathrm{data}}$.

This motivates a distribution-level question:

<span id="question-B-3-1"></span>

**Question B.3.1**

Can we first describe the ideal way to move the model distribution $p_{{\boldsymbol{\phi}}}$ toward $p_{\mathrm{data}}$, and then update the neural network parameters so that the generator approximately realizes this movement?

Throughout this optional section, $\tau$ denotes *training time*: an idealized continuous-time version of the discrete training iteration index. It should not be confused with diffusion/noising time. We write ${\boldsymbol{\phi}}_\tau$ for the parameter at training time $\tau$, and $p_{{\boldsymbol{\phi}}_\tau}$ for the corresponding model distribution.

**Moving Model Distributions by Moving Particles.** We first describe an ideal distribution-level evolution, before accounting for the finite-dimensional parameterization of the neural generator. Suppose particles from the current model distribution move according to a training-time velocity field ${\mathbf{w}}_\tau$:

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_\tau}{\mathop{}\!\mathrm{d} \tau}
    =
    {\mathbf{w}}_\tau({\mathbf{x}}_\tau),
    \qquad
    {\mathbf{x}}_\tau\sim p_{{\boldsymbol{\phi}}_\tau}.
$$

Then the model distribution evolves according to the continuity equation

$$
\partial_\tau p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}})
    +
    \nabla_{{\mathbf{x}}}\cdot
    \big(
        p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}}){\mathbf{w}}_\tau({\mathbf{x}})
    \big)
    =
    0.
$$

Thus, ${\mathbf{w}}_\tau$ describes how each generated particle moves, while the continuity equation describes how the entire generated distribution moves[^1].

**Choosing the Velocity: Gradient Descent in Distribution Space.** Now we need to choose a velocity field ${\mathbf{w}}_\tau$ that moves $p_{{\boldsymbol{\phi}}_\tau}$ toward $p_{\mathrm{data}}$. Let

$$
\mathcal{D}(p_{{\boldsymbol{\phi}}_\tau},p_{\mathrm{data}})
$$

be a discrepancy between the current model distribution and the data distribution. *Wasserstein gradient flow* ([Jordan et al., 1998](../../references.md#ref-jordan1998variational); [Ambrosio et al., 2005](../../references.md#ref-ambrosio2005gradient)) chooses the particle velocity that decreases this discrepancy most directly under the geometry of moving probability mass.

To choose the velocity field, we first ask how the discrepancy changes when the current distribution is infinitesimally perturbed. Think of $\mathcal{D}(p,p_{\mathrm{data}})$ as a global energy of the distribution $p$. Its first variation gives a pointwise energy potential: it tells us which regions of space are costly for the current distribution and which regions are favorable.

Concretely, for a small perturbation $p+\epsilon h$, where $\epsilon>0$ is small and $\int h({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}=0$, the first variation is defined by

$$
\mathcal{D}(p+\epsilon h,p_{\mathrm{data}})
    =
    \mathcal{D}(p,p_{\mathrm{data}})
    +
    \epsilon
    \int
    \frac{\delta\mathcal{D}}{\delta p}
    (p,p_{\mathrm{data}})({\mathbf{x}})\,
    h({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}
    +
    \mathop{\mathrm{o}}(\epsilon).
$$

Here $p$ is a dummy distribution variable. After taking the variation, we evaluate it at the current model distribution $p=p_{{\boldsymbol{\phi}}_\tau}$.

We define the resulting pointwise energy potential by

$$
E_\tau({\mathbf{x}})
    :=
    \frac{\delta\mathcal{D}}{\delta p}
    (p_{{\boldsymbol{\phi}}_\tau},p_{\mathrm{data}})({\mathbf{x}}).
$$

Intuitively, if $E_\tau({\mathbf{x}})$ is large, then placing more probability mass near ${\mathbf{x}}$ would increase the discrepancy. If $E_\tau({\mathbf{x}})$ is small, then moving mass toward ${\mathbf{x}}$ is favorable. Therefore, the Wasserstein gradient-flow velocity moves particles downhill in this potential:

$$
{\mathbf{w}}_\tau({\mathbf{x}})
    =
    -
    \nabla_{{\mathbf{x}}}E_\tau({\mathbf{x}}).
$$

With this choice,

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} \tau}
    \mathcal{D}(p_{{\boldsymbol{\phi}}_\tau},p_{\mathrm{data}})
    =
    -
    \int
    p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}})
    \left\|
        \nabla_{{\mathbf{x}}}E_\tau({\mathbf{x}})
    \right\|_2^2
    \mathop{}\!\mathrm{d}{\mathbf{x}}
    \le 0.
$$

Therefore, Wasserstein gradient flow is gradient descent in the $2$-Wasserstein geometry of probability distributions[^2]. The global discrepancy $\mathcal{D}$ defines the energy of the whole distribution, the first variation $E_\tau$ defines the local energy landscape seen by particles, and $-\nabla_{{\mathbf{x}}}E_\tau$ gives the ideal particle velocity for decreasing the discrepancy.

**Realizing the Distributional Velocity with a Neural Network.** The velocity above is a distribution-level object: it says how generated particles should move. A neural generator, however, cannot move every particle independently. It can only move particles through a shared parameter update of ${\boldsymbol{\phi}}$.

At training time $\tau$, the current generator produces particles

$$
{\mathbf{x}}_i
    =
    {\mathbf{G}}_{{\boldsymbol{\phi}}_\tau}({\mathbf{z}}_i),
    \qquad
    {\mathbf{z}}_i\sim p_{\mathrm{prior}},
    \qquad
    i=1,\ldots,N.
$$

A small Wasserstein step moves these particles to

$$
\widetilde{{\mathbf{x}}}_i
    =
    {\mathbf{x}}_i+\eta{\mathbf{w}}_\tau({\mathbf{x}}_i),
$$

where $\eta>0$ is a small step size. The neural network is then updated so that its new outputs match these moved particles:

$$
{\boldsymbol{\phi}}_{\tau+\eta}
    \approx
    \arg\min_{{\boldsymbol{\phi}}}
    \frac{1}{N}
    \sum_{i=1}^N
    \left\|
        {\mathbf{G}}_{{\boldsymbol{\phi}}}({\mathbf{z}}_i)
        -
        \operatorname{sg}
        \big(
            {\mathbf{x}}_i+\eta{\mathbf{w}}_\tau({\mathbf{x}}_i)
        \big)
    \right\|_2^2.
$$

Here $\operatorname{sg}(\cdot)$ denotes stop-gradient, so the moved particles are treated as fixed targets.

For a small parameter update $\Delta{\boldsymbol{\phi}}$, we have the linearization

$$
{\mathbf{G}}_{{\boldsymbol{\phi}}_\tau+\Delta{\boldsymbol{\phi}}}({\mathbf{z}}_i)
    \approx
    {\mathbf{G}}_{{\boldsymbol{\phi}}_\tau}({\mathbf{z}}_i)
    +
    \big(\partial_{{\boldsymbol{\phi}}}{\mathbf{G}}_{{\boldsymbol{\phi}}_\tau}({\mathbf{z}}_i)\big)\Delta{\boldsymbol{\phi}}.
$$

Therefore, the parameter update approximately solves

$$
\min_{\Delta{\boldsymbol{\phi}}}
    \frac{1}{N}
    \sum_{i=1}^N
    \left\|
        \big(\partial_{{\boldsymbol{\phi}}}{\mathbf{G}}_{{\boldsymbol{\phi}}_\tau}({\mathbf{z}}_i)\big)\Delta{\boldsymbol{\phi}}
        -
        \eta{\mathbf{w}}_\tau({\mathbf{x}}_i)
    \right\|_2^2.
$$

In words, Wasserstein gradient flow specifies the desired distribution-level particle motion, while neural network training projects this motion onto the set of movements that can be realized by changing the finite-dimensional parameter ${\boldsymbol{\phi}}$.

**Forward-KL Training: Data-Side Likelihood Learning.** A classical distribution-level objective is the data-side, or forward, KL divergence:

$$
\mathcal{D}_{\mathrm{FKL}}(p_{{\boldsymbol{\phi}}},p_{\mathrm{data}})
    :=
    \mathcal D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{{\boldsymbol{\phi}}})
    =
    \int
    p_{\mathrm{data}}({\mathbf{x}})
    \log
    \frac{p_{\mathrm{data}}({\mathbf{x}})}{p_{{\boldsymbol{\phi}}}({\mathbf{x}})}
    \mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

Since $p_{\mathrm{data}}$ is fixed, minimizing this objective is equivalent to maximum likelihood (see [Equation (1.1.2)](../../chapters/01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-mle)):

$$
\min_{{\boldsymbol{\phi}}}
    \mathcal D_{\mathrm{KL}}(p_{\mathrm{data}}\|p_{{\boldsymbol{\phi}}})
    \quad
    \Longleftrightarrow
    \quad
    \max_{{\boldsymbol{\phi}}}
    {\mathbb{E}}_{{\mathbf{x}}\sim p_{\mathrm{data}}}
    \big[
        \log p_{{\boldsymbol{\phi}}}({\mathbf{x}})
    \big].
$$

This is the distribution-level view behind many likelihood-based models. Autoregressive models and normalizing flows optimize exact likelihoods; likelihood-based VAEs optimize variational lower bounds; energy-based models optimize likelihood through positive and negative phases; and diffusion models can be related to likelihood or score-matching surrogates through their variational and denoising objectives.

From the Wasserstein gradient-flow viewpoint, the forward KL has first variation

$$
\frac{\delta}{\delta p}
    \mathcal D_{\mathrm{KL}}(p_{\mathrm{data}}\|p)
    =
    -
    \frac{p_{\mathrm{data}}}{p}.
$$

Evaluating at $p=p_{{\boldsymbol{\phi}}_\tau}$ gives the velocity

$$
{\mathbf{w}}_\tau^{\mathrm{FKL}}({\mathbf{x}})
    =
    \nabla_{{\mathbf{x}}}
    \left(
        \frac{p_{\mathrm{data}}({\mathbf{x}})}
             {p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}})}
    \right).
$$

This expression involves the density ratio $p_{\mathrm{data}}/p_{{\boldsymbol{\phi}}_\tau}$, which is usually difficult to estimate for implicit generators. In practice, forward-KL or maximum-likelihood training is most direct when likelihoods, variational bounds, or suitable approximations to the likelihood gradient are available. Score matching provides a different route to distribution learning by matching score functions rather than directly realizing the forward-KL Wasserstein flow.

The main strength of forward-KL training is that it is data-covering and statistically well grounded when a likelihood or a variational surrogate is available. Its limitation is that it often requires tractable densities, variational bounds, score-matching surrogates, or expensive negative sampling, which makes it less directly compatible with arbitrary implicit generators.

**Reverse-KL Training: Model-Side Score-Mismatch Dynamics.** The model-side, or reverse, KL divergence is

$$
\mathcal{D}_{\mathrm{RKL}}(p_{{\boldsymbol{\phi}}},p_{\mathrm{data}})
    :=
    \mathcal D_{\mathrm{KL}}(p_{{\boldsymbol{\phi}}}\|p_{\mathrm{data}})
    =
    \int
    p_{{\boldsymbol{\phi}}}({\mathbf{x}})
    \log
    \frac{p_{{\boldsymbol{\phi}}}({\mathbf{x}})}
         {p_{\mathrm{data}}({\mathbf{x}})}
    \mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

Its first variation is

$$
\frac{\delta}{\delta p}
    \mathcal D_{\mathrm{KL}}(p\|p_{\mathrm{data}})
    =
    \log p-\log p_{\mathrm{data}}+1.
$$

Evaluating at $p=p_{{\boldsymbol{\phi}}_\tau}$ gives the Wasserstein velocity

$$
{\mathbf{w}}_\tau^{\mathrm{RKL}}({\mathbf{x}})
    :=
    \nabla_{{\mathbf{x}}}\log p_{\mathrm{data}}({\mathbf{x}})
    -
    \nabla_{{\mathbf{x}}}\log p_{{\boldsymbol{\phi}}_\tau}({\mathbf{x}}).
$$

Thus, reverse-KL training induces a score-mismatch force:

$$
\text{target score}
    -
    \text{model score}.
$$

The target score attracts generated particles toward high-density regions of the data distribution, while the model score repels particles from regions where the current model distribution is already too concentrated.

In diffusion distillation and distribution matching (see [Section 10.2](../../chapters/10-distillation/02-distribution-based-distillation.md#sec-vsd)), this idea is usually applied at a noise level $s$. Let $p_s$ denote the teacher or noised-data distribution, and let $p_{{\boldsymbol{\phi}}_\tau,s}$ denote the noised distribution of current generator samples. The reverse-KL objective

$$
{\mathbb{E}}_s
    \left[
        w(s)
        \mathcal D_{\mathrm{KL}}(p_{{\boldsymbol{\phi}}_\tau,s}\|p_s)
    \right]
$$

induces the score-mismatch velocity

$$
{\mathbf{w}}_{\tau,s}^{\mathrm{RKL}}({\mathbf{x}})
    :=
    \nabla_{{\mathbf{x}}}\log p_s({\mathbf{x}})
    -
    \nabla_{{\mathbf{x}}}\log p_{{\boldsymbol{\phi}}_\tau,s}({\mathbf{x}}).
$$

At the parameter level, for generated noisy samples $\hat{\mathbf{x}}_s\sim p_{{\boldsymbol{\phi}}_\tau,s}$, one obtains gradients of the form

$$
\nabla_{{\boldsymbol{\phi}}}
    \mathcal D_{\mathrm{KL}}(p_{{\boldsymbol{\phi}},s}\|p_s)
    \big|_{{\boldsymbol{\phi}}={\boldsymbol{\phi}}_\tau}
    =
    {\mathbb{E}}_{\hat{\mathbf{x}}_s\sim p_{{\boldsymbol{\phi}}_\tau,s}}
    \left[
        \left(
            \nabla_{{\mathbf{x}}}\log p_{{\boldsymbol{\phi}}_\tau,s}(\hat{\mathbf{x}}_s)
            -
            \nabla_{{\mathbf{x}}}\log p_s(\hat{\mathbf{x}}_s)
        \right)^\top
        \partial_{{\boldsymbol{\phi}}}\hat{\mathbf{x}}_s
    \right].
$$

Thus, the score difference

$$
\nabla_{{\mathbf{x}}}\log p_s(\hat{\mathbf{x}}_s)
-
\nabla_{{\mathbf{x}}}\log p_{{\boldsymbol{\phi}}_\tau,s}(\hat{\mathbf{x}}_s)
$$

provides the distribution-level descent direction that drives the parameter gradient. The actual movement of individual generated samples is mediated by the shared neural parameterization, as discussed above. This score-difference gradient is the basic distribution-level mechanism behind reverse-KL-based methods such as DMD and VSD: estimate the target and model scores on generated samples and use their difference to update the generator.

The main strength of reverse-KL training is that it is naturally compatible with implicit generators once the relevant scores can be estimated. It is also mode-seeking, which can improve visual fidelity in one-step or few-step generation. Its limitation is that it can drop modes, depends strongly on the quality of score estimates, and may require an additional model-score network or a strong teacher.

The same attraction-repulsion interpretation also appears in drifting-style particle updates ([Weber, 2023](../../references.md#ref-weber2023the); [Deng et al., 2026](../../references.md#ref-deng2026generative)). In the terminology of [Lai et al. (2026)](../../references.md#ref-lai2026unified), DMD/VSD/SiD provide parametric score-based instantiations of this score-mismatch dynamics, while drifting-style approaches ([Weber, 2023](../../references.md#ref-weber2023the); [Deng et al., 2026](../../references.md#ref-deng2026generative)) provide a nonparametric kernel-based instantiation.

**Closing Remarks.** In summary, the continuity equation provides the common language behind both diffusion dynamics and distribution-level training. In diffusion models, it describes how the intermediate distributions $p_t$ evolve along a prescribed noise or probability-flow dynamics. In Wasserstein gradient flow, the same equation describes how the model distribution $p_{{\boldsymbol{\phi}}_\tau}$ evolves during training as generated particles move toward the data distribution $p_{\mathrm{data}}$.

This viewpoint therefore broadens the role of the continuity equation beyond diffusion models. It lets us interpret generative modeling as the design of distributional dynamics: different methods choose different discrepancies, different particle velocities, and different parameter updates to drive $p_{{\boldsymbol{\phi}}}$ toward $p_{\mathrm{data}}$. From this perspective, diffusion models, likelihood-based models, distribution-matching methods, and score-mismatch-based approaches can be viewed under a shared distribution-level taxonomy.

[^1]: This velocity should not be confused with the diffusion-time PF-ODE velocity ${\mathbf{v}}^*({\mathbf{x}},t)$. The variable $t$ or $s$ describes noise/denoising time, whereas $\tau$ describes training-time evolution of the model distribution.

[^2]: We remark that this ideal distribution-space dynamics should not be confused with ordinary gradient descent on the neural parameters ${\boldsymbol{\phi}}$: the latter is constrained by the finite-dimensional generator family and generally realizes only an approximation to the desired particle motion.
