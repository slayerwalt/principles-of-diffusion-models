# 1.2 Prominent Deep Generative Models {#sec-examples-dgm}

> English source conversion of arXiv:2510.21890v3, Chapter 1. References to chapters not yet migrated open the original PDF.

<span id="fig-ebm-ar-vae-nf-gan-dm-unified"></span>

![Computation graphs for EBM, AR, VAE, NF, GAN, and diffusion models](../../assets/chapter-01/computation-graphs.png)

**Figure 1.2:** **Computation graphs of prominent deep generative models.** Top to bottom: **EBM** maps an input ${\mathbf{x}}$ to a scalar energy; **AR** generates a sequence $\{{\mathbf{x}}_\ell\}_{\ell=0}^{L}$ left to right with causal dependencies; **VAE** encodes ${\mathbf{x}}$ to a latent ${\mathbf{z}}$ and decodes to a reconstruction ${\mathbf{x}}'$; **NF** applies an invertible map ${\mathbf{f}}_{\boldsymbol{\phi}}$ between ${\mathbf{x}}$ and ${\mathbf{z}}$ and uses ${\mathbf{f}}_{\boldsymbol{\phi}}^{-1}$ to produce ${\mathbf{x}}'$; **GAN** transforms noise ${\mathbf{z}}$ to a sample ${\mathbf{x}}'$ that is judged against real ${\mathbf{x}}$ by a discriminator $D_{\boldsymbol\zeta}$; **DM** iteratively refines a noisy sample through a multi-step denoising chain $\{{\mathbf{x}}_\ell\}_{\ell=0}^{L}$. Boxes denote variables, trapezoids are learnable networks, ovals are scalars; arrows indicate computation flow. The trapezoid was intended to indicate a dimension-changing transformation, while rectangles denote variables or mappings that preserve dimensionality.

Source: Created by the authors.

A central challenge in generative modeling is to learn expressive probabilistic models that can capture the rich and complex structure of high-dimensional data. Over the years, various modeling strategies have been developed, each making different trade-offs between tractability, expressiveness, and training efficiency. In this section, we explore some of the most influential strategies that have shaped the field, accompanied by a comparison of their computation graphs in [Figure 1.2](02-examples.md#fig-ebm-ar-vae-nf-gan-dm-unified).

**Energy-Based Models (EBMs).** EBMs ([Ackley et al., 1985](references.md#ref-ackley1985learning); [LeCun et al., 2006](references.md#ref-lecun2006tutorial)) define a probability distribution through an energy function $E_{\boldsymbol{\phi}}({\mathbf{x}})$ that assigns lower energy to more probable data points. The probability of a data point is defined as:

$$
p_{\boldsymbol{\phi}}({\mathbf{x}}) := \frac{1}{Z({\boldsymbol{\phi}})} \exp(-E_{\boldsymbol{\phi}}({\mathbf{x}})),
$$

where

$$
Z({\boldsymbol{\phi}}) = \int \exp(-E_{\boldsymbol{\phi}}({\mathbf{x}})) \mathop{}\!\mathrm{d}{\mathbf{x}}
$$

is the partition function. Training EBMs typically involves maximizing the log-likelihood of the data. However, this requires techniques to address the computational challenges arising from the intractability of the partition function. In the following chapter, we will explore how Diffusion Models offer an alternative by generating data from *the gradient of the log density*, which does not depend on the normalizing constant, thereby circumventing the need for partition function computation.

**Autoregressive Models.** Deep autoregressive (AR) models ([Frey et al., 1995](references.md#ref-frey1995does); [Larochelle and Murray, 2011](references.md#ref-larochelle2011neural); [Uria et al., 2016](references.md#ref-uria2016neural)) build on a simple but powerful observation: any joint distribution can be decomposed into a sequence of conditional distributions using the *chain rule of probability*. For a chosen ordering of the variables $\mathbf{x}=(x_1,\ldots,x_D)$, the data distribution satisfies

$$
p_{\mathrm{data}}(\mathbf{x})
    =
    \prod_{i=1}^{D}
    p_{\mathrm{data}}(x_i | \mathbf{x}_{<i}),
    \qquad
    \mathbf{x}_{<i}=(x_1,\ldots,x_{i-1}).
$$

Importantly, this factorization is an exact identity rather than a modeling assumption: any joint distribution admits such a decomposition for any chosen ordering of its variables.

The challenge is that the true conditional distributions $p_{\mathrm{data}}(x_i|\mathbf{x}_{<i})$ are unknown. AR models therefore parameterize a model with the same factorized form,

$$
p_{\boldsymbol{\phi}}(\mathbf{x})
    =
    \prod_{i=1}^{D}
    p_{\boldsymbol{\phi}}(x_i|\mathbf{x}_{<i}),
$$

and use a neural network, such as a Transformer, to approximate these conditional distributions from data. For example, for discrete data the network may output a categorical distribution through a softmax, whereas for continuous data it may output the parameters of a tractable conditional density. Because every conditional distribution is normalized, their product automatically defines a normalized joint distribution.

This construction also makes likelihood-based training particularly simple. For a data sample $\mathbf{x}$, the log-likelihood decomposes as

$$
\log p_{\boldsymbol{\phi}}(\mathbf{x})
    =
    \sum_{i=1}^{D}
    \log p_{\boldsymbol{\phi}}(x_i|\mathbf{x}_{<i}),
$$

so the model can be trained directly by maximum likelihood, or equivalently by minimizing the negative log-likelihood. During training, the observed preceding variables $\mathbf{x}_{<i}$ are available, so many of these conditional predictions can typically be evaluated in parallel.

Generation, however, remains inherently sequential:

$$
x_1 \sim p_{\boldsymbol{\phi}}(x_1),
    \qquad
    x_2 \sim p_{\boldsymbol{\phi}}(x_2| x_1),
    \qquad
    \ldots,
    \qquad
    x_D \sim p_{\boldsymbol{\phi}}(x_D|\mathbf{x}_{<D}).
$$

Thus, AR models provide tractable exact likelihoods and flexible density modeling, but sampling requires generating variables one after another and can therefore become computationally expensive for long sequences. AR models remain a foundational class of likelihood-based generative models and underlie many modern sequence and language models.

**Variational Autoencoders (VAEs).** VAEs ([Kingma and Welling, 2013](references.md#ref-kingma2013auto)) extend classical autoencoders by introducing latent variables ${\mathbf{z}}$ that capture hidden structure in the data ${\mathbf{x}}$. Instead of directly learning a mapping between ${\mathbf{x}}$ and ${\mathbf{z}}$, VAEs adopt a probabilistic view: they learn both an *encoder*, $q_{{\boldsymbol{\theta}}}({\mathbf{z}}|{\mathbf{x}})$, which approximates the unknown distribution of latent variables given the data, and a *decoder*, $p_{{\boldsymbol{\phi}}}({\mathbf{x}}|{\mathbf{z}})$, which reconstructs data from these latent variables. To make training feasible, VAEs maximize a tractable surrogate to the true log-likelihood, called the Evidence Lower Bound (ELBO):

$$
\mathcal{L}_{\text{ELBO}}({\boldsymbol{\theta}}, {\boldsymbol{\phi}}; {\mathbf{x}}) 
    = \mathbb{E}_{q_{{\boldsymbol{\theta}}}({\mathbf{z}} | {\mathbf{x}})} \left[ \log p_{{\boldsymbol{\phi}}}({\mathbf{x}} | {\mathbf{z}}) \right] 
    - \mathcal D_{\mathrm{KL}} \left( q_{{\boldsymbol{\theta}}}({\mathbf{z}} | {\mathbf{x}}) \,\|\, p_{\mathrm{prior}}({\mathbf{z}}) \right).
$$

Here, the first term encourages accurate reconstruction of the data, while the second regularizes the latent variables by keeping them close to a simple prior distribution $p_{\mathrm{prior}}({\mathbf{z}})$ (often Gaussian).

VAEs provide a principled way to combine neural networks with latent-variable models and remain one of the most widely used likelihood-based approaches. However, they also face practical challenges, such as limited sample sharpness and training pathologies (e.g., the tendency of the decoder to ignore latent variables). Despite these limitations, VAEs laid important foundations for later advances, including diffusion models.

**Normalizing Flows.** Classic flow-based generative models include Normalizing Flows (NFs) ([Rezende and Mohamed, 2015](references.md#ref-rezende2015variational)) and their continuous-time counterparts based on Neural Ordinary Differential Equations (NODEs) ([Chen et al., 2018](references.md#ref-chen2018neural)), aim to learn a bijective mapping ${\mathbf{f}}_{\boldsymbol{\phi}}$ between a simple latent distribution ${\mathbf{z}}$ and a complex data distribution ${\mathbf{x}}$ via an invertible operator. This is achieved either through a sequence of bijective transformations (in NFs) or by modeling the transformation as an Ordinary Differential Equation (in NODEs). These models leverage the “change-of-variable formula for densities”, enabling MLE training:

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}) = \log p({\mathbf{z}}) + \log \left| \det \frac{\partial {\mathbf{f}}_{\boldsymbol{\phi}}^{-1}({\mathbf{x}})}{\partial {\mathbf{x}}} \right|,
$$

where ${\mathbf{f}}_{\boldsymbol{\phi}}$ represents the invertible transformation mapping ${\mathbf{z}}$ to ${\mathbf{x}}$. NFs explicitly model normalized densities using invertible transformations with tractable Jacobian determinants. The normalization constant is absorbed analytically via the change-of-variables formula, making likelihood computation exact and tractable.

Despite their conceptual elegance, classic flow-based models often face practical limitations. For instance, NFs typically impose restrictive architectural constraints to ensure bijectivity, while NODEs may encounter training inefficiencies due to the computational overhead of solving ODEs. Both approaches face challenges when scaling to high-dimensional data. In later chapters, we will explore how Diffusion Models relate to and build upon these classic flow-based methods.

**Generative Adversarial Networks (GANs).** GANs ([Goodfellow et al., 2014](references.md#ref-goodfellow2014generative)) consist of two neural networks, a generator ${\mathbf{G}}_{\boldsymbol{\phi}}$ and a discriminator $D_{\boldsymbol{\zeta}}$, that compete against each other. The generator aims to create realistic samples ${\mathbf{G}}_{\boldsymbol{\phi}}({\mathbf{z}})$ from random noise ${\mathbf{z}} \sim p_{\mathrm{prior}}$, while the discriminator attempts to distinguish between real samples ${\mathbf{x}}$ and generated samples ${\mathbf{G}}_{\boldsymbol{\phi}}({\mathbf{z}})$. The objective function for GANs can be formulated as:

$$
\min_{{\mathbf{G}}_{\boldsymbol{\phi}}} \max_{D_{\boldsymbol{\zeta}}} \underbrace{\mathbb{E}_{{\mathbf{x}} \sim p_{\mathrm{data}}({\mathbf{x}})}[\log D_{\boldsymbol{\zeta}}({\mathbf{x}})]}_{\text{real}} + \underbrace{\mathbb{E}_{{\mathbf{z}} \sim p_{\mathrm{prior}}({\mathbf{z}})}\left[\log(1 - D_{\boldsymbol{\zeta}}\left({\mathbf{G}}_{\boldsymbol{\phi}}({\mathbf{z}}))\right)\right]}_{\text{fake}}.
$$

GANs do not define an explicit density function and therefore bypass likelihood estimation entirely. Instead of computing a normalization constant, they focus on generating samples that closely mimic the data distribution.

From a divergence perspective, the discriminator implicitly measures the discrepancy between the true data distribution $p_{\mathrm{data}}$ and the generator distribution $p_{{\mathbf{G}}_{{\boldsymbol{\phi}}}}$, where $p_{{\mathbf{G}}_{{\boldsymbol{\phi}}}}$ denotes the distribution of generated samples ${\mathbf{G}}_{{\boldsymbol{\phi}}}({\mathbf{z}})$ obtained from noise ${\mathbf{z}} \sim p_{\mathrm{prior}}$. With an optimal discriminator for a fixed generator ${\mathbf{G}}_{{\boldsymbol{\phi}}}$ computed as

$$
\frac{p_{\mathrm{data}}({\mathbf{x}})}{p_{\mathrm{data}}({\mathbf{x}}) + p_{{\mathbf{G}}_{{\boldsymbol{\phi}}}}({\mathbf{x}})},
$$

the generator’s minimization reduces to

$$
\min_{{\mathbf{G}}_{{\boldsymbol{\phi}}}} \; 2\,\mathcal{D}_\mathrm{JS} \left(p_{\mathrm{data}} \,\|\, p_{{\mathbf{G}}_{\boldsymbol{\phi}}}\right) - \log 4.
$$

Here, $\mathcal{D}_\mathrm{JS}$ denotes the Jensen–Shannon divergence, defined as

$$
\mathcal{D}_\mathrm{JS}(p \,\|\, q) 
    := \tfrac{1}{2} \mathcal{D}_\mathrm{KL} \left(p \,\middle\|\, \tfrac{p+q}{2}\right) 
    + \tfrac{1}{2} \mathcal{D}_\mathrm{KL} \left(q \,\middle\|\, \tfrac{p+q}{2}\right).
$$

This shows that GANs implicitly minimize $\mathcal{D}_\mathrm{JS}(p_{\mathrm{data}} \,\|\, p_{{\mathbf{G}}_{\boldsymbol{\phi}}})$. More broadly, extensions such as $f$-GANs ([Nowozin et al., 2016](references.md#ref-nowozin2016f)) generalize this view by demonstrating that adversarial training can minimize a family of $f$-divergences, placing GANs within the same divergence-minimization framework as other generative models.

Although GANs are capable of generating high-quality data, their min-max training process is notoriously unstable, often requiring carefully designed architectures and engineering techniques to achieve satisfactory performance. However, GANs have since been revived as an auxiliary component to enhance other generative models, particularly Diffusion Models.
