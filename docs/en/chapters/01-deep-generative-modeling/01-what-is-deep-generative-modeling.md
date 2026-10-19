# 1.1 What is Deep Generative Modeling? {#sec-formulation-dgm}

DGMs take as input a large collection of real-world examples (e.g., images, text) drawn from an unknown and complex distribution $p_{\mathrm{data}}$ and output a trained neural network that parameterizes an approximate distribution $p_{{\boldsymbol{\phi}}}$. Their goals are twofold:

1.  **Realistic Generation:** To generate novel, realistic samples indistinguishable from real data.

2.  **Controllable Generation:** To enable fine-grained and interpretable control over the generative process.

This section presents the fundamental concepts and motivations behind DGMs, preparing for a detailed exploration of their mathematical framework and practical applications.

## 1.1.1 Mathematical Setup {#subsec-intro-math-setup}

Suppose we are given a finite collection of examples $\{{\mathbf{x}}^{(i)}\}_{i=1}^N$, such as images, audio, or text. We imagine that these examples were drawn from some underlying data distribution $p_{\mathrm{data}}$, but we do not know this distribution explicitly. In particular, we only observe the available training examples; we do not have a formula for $p_{\mathrm{data}}$ or a direct procedure for drawing arbitrary new samples from it[^1].

**Goal of DGM.** The goal of deep generative modeling is therefore to learn a model that can produce *new* samples whose statistical characteristics resemble those of the observed data. We denote the distribution represented by a model with parameters $\boldsymbol{\phi}$ by $p_{\boldsymbol{\phi}}$. After training, we denote the learned parameters by $\boldsymbol{\phi}^{\times}$ and hope that, in the aspects relevant to generation,

$$
p_{\boldsymbol{\phi}^{\times}}
    \approx
    p_{\mathrm{data}}.
$$

This expression should be understood conceptually: with finite data, finite model capacity, and imperfect optimization, we do not expect the two distributions to be exactly identical. Rather, the learned model should capture enough of the structure of the data distribution to generate realistic and diverse new examples.

Once the model is trained, generation simply means drawing a new sample

$$
{\mathbf{x}} \sim p_{\boldsymbol{\phi}^{\times}}.
$$

Different families of generative models provide different mechanisms for doing this. Some models additionally allow us to evaluate how much density they assign to an observed sample, while others are primarily defined through their sampling procedure. This distinction will become useful later when we discuss how different generative models are trained; for now, the central objective is simply to learn a distribution from data from which we can generate new samples.

<span id="fig-dgm-training"></span>

![Target and model distributions with a dataset of samples](../../assets/chapter-01/dgm-learning.png)

**Figure 1.1:** **Illustration of the target in DGM.** Training a DGM is essentially minimizing the discrepancy between the model distribution $p_{\boldsymbol{\phi}}$ and the unknown data distribution $p_{\mathrm{data}}$. Since $p_{\mathrm{data}}$ is not directly accessible, this discrepancy must be estimated efficiently using a finite set of independent and identically distributed (i.i.d.) samples, ${\mathbf{x}}_i$, drawn from it.

Source: Created by the authors.

**Training of DGM.** We learn parameters $\boldsymbol{\phi}$ of a model family $\{p_{\boldsymbol{\phi}}\}$ by minimizing a discrepancy $\mathcal{D}(p_{\mathrm{data}},p_{\boldsymbol{\phi}})$:

<span id="eq-dgm-optimization"></span>

$$
\begin{aligned}
\boldsymbol{\phi}^\ast \in \arg\min_{\boldsymbol{\phi}}\; \mathcal{D}(p_{\mathrm{data}},p_{\boldsymbol{\phi}}).
\end{aligned}
\tag{1.1.1}
$$

Because $p_{\mathrm{data}}$ is unknown, a practical choice of $\mathcal{D}$ must admit efficient estimation from i.i.d. samples from $p_{\mathrm{data}}$. With sufficient capacity, $p_{\boldsymbol{\phi}^\ast}$ can closely approximate $p_{\mathrm{data}}$.

**Forward KL and Maximum Likelihood Estimation (MLE).** How should we choose the model parameters so that $p_{\boldsymbol{\phi}}$ resembles $p_{\mathrm{data}}$? When the model density $p_{\boldsymbol{\phi}}({\mathbf{x}})$ can be evaluated, a natural idea is to make the observed data receive high density under the model. This leads to *maximum likelihood estimation* (MLE): we choose $\boldsymbol{\phi}$ so that the training examples are collectively assigned high model density[^2].

A standard way to formalize this idea is through the (forward) Kullback–Leibler divergence[^3]

$$
\begin{aligned}
\mathcal{D}_{\mathrm{KL}}
\big(p_{\mathrm{data}}\|p_{\boldsymbol{\phi}}\big)
:=\;&
\int
p_{\mathrm{data}}({\mathbf{x}})
\log
\frac{p_{\mathrm{data}}({\mathbf{x}})}
     {p_{\boldsymbol{\phi}}({\mathbf{x}})}
\,\mathop{}\!\mathrm{d} {\mathbf{x}}
\\
=\;&
\mathbb{E}_{{\mathbf{x}}\sim p_{\mathrm{data}}}
\big[
\log p_{\mathrm{data}}({\mathbf{x}})
-
\log p_{\boldsymbol{\phi}}({\mathbf{x}})
\big].
\end{aligned}
$$

The KL divergence is asymmetric, i.e.,

$$
\mathcal{D}_{\mathrm{KL}}
\big(p_{\mathrm{data}}\|p_{\boldsymbol{\phi}}\big)
\neq
\mathcal{D}_{\mathrm{KL}}
\big(p_{\boldsymbol{\phi}}\|p_{\mathrm{data}}\big).
$$

At first, the forward KL may appear unusable because the data density $p_{\mathrm{data}}({\mathbf{x}})$ is unknown. However, it can be decomposed as

$$
\begin{aligned}
\mathcal{D}_{\mathrm{KL}}
\big(p_{\mathrm{data}}\|p_{\boldsymbol{\phi}}\big)
=
\mathbb{E}_{{\mathbf{x}}\sim p_{\mathrm{data}}}
\left[
\log
\frac{p_{\mathrm{data}}({\mathbf{x}})}
     {p_{\boldsymbol{\phi}}({\mathbf{x}})}
\right]
=
-\mathbb{E}_{{\mathbf{x}}\sim p_{\mathrm{data}}}
\big[
\log p_{\boldsymbol{\phi}}({\mathbf{x}})
\big]
-
\mathcal{H}\big(p_{\mathrm{data}}\big),
\end{aligned}
$$

where

$$
\mathcal{H}\big(p_{\mathrm{data}}\big)
:=
-\mathbb{E}_{{\mathbf{x}}\sim p_{\mathrm{data}}}
\big[
\log p_{\mathrm{data}}({\mathbf{x}})
\big]
$$

is the entropy of the data distribution. Crucially, this term does not depend on $\boldsymbol{\phi}$. Therefore, although we cannot evaluate $p_{\mathrm{data}}({\mathbf{x}})$ itself, we do not need to know it in order to optimize the model parameters. This gives the following equivalence:

<span id="mle-kl"></span>

**Lemma 1.1.1: Minimizing KL $\Leftrightarrow$ MLE**

<span id="eq-mle"></span>

$$
\begin{aligned}
\min_{\boldsymbol{\phi}}\,
\mathcal{D}_{\mathrm{KL}}
\big(
p_{\mathrm{data}}
\,\|\,
p_{\boldsymbol{\phi}}
\big)
\;\Longleftrightarrow\;
\max_{\boldsymbol{\phi}}\,
\mathbb{E}_{{\mathbf{x}}\sim p_{\mathrm{data}}}
\big[
\log p_{\boldsymbol{\phi}}({\mathbf{x}})
\big].
\end{aligned}
\tag{1.1.2}
$$

In words, minimizing the forward KL divergence is equivalent to making real data samples receive high log-likelihood under the model. This is exactly maximum likelihood estimation.

In practice, the expectation over $p_{\mathrm{data}}$ is replaced by the available training samples $\{{\mathbf{x}}^{(i)}\}_{i=1}^{N}$, assumed to be drawn i.i.d. from $p_{\mathrm{data}}$. This gives the empirical negative log-likelihood objective

$$
\begin{aligned}
\hat{\mathcal{L}}_{\mathrm{MLE}}(\boldsymbol{\phi})
:=
-\frac{1}{N}
\sum_{i=1}^{N}
\log p_{\boldsymbol{\phi}}\big({\mathbf{x}}^{(i)}\big),
\end{aligned}
$$

which can be optimized using stochastic gradients over minibatches. Thus, MLE requires only samples from $p_{\mathrm{data}}$; it does not require evaluating the unknown data density itself.

The direction of the KL divergence matters because it determines *where the discrepancy is averaged*. For the forward KL,

$$
\mathcal{D}_{\mathrm{KL}}
\big(p_{\mathrm{data}}\|p_{\boldsymbol{\phi}}\big)
=
\mathbb{E}_{{\mathbf{x}}\sim p_{\mathrm{data}}}
\left[
\log
\frac{p_{\mathrm{data}}({\mathbf{x}})}
     {p_{\boldsymbol{\phi}}({\mathbf{x}})}
\right],
$$

the expectation is taken over the data distribution. Thus, regions that occur under $p_{\mathrm{data}}$ must be accounted for by the model. By contrast, the reverse KL, $\mathcal{D}_{\mathrm{KL}} (p_{\boldsymbol{\phi}}\|p_{\mathrm{data}})$, averages instead over samples from the model distribution.

This difference has an important consequence. Suppose there exists a region $A$ with positive data probability, $p_{\mathrm{data}}(A)>0$, but the model assigns zero density there, $p_{\boldsymbol{\phi}}({\mathbf{x}})=0$ for ${\mathbf{x}}\in A$. Then the forward KL becomes infinite, since

$$
\log
\frac{p_{\mathrm{data}}({\mathbf{x}})}
     {p_{\boldsymbol{\phi}}({\mathbf{x}})}
=
+\infty
$$

on that region. Therefore, minimizing the forward KL strongly penalizes a model for completely missing regions where the data distribution places probability mass. This behavior is commonly referred to as *mode covering*.

**Fisher Divergence.** The Fisher divergence is another important concept for (score-based) diffusion modeling (see [Chapter 3](../03-score-based-perspective/index.md#ch-score-based)). For two distributions $p$ and $q$, it is defined as

<span id="eq-fisher"></span>

$$
\begin{aligned}
\mathcal D_{\mathrm F}(p \|  q)
:=
\mathbb{E}_{\mathbf{x}\sim p} \left[
\left\|\nabla_{\mathbf x}\log p(\mathbf x)-\nabla_{\mathbf x}\log q(\mathbf x)\right\|_2^{2}
\right].
\end{aligned}
\tag{1.1.3}
$$

It measures the discrepancy between the *score functions* $\nabla_{\mathbf x}\log p(\mathbf x)$ and $\nabla_{\mathbf x}\log q(\mathbf x)$, which are vector fields pointing toward regions of higher probability. In short, $\mathcal D_{\mathrm F}(p \| q)\ge 0$ with equality if and only if $p=q$ almost everywhere. It is invariant to normalization constants, since scores depend only on gradients of log-densities, and it forms the basis of *score matching* ([Equation (3.1.3)](../03-score-based-perspective/index.md#eq-ebm-sm) and [Equation (3.2.1)](../03-score-based-perspective/index.md#eq-sm)): a method that learns the gradient of the log-density for generation (score-based models). In this setting, the data distribution $p=p_{\mathrm{data}}$ serves as the target, while the model $q=p_{\boldsymbol{\phi}}$ is trained to align its score field with that of the data.

**Beyond KL.** Although the KL divergence is the most widely used measure of difference between probability distributions, it is not the only one. Different divergences capture different geometric or statistical notions of discrepancy, which in turn affect the optimization dynamics of learning algorithms. A broad family is the *$f$-divergences* ([Csiszár, 1963](references.md#ref-csiszar1963informationstheoretische)):

<span id="eq-f-div"></span>

$$
\begin{aligned}
\mathcal{D}_f(p\|q)
=\int q({\mathbf{x}}) f \left(\frac{p({\mathbf{x}})}{q({\mathbf{x}})}\right)\mathop{}\!\mathrm{d} {\mathbf{x}},
\qquad f(1)=0,
\end{aligned}
\tag{1.1.4}
$$

where $f:\mathbb{R}_+ \to\mathbb{R}$ is a convex function. By changing $f$, we obtain many well-known divergences:

$$
\begin{aligned}
f(u)&=u\log u &&\Rightarrow&& \mathcal{D}_f=\mathcal{D}_{\mathrm{KL}}(p\|q)\quad\text{(forward KL)},\\
f(u)&=\tfrac12 \left[u\log u-(u+1)\log \tfrac{1+u}{2}\right] &&\Rightarrow&& \mathcal{D}_f=\mathcal{D}_{\mathrm{JS}}(p\|q)\quad\text{(Jensen--Shannon)},\\
f(u)&=\tfrac12|u-1| &&\Rightarrow&& \mathcal{D}_f=\mathcal{D}_{\mathrm{TV}}(p,q)\quad\text{(total variation)}.
\end{aligned}
$$

For clarity, the explicit forms are

$$
\mathcal{D}_{\mathrm{JS}}(p\|q)=\tfrac12 \mathcal{D}_{\mathrm{KL}}\big(p\| \tfrac12(p+q)\big)+\tfrac12 \mathcal{D}_{\mathrm{KL}}\big(q\| \tfrac12(p+q)\big),
$$

and

$$
\mathcal{D}_{\mathrm{TV}}(p,q)=\tfrac12 \int_{\mathbb{R}^D
} |p-q|\mathop{}\!\mathrm{d} {\mathbf{x}}
=\sup_{A\subset \mathbb{R}^D } |p(A)-q(A)|.
$$

Intuitively, the JS divergence provides a smooth and symmetric measure that balances both distributions and avoids the unbounded penalties of KL (we will later see that it helps interpret the Generative Adversarial Network (GAN) framework), while the total variation distance captures the largest possible probability difference between the two.

A different viewpoint comes from *optimal transport* (see [Chapter 7](../07-optimal-transport/index.md#ch-ot-eot)), whose representative is the Wasserstein distance (see [Equation (7.2.2)](../07-optimal-transport/index.md#eq-monge-formulation)). It measures the minimal cost of moving probability mass from one distribution to another. Unlike $f$-divergences, which compare density ratios, Wasserstein distances depend on the geometry of the sample space and remain meaningful even when the supports of $p$ and $q$ do not overlap.

Each divergence embodies a different notion of closeness between distributions and thus induces distinct learning behavior. We will revisit these divergences when they arise naturally in the context of generative modeling throughout this monograph.

## 1.1.2 Challenges in Modeling Distributions {#challenges-in-modeling-distributions}

To model a complex data distribution, we can parameterize the probability density function $p_{\mathrm{data}}$ using a neural network with parameters $\boldsymbol{\phi}$, creating a model we denote as $p_{\boldsymbol{\phi}}$. For $p_{\boldsymbol{\phi}}$ to be a valid probability density function, it must satisfy two fundamental properties:

-   **(i)** **Non-Negativity:** $p_{\boldsymbol{\phi}}({\mathbf{x}}) \ge 0$ for all ${\mathbf{x}}$ in the domain.

-   **(ii)** **Normalization:** The integral over the entire domain must equal one, i.e., $\int p_{\boldsymbol{\phi}}({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} = 1$.

A network can naturally produce a real scalar $E_{{\boldsymbol{\phi}}}({\mathbf{x}}) \in {\mathbb{R}}$ for input ${\mathbf{x}}$. To interpret this output as a valid density, it must be transformed to satisfy conditions (i) and (ii). A practical alternative is to view $E_{{\boldsymbol{\phi}}}\colon\mathbb{R}^D\to\mathbb{R}$ as defining an *unnormalized* density and then enforce these properties explicitly.

**Step 1: Ensuring Non-Negativity.** We can guarantee that our model’s output is always non-negative by applying a positive function to the raw output of the neural network $E_{\boldsymbol{\phi}}({\mathbf{x}})$, such as $\left|E_{\boldsymbol{\phi}}({\mathbf{x}})\right|$, $E^2_{\boldsymbol{\phi}}({\mathbf{x}})$. A standard and convenient choice is the exponential function. This gives us an unnormalized density, $\tilde{p}_{\boldsymbol{\phi}}({\mathbf{x}})$, that is guaranteed to be positive:

$$
\tilde{p}_{\boldsymbol{\phi}}({\mathbf{x}}) = \exp(E_{\boldsymbol{\phi}}({\mathbf{x}})).
$$

**Step 2: Enforcing Normalization.** The function $\tilde{p}_{\boldsymbol{\phi}}({\mathbf{x}})$ is positive but does not integrate to one. To create a valid probability density, we must divide it by its integral over the entire space. This leads to the final form of our model:

$$
p_{\boldsymbol{\phi}}({\mathbf{x}}) = \frac{\tilde{p}_{\boldsymbol{\phi}}({\mathbf{x}})}{\int \tilde{p}_{\boldsymbol{\phi}}({\mathbf{x}}') \mathop{}\!\mathrm{d} {\mathbf{x}}'} = \frac{\exp(E_{\boldsymbol{\phi}}({\mathbf{x}}))}{\int \exp(E_{\boldsymbol{\phi}}({\mathbf{x}}')) \mathop{}\!\mathrm{d} {\mathbf{x}}'}.
$$

The denominator in this expression is known as the *normalizing constant* or *partition function*, denoted by $Z(\boldsymbol{\phi})$:

$$
Z(\boldsymbol{\phi}) := \int \exp(E_{\boldsymbol{\phi}}({\mathbf{x}}')) \mathop{}\!\mathrm{d} {\mathbf{x}}'.
$$

While this procedure provides a valid construction for $p_{\boldsymbol{\phi}}({\mathbf{x}})$, it introduces a major computational challenge. For most high-dimensional problems, the integral required to compute the normalizing constant $Z(\boldsymbol{\phi})$ is intractable. This intractability is a central problem that motivates the development of many different families of deep generative models.

In the following sections, we introduce several prominent approaches of DGM. Each is designed to circumvent or reduce the computational cost of evaluating this normalizing constant.

[^1]: This is a common assumption in machine learning. For simplicity, we use the symbol $p$ to represent either a probability distribution or its probability density/mass function, depending on the context.

[^2]: For a fixed observed sample ${\mathbf{x}}$, the quantity $p_{\boldsymbol{\phi}}({\mathbf{x}})$, viewed as a function of the model parameters $\boldsymbol{\phi}$, is called the *likelihood*. For continuous data, $p_{\boldsymbol{\phi}}({\mathbf{x}})$ is a probability density rather than the probability of observing exactly ${\mathbf{x}}$.

[^3]: All integrals are in the Lebesgue sense and reduce to sums under counting measures.
