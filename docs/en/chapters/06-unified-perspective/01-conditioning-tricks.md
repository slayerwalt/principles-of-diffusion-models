# 6.1 Conditional Tricks: The Secret Sauce of Diffusion Models {#sec-conditional-trick}

Until now, we have explored diffusion models from three seemingly distinct origins: variational, score-based, and flow based perspectives. Each was originally motivated by different goals and led to its own training objectives (with a fixed $t$):

-   **Variational View:** Learn a parametrized density $p_{\boldsymbol{\phi}}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)$ to approximate the oracle reverse transition $p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)$ by minimizing:

    $$
    \mathcal{J}_{\text{KL}}(\boldsymbol{\phi}) := \mathbb{E}_{p_t(\mathbf{x}_t)}\left[
            \mathcal{D}_{\mathrm{KL}}\big(p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) \| p_{\boldsymbol{\phi}}(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t)\big)
        \right];
    $$

-   **Score-Based View:** Learn a score model ${\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$ to approximate the marginal score $\nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t)$ via:

    $$
    \mathcal{J}_{\text{SM}}(\boldsymbol{\phi}) := \mathbb{E}_{p_t({\mathbf{x}}_t)}\left[
           \left\| {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}} \log p_t({\mathbf{x}}_t) \right\|_2^2
        \right];
    $$

-   **Flow-Based View:** Learn a velocity model ${\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t)$ to match the oracle velocity ${\mathbf{v}}_t({\mathbf{x}}_t)$ (e.g., defined by [Equation (5.2.9)](../05-flow-based-perspective/02-flow-matching-framework.md#eq-marginal-oracle-velocity)) by minimizing:

    $$
    \mathcal{J}_{\text{FM}}(\boldsymbol{\phi}) := \mathbb{E}_{p_t({\mathbf{x}}_t)} \left[
            \left\| {\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - {\mathbf{v}}_t({\mathbf{x}}_t) \right\|_2^2
        \right].
    $$

At first glance, these objectives seem hopelessly intractable, since they all require access to oracle quantities that are fundamentally unknowable in general. But here comes the exciting twist: each method independently arrives at the same elegant solution to this problem: *conditioning on the data ${\mathbf{x}}_0$*. This technique transforms each intractable training target into a tractable one.

This elegant “conditioning technique” rewrites the objectives as expectations over the known Gaussian conditionals $p_t({\mathbf{x}}_t | {\mathbf{x}}_0)$, yielding tractable conditional targets and training objectives that are gradient-equivalent to their marginal counterparts:

-   **Variational View** ([Equation (2.2.3)](../02-variational-perspective/02-ddpm.md#eq-kl-matching))**:**

    $$
    \mathcal{J}_{\text{KL}}(\boldsymbol{\phi}) = \underbrace{\mathbb{E}_{{\mathbf{x}}_0} \mathbb{E}_{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)} \left[
            \mathcal{D}_{\mathrm{KL}}\big(p({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t, {\mathbf{x}}_0) \| p_{\boldsymbol{\phi}}({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t)\big)
        \right]}_{\mathcal{J}_{\text{CKL}}(\boldsymbol{\phi})} + C;
    $$

-   **Score-Based View** ([Equation (3.3.3)](../03-score-based-perspective/03-denoising-score-matching.md#eq-score-matching))**:**

    $$
    \mathcal{J}_{\text{SM}}(\boldsymbol{\phi}) = \underbrace{\mathbb{E}_{{\mathbf{x}}_0} \mathbb{E}_{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)} \left[
            \left\| {\mathbf{s}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t | {\mathbf{x}}_0) \right\|_2^2
        \right]}_{\mathcal{J}_{\text{DSM}}(\boldsymbol{\phi})} + C;
    $$

-   **Flow-Based View** ([Equation (5.2.10)](../05-flow-based-perspective/02-flow-matching-framework.md#eq-fm-cfm))**:**

    $$
    \mathcal{J}_{\text{FM}}(\boldsymbol{\phi}) = \underbrace{\mathbb{E}_{{\mathbf{x}}_0} \mathbb{E}_{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)} \left[
            \left\| {\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - {\mathbf{v}}_t({\mathbf{x}}_t | {\mathbf{x}}_0) \right\|^2
        \right]}_{\mathcal{J}_{\text{CFM}}(\boldsymbol{\phi})} + C.
    $$

To build a unified view, we next revisit the conditional KL, score, and velocity objectives in a systematic manner. Crucially, these objectives are not only tractable but also equivalent to their original forms up to a constant vertical shift. The conditional versions ($\mathcal{J}_{\text{CKL}}$, $\mathcal{J}_{\text{DSM}}$, $\mathcal{J}_{\text{CFM}}$) differ from the originals ($\mathcal{J}_{\text{KL}}$, $\mathcal{J}_{\text{SM}}$, $\mathcal{J}_{\text{FM}}$) only by a constant that is independent of the model parameters. Consequently, each conditional objective has the same minimizers as its original counterpart within the same model class.

To see what these minimizers ideally recover, suppose the model is expressive enough to represent the exact oracle. Despite their different objectives, all three perspectives then share the same conceptual pattern:

$$
\underbrace{\text{marginal oracle}}_{\text{generally intractable}}
=
\mathbb{E}_{{\mathbf{x}}_0\sim p(\cdot|{\mathbf{x}}_t)}
\left[
\underbrace{\text{conditional oracle}}_{\text{tractable given }{\mathbf{x}}_0}
\right].
$$

That is, conditioning on the clean sample ${\mathbf{x}}_0$ gives a tractable target, and averaging these conditional targets over the possible clean samples consistent with ${\mathbf{x}}_t$ recovers the desired marginal oracle. Concretely,

<span id="eq-three-minimizer-oracle"></span>

$$
\begin{aligned}
        p^*(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t, \mathbf{x}_0)\big] &&= p(\mathbf{x}_{t-\Delta t}|\mathbf{x}_t), \\
    \mathbf{s}^*(\mathbf{x}_t, t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0)\big] &&= \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t), \\
    \mathbf{v}^*(\mathbf{x}_t, t) &= \mathbb{E}_{\mathbf{x}_0 \sim p(\cdot|\mathbf{x}_t)} \big[\mathbf{v}_t(\mathbf{x}_t|\mathbf{x}_0)\big] &&= \mathbf{v}_t(\mathbf{x}_t).
\end{aligned}
\tag{6.1.1}
$$

For the variational view, the first identity follows from marginalizing the conditional reverse transition over ${\mathbf{x}}_0$, and the KL objective is minimized by the resulting marginal reverse transition. For score and velocity matching, the squared-error objectives are least-squares regressions whose population minimizers are the corresponding conditional expectations. These identities describe the unrestricted population oracles; a restricted neural model need not represent them exactly. Below is a concrete example of a scenario with a finite number of data points.

**Example: Closed-Form Oracle Targets on a Finite Dataset**

To make [Equation (6.1.1)](01-conditioning-tricks.md#eq-three-minimizer-oracle) more concrete, suppose the data distribution is the empirical measure

$$
\hat p_0=\frac{1}{N}\sum_{i=1}^N \delta_{{\mathbf{x}}_0^{(i)}},
$$

where $\{{\mathbf{x}}_0^{(i)}\}_{i=1}^N$ denotes the finite training set.

In [Equation (6.1.1)](01-conditioning-tricks.md#eq-three-minimizer-oracle), the notation

$$
{\mathbf{x}}_0 \sim p(\cdot|{\mathbf{x}}_t)
$$

means that we average over the posterior distribution of the clean sample given the noisy observation ${\mathbf{x}}_t$. Equivalently, for any quantity ${\mathbf{h}}({\mathbf{x}}_0)$,

$$
\mathbb{E}_{{\mathbf{x}}_0 \sim p(\cdot|{\mathbf{x}}_t)}[{\mathbf{h}}({\mathbf{x}}_0)]
=
\int {\mathbf{h}}({\mathbf{x}}_0)\, p({\mathbf{x}}_0|{\mathbf{x}}_t)\mathop{}\!\mathrm{d}{\mathbf{x}}_0.
$$

By Bayes’ rule,

$$
p({\mathbf{x}}_0|{\mathbf{x}}_t)
=
\frac{p_t({\mathbf{x}}_t|{\mathbf{x}}_0)\,p_{\text{data}}({\mathbf{x}}_0)}
{\int p_t({\mathbf{x}}_t|\tilde{{\mathbf{x}}}_0)\,p_{\text{data}}(\tilde{{\mathbf{x}}}_0)\mathop{}\!\mathrm{d}\tilde{{\mathbf{x}}}_0}.
$$

Under the standard Gaussian diffusion forward process,

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)})
=
\mathcal{N}\left({\mathbf{x}}_t;\alpha_t {\mathbf{x}}_0^{(i)}, \sigma_t^2 {\mathbf{I}}\right),
$$

so the conditional likelihood is explicit. However, at the population level, the posterior $p({\mathbf{x}}_0|{\mathbf{x}}_t)$ still requires averaging over the full data distribution $p_{\text{data}}$.

If we now replace $p_{\text{data}}$ by the empirical measure

$$
\hat p_0=\frac{1}{N}\sum_{i=1}^N \delta_{{\mathbf{x}}_0^{(i)}},
$$

then the posterior $p({\mathbf{x}}_0|{\mathbf{x}}_t)$ is supported only on the training examples, and its mass at ${\mathbf{x}}_0^{(i)}$ is

$$
p({\mathbf{x}}_0={\mathbf{x}}_0^{(i)}|{\mathbf{x}}_t)
=
\frac{p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)})}
{\sum_{j=1}^N p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(j)})}
=: w_i({\mathbf{x}}_t,t),
\qquad
\sum_{i=1}^N w_i({\mathbf{x}}_t,t)=1.
$$

Hence the posterior expectation above reduces to the finite sum

$$
\mathbb{E}_{{\mathbf{x}}_0 \sim p(\cdot|{\mathbf{x}}_t)}[{\mathbf{h}}({\mathbf{x}}_0)]
=
\sum_{i=1}^N w_i({\mathbf{x}}_t,t){\mathbf{h}}({\mathbf{x}}_0^{(i)}).
$$

Thus, in the finite-dataset case, the abstract posterior averaging in [Equation (6.1.1)](01-conditioning-tricks.md#eq-three-minimizer-oracle) becomes a concrete weighted average over the training set, with weights $w_i({\mathbf{x}}_t,t)$. We now make these three cases explicit.

**Variational View.** The oracle reverse transition becomes

$$
p^*({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t)
=
\sum_{i=1}^N
w_i({\mathbf{x}}_t,t)
p({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t,{\mathbf{x}}_0^{(i)}).
$$

Here each conditional reverse kernel

$$
p({\mathbf{x}}_{t-\Delta t}|{\mathbf{x}}_t,{\mathbf{x}}_0^{(i)})
$$

is itself available in closed form under Gaussian diffusion; in fact, it is Gaussian as shown in [Lemma 2.2.2](../02-variational-perspective/02-ddpm.md#ddpm-reverse-kernel). So the true reverse kernel is simply a posterior-weighted mixture of explicit Gaussian bridges.

**Score-Based View.** Likewise, the oracle score becomes

$$
\mathbf{s}^*({\mathbf{x}}_t,t)
=
\sum_{i=1}^N
w_i({\mathbf{x}}_t,t)
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)}).
$$

The conditional score is explicit:

$$
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)})
=
-\frac{{\mathbf{x}}_t-\alpha_t{\mathbf{x}}_0^{(i)}}{\sigma_t^2}.
$$

Therefore,

$$
\mathbf{s}^*({\mathbf{x}}_t,t)
=
-\frac{1}{\sigma_t^2}
\left(
{\mathbf{x}}_t-\alpha_t\sum_{i=1}^N w_i({\mathbf{x}}_t,t) {\mathbf{x}}_0^{(i)}
\right).
$$

So the marginal score points from the noisy sample toward a posterior-weighted average of training examples.

**Flow-Based View.** The same phenomenon appears for flow matching:

$$
\mathbf{v}^*({\mathbf{x}}_t,t)
=
\sum_{i=1}^N
w_i({\mathbf{x}}_t,t)
\mathbf{v}_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)}).
$$

Again, the building blocks are explicit. The corresponding conditional velocity field has closed form:

$$
\mathbf{v}_t({\mathbf{x}}_t|{\mathbf{x}}_0^{(i)})
=
{\alpha}'_t {\mathbf{x}}_0^{(i)}
+
\frac{{\sigma}'_t}{\sigma_t}\bigl({\mathbf{x}}_t-\alpha_t{\mathbf{x}}_0^{(i)}\bigr).
$$

Hence the oracle velocity is also an explicit posterior-weighted average over the dataset.

Taken together, these formulas show that, on a finite dataset, all three oracle targets reduce to explicit posterior-weighted averages of closed-form conditional quantities.

**Remark on Memorization.** The formulas above characterize the exact oracle targets induced by the empirical distribution $\hat p_0$, rather than by the unknown population distribution $p_{\mathrm{data}}$. If the model learned these empirical targets perfectly at every noise level, and if the reverse process were computed without error, then generation would exactly reproduce the empirical distribution $\hat p_0$. Since $\hat p_0$ assigns probability only to the training examples, the generated samples would be drawn from the training set itself. In this idealized sense, perfect fitting of the empirical oracle amounts to memorization.

However, a practically trained diffusion model, need not realize this oracle exactly: it is learned with finite model capacity and finite optimization, and its generated distribution is further affected by model approximation and numerical sampling errors. Therefore, the finite-dataset calculation establishes memorization as an exact empirical solution, but does not imply that every practically trained model necessarily memorizes its training set.

The common conditional reformulation is no coincidence: by making training tractable, it reveals a profound unification. Variational diffusion, score-based SDEs, and flow matching are simply different facets of the same principle. Three perspectives, one insight, elegantly connected. We will continue to explore their equivalence throughout the rest of this chapter.
