# 6.2 A Roadmap for Elucidating Training Losses in Diffusion Models {#sec-elucidating}

This section builds a systematic view of training losses in diffusion models. In [Section 6.2.1](02-training-objectives.md#subsec-four-predictions), we extend the standard three objectives to a broader set of four parameterizations, showing how they arise from different modeling perspectives. In [Section 6.2.2](02-training-objectives.md#subsec-summary-disentagle), we then distill these results into a general framework that disentangles the structure of diffusion objectives, laying the groundwork for the equivalence results in [Section 6.3](03-equivalent-parameterizations.md#sec-equivalent-parametrizations).

## 6.2.1 Four Common Parameterizations in Diffusion Models {#subsec-four-predictions}

Throughout this section, we consider the forward perturbation kernel

$$
p_t(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t; \alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I} \right),
$$

where $\mathbf{x}_0 \sim p_{\mathrm{data}}$, as defined in [Equation (4.5.1)](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#eq-forward-kernel), unless stated otherwise.

Let $\omega: [0, T] \to \mathbb{R}_{>0}$ denote a positive time-weighting function. The four standard parameterizations (noise $\boldsymbol{\epsilon}_{\boldsymbol{\phi}}$, clean ${\mathbf{x}}_{\boldsymbol{\phi}}$, score ${\mathbf{s}}_{\boldsymbol{\phi}}$, and velocity ${\mathbf{v}}_{\boldsymbol{\phi}}$), together with their respective minimizers $\boldsymbol{\epsilon}^*$, ${\mathbf{x}}^*$, ${\mathbf{s}}^*$, and ${\mathbf{v}}^*$, are summarized below for clarity and to facilitate further discussion.

**Variational View.** Based on the KL divergence in DDPMs (see [Section 2.2.4](../02-variational-perspective/02-ddpm.md#subsec-ddpm-prediction) and [Section 4.5.2](../04-score-sde-framework/05-accumulated-perturbation-kernels.md#subsec-connection-vdm)), this approach reduces to predicting either the expected noise that produces $\mathbf{x}_t$ or the expected clean signal that $\mathbf{x}_t$ was perturbed from.

1.  **${\boldsymbol{\epsilon}}$-Prediction (Noise Prediction)** ([Ho et al., 2020](references.md#ref-ho2020denoising)):

    <span id="eq-noise-def"></span>

    $$
    \boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) \approx \mathbb{E}[\boldsymbol{\epsilon}|\mathbf{x}_t]=\boldsymbol{\epsilon}^*({\mathbf{x}}_t, t)
    \tag{6.2.1}
    $$

    with training objective

    $$
    \mathcal{L}_{\text{noise}}(\boldsymbol{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \boldsymbol{\epsilon}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) - \boldsymbol{\epsilon} \right\|_2^2 \right].
    $$

    Here, ${\boldsymbol{\epsilon}}^*$ means the average noise that was injected to obtain the given ${\mathbf{x}}_t$.

2.  **${\mathbf{x}}$-Prediction (Clean Prediction)** ([Kingma et al., 2021](references.md#ref-kingma2021variational); [Karras et al., 2022](references.md#ref-karras2022elucidating); [Song et al., 2023](references.md#ref-song2023consistency)):

    <span id="eq-clean-def"></span>

    $$
    \mathbf{x}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) \approx \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t]={\mathbf{x}}^*({\mathbf{x}}_t, t)
    \tag{6.2.2}
    $$

    with training objective

    $$
    \mathcal{L}_{\text{clean}}(\boldsymbol{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \mathbf{x}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) - \mathbf{x}_0 \right\|_2^2 \right].
    $$

    Here, ${\mathbf{x}}^*$ means the average of all plausible clean guesses, given the noisy observation ${\mathbf{x}}_t$.

**Score-Based View.** Predicts the score function at noise level $t$, which points in the average direction to denoise $\mathbf{x}_t$ back toward all possible clean samples that could have generated it:

3.  **Score Prediction** ([Song and Ermon, 2019](references.md#ref-song2019generative); [Song et al., 2020](references.md#ref-song2020score)):

    <span id="eq-score-def"></span>

    $$
    \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \mathbb{E}\left[\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) |\mathbf{x}_t\right] = {\mathbf{s}}^*({\mathbf{x}}_t, t)
    \tag{6.2.3}
    $$

    with training objective

    $$
    \mathcal{L}_{\text{score}}(\boldsymbol{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \mathbf{s}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) - \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) \right\|_2^2 \right],
    $$

    where the conditional score satisfies $\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t|\mathbf{x}_0) = -\frac{1}{\sigma_t} \boldsymbol{\epsilon}$.

**Flow-Based View.** Predicts the instantaneous average velocity of the data as it evolves through $\mathbf{x}_t$:

4.  **${\mathbf{v}}$-Prediction (Velocity Prediction)** ([Lipman et al., 2022](references.md#ref-lipman2022flow); [Liu, 2022](references.md#ref-liu2022rectified); [Salimans and Ho, 2021](references.md#ref-salimans2021progressive); [Albergo et al., 2023](references.md#ref-albergo2023stochastic)):

    <span id="eq-velocity-def"></span>

    $$
    \mathbf{v}_{\boldsymbol{\phi}}(\mathbf{x}_t, t) \approx \mathbb{E}\left[\left. \frac{\mathop{}\!\mathrm{d}  \mathbf{x}_t}{\mathop{}\!\mathrm{d}  t} \right| {\mathbf{x}_t}\right] ={\mathbf{v}}^*({\mathbf{x}}_t, t)
    \tag{6.2.4}
    $$

    with training objective

    $$
    \mathcal{L}_{\text{velocity}}(\boldsymbol{\phi}) := \mathbb{E}_t \left[\omega(t) \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left\| \mathbf{v}_{\boldsymbol{\phi}}(\mathbf{x}_t,t) - {\mathbf{v}}_t({\mathbf{x}}_t | {\mathbf{x}}_0,\boldsymbol{\epsilon}) \right\|_2^2 \right],
    $$

    where the conditional velocity is ${\mathbf{v}}_t({\mathbf{x}}_t|{\mathbf{x}}_0,\boldsymbol{\epsilon}) = \alpha_t' \mathbf{x}_0 + \sigma_t' \boldsymbol{\epsilon}$.

    Here, ${\mathbf{v}}^*$ indicates the average velocity vector passing through the observation point ${\mathbf{x}}_t$.

Building on the insight from [Equation (6.1.1)](01-conditioning-tricks.md#eq-three-minimizer-oracle), all four prediction types ultimately aim to approximate a conditional expectation in the form of the average noise, clean data, score, or velocity given an observed $\mathbf{x}_t$.

## 6.2.2 Disentangling the Training Objective of Diffusion Models {#subsec-summary-disentagle}

As shown in [Section 6.2.1](02-training-objectives.md#subsec-four-predictions), the objective functions for the four prediction types commonly share the following template form for diffusion model training:

<span id="eq-general-dm-loss-d"></span>

$$
\begin{aligned}
   \mathcal{L}(\boldsymbol{\phi}):=\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}} \underbrace{\bbox[2px,#d5dcf2]{\color{black}{\mathbb{E}_{p_{\text{time}}(t)}}}}_{\substack{\text{time} \\  \text{distribution}}}\Big[ \underbrace{\bbox[2px,#d6e5c9]{\color{black}{\omega(t)}}}_{\substack{\text{time} \\ \text{weighting}}}\underbrace{\left\lVert \bbox[2px,#e7cfea]{\color{black}{\mathrm{NN}_{\boldsymbol{\phi}}}}(\bbox[2px,#ffe7e4]{\color{black}{{\mathbf{x}}_t}}, t) - \bbox[2px,#e7cfea]{\color{black}{(A_t {\mathbf{x}}_0 + B_t \boldsymbol{\epsilon}) }}\right\rVert_2^2}_{\text{MSE part}}\Big].
\end{aligned}
\tag{6.2.5}
$$

Here, to enhance training efficiency and optimize the diffusion model learning pipeline, several key design choices are crucial ([Karras et al., 2022](references.md#ref-karras2022elucidating); [Lu and Song, 2024](references.md#ref-lu2024simplifying)):

-   **(A)** Noise schedule in the forward process of $\bbox[2px,#ffe7e4]{\color{black}{{\mathbf{x}}_t}}$ via $\alpha_t$ and $\sigma_t$;

-   **(B)** Prediction types of $\bbox[2px,#e7cfea]{\color{black}{\mathrm{NN}_{\boldsymbol{\phi}}}}$ and their associated regression targets $\bbox[2px,#e7cfea]{\color{black}{(A_t {\mathbf{x}}_0 + B_t \boldsymbol{\epsilon}) }}$;

-   **(C)** Time-weighting function $\bbox[2px,#d6e5c9]{\color{black}{\omega(\cdot)}} \colon [0,T] \to \mathbb{R}_{\geq 0}$;

-   **(D)** Time distribution $\bbox[2px,#d5dcf2]{\color{black}{p_{\text{time}}}}$.

We elaborate on these four components here to serve as a roadmap for the discussions in the following sections.

**(A) Noise Schedule $\alpha_t$ and $\sigma_t$.** Users have the flexibility to choose schedules tailored to their applications, with common examples summarized in [Table 5.2](../05-flow-based-perspective/03-probability-paths-and-velocities.md#tb-interpolant-instances). Importantly, as we will demonstrate in [Equation (6.3.3)](03-equivalent-parameterizations.md#eq-fm-general-affine-formula) and [Equation (6.3.5)](03-equivalent-parameterizations.md#eq-any-to-trig), all affine flows of the form $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$ are mathematically equivalent. Specifically, any such interpolation can be converted to the canonical linear schedule ($\alpha_t = 1 - t$, $\sigma_t = t$) or to a trigonometric schedule ($\alpha_t = \cos t$, $\sigma_t = \sin t$) by appropriate time reparametrization and spatial rescaling.

**(B) Parameterization $\mathrm{NN}_{\boldsymbol{\phi}}$ and Training Target $A_t {\mathbf{x}}_0 + B_t \boldsymbol{\epsilon}$.**

<span id="tb-a-t-b-t-parametrizations"></span>

**Table 6.1:** Summary of the Relationships Between Different Parameterizations. All four parameterizations are mathematically equivalent and can be converted into one another through straightforward algebraic transformations.

<div style="overflow-x: auto;" markdown="1">
<table markdown="1">
<tr markdown="1">
<th markdown="block">

Regression Target =

</th>
<th colspan="2" markdown="block">

$A_t{\mathbf{x}}_0 + B_t \boldsymbol{\epsilon}$

</th>
</tr>
<tr markdown="1">
<td markdown="block">



</td>
<td markdown="block">

$A_t$

</td>
<td markdown="block">

$B_t$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

Clean

</td>
<td markdown="block">

$1$

</td>
<td markdown="block">

$0$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

Noise

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

$1$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

Conditional Score

</td>
<td markdown="block">

$0$

</td>
<td markdown="block">

-$\frac{1}{\sigma_t}$

</td>
</tr>
<tr markdown="1">
<td markdown="block">

Conditional Velocity

</td>
<td markdown="block">

$\alpha_t'$

</td>
<td markdown="block">

$\sigma_t'$

</td>
</tr>
</table>
</div>

There are two choices that are often coupled in diffusion training: *what quantity the network directly predicts* and *which quantity is compared in the training loss*. Clean data, noise, score, and velocity can be used for either purpose.

**Matched Prediction and Loss.** The common choice, used in [Equation (6.2.5)](02-training-objectives.md#eq-general-dm-loss-d), is to train the network directly in the same prediction type that it outputs. Each prediction type then has a corresponding conditional training target of the form

$$
\text{Regression Target}
=
A_t{\mathbf{x}}_0+B_t{\boldsymbol{\epsilon}},
$$

where $A_t$ and $B_t$ depend on the prediction type and the forward schedule $(\alpha_t,\sigma_t)$. These targets are summarized in [Table 6.1](02-training-objectives.md#tb-a-t-b-t-parametrizations).

For example, an ${\mathbf{x}}$-prediction network is compared with the clean target ${\mathbf{x}}_0$, whereas a ${\mathbf{v}}$-prediction network is compared with the conditional velocity

$$
{\mathbf{v}}_t
=
\alpha_t'{\mathbf{x}}_0+\sigma_t'{\boldsymbol{\epsilon}}.
$$

As shown later in [Equation (6.3.1)](03-equivalent-parameterizations.md#eq-predictions-equivalence), the four prediction types encode the same oracle information and can be converted into one another.

**Predicting One Quantity and Evaluating the Loss in Another.** The network output and the quantity used in the loss can also be chosen separately. The network prediction is first converted using the relations in [Equation (6.3.1)](03-equivalent-parameterizations.md#eq-predictions-equivalence), and the loss is then computed against the corresponding target in the converted prediction type.

For example, under the canonical schedule

$$
{\mathbf{x}}_t=(1-t){\mathbf{x}}_0+t{\boldsymbol{\epsilon}},
$$

suppose the network directly predicts clean data ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)$. Converting this prediction to velocity gives

$$
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
=
\frac{{\mathbf{x}}_t-{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)}{t},
$$

while the conditional velocity target satisfies

$$
{\mathbf{v}}_t
=
\frac{{\mathbf{x}}_t-{\mathbf{x}}_0}{t}.
$$

Therefore,

$$
\left\|
{\mathbf{v}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)-{\mathbf{v}}_t
\right\|_2^2
=
\frac{1}{t^2}
\left\|
{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)-{\mathbf{x}}_0
\right\|_2^2.
$$

Thus, with the network output fixed as clean prediction, evaluating a velocity loss is exactly equivalent to applying an additional time-dependent weight to the clean-prediction loss. The corresponding relations for general affine schedules and other prediction types are summarized later in [Equation (6.3.6)](03-equivalent-parameterizations.md#eq-s-e-x-v-equi).

Changing the quantity predicted directly by the network has an additional effect. Although the oracle predictions are algebraically convertible, a finite neural network is then trained to represent a different function, with different output scales and parameter gradients. The choice of network prediction can therefore interact with model architecture and optimization, while the choice of loss also controls the relative weighting of prediction errors across noise levels.

For clarity, [Equation (6.2.5)](02-training-objectives.md#eq-general-dm-loss-d) uses the matched prediction-and-loss case unless stated otherwise.

**(C) Time Distribution $p_{\text{time}}(t)$.** We next consider how time $t$, and hence the noise level, is sampled during training. For convenience, define the fixed-time regression loss

$$
\mathcal{L}_t(\boldsymbol{\phi})
:=
\mathbb{E}_{{\mathbf{x}}_0,{\boldsymbol{\epsilon}}}
\left[
\left\|
\mathrm{NN}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)
-
\bigl(A_t{\mathbf{x}}_0+B_t{\boldsymbol{\epsilon}}\bigr)
\right\|_2^2
\right],
\qquad
{\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}}.
$$

Then [Equation (6.2.5)](02-training-objectives.md#eq-general-dm-loss-d) can be written compactly as

$$
\mathcal{L}(\boldsymbol{\phi})
=
\mathbb{E}_{t\sim p_{\text{time}}}
\left[
\omega(t)\mathcal{L}_t(\boldsymbol{\phi})
\right].
$$

Thus, if no correction is applied, sampling some times more frequently also gives them more weight in the objective. At the level of the exact expected loss, only the product $p_{\text{time}}(t)\omega(t)$ matters.

A different use of nonuniform sampling is *importance sampling*. Suppose the objective above is fixed, but we instead draw times from a proposal $q_{\text{time}}(t)$. Then

$$
\mathcal{L}(\boldsymbol{\phi})
=
\mathbb{E}_{t\sim q_{\text{time}}}
\left[
\frac{p_{\text{time}}(t)}{q_{\text{time}}(t)}
\omega(t)\mathcal{L}_t(\boldsymbol{\phi})
\right],
$$

provided $q_{\text{time}}(t)>0$ wherever $p_{\text{time}}(t)\omega(t)>0$. With this correction, changing $q_{\text{time}}$ leaves the objective unchanged and changes only its Monte Carlo estimator, for example its variance.

The key distinction is therefore:

<div style="overflow-x: auto;" markdown="1">
<table markdown="1">
<tr markdown="1">
<td markdown="block">

change $p_{\text{time}}$ in the objective

</td>
<td markdown="block">

$\Longrightarrow$

</td>
<td markdown="block">

different weighting, different objective

</td>
</tr>
<tr markdown="1">
<td markdown="block">

change $q_{\text{time}}$ $+$ importance correction

</td>
<td markdown="block">

$\Longrightarrow$

</td>
<td markdown="block">

same objective, different estimator

</td>
</tr>
</table>
</div>

We therefore keep time sampling and loss weighting conceptually separate. Uniform time sampling makes this distinction less visible because its density contributes only a constant factor, whereas with nonuniform sampling one must specify whether the sampling distribution defines the objective or merely serves as an importance-sampling proposal.

Both uses appear in diffusion-model training. ScoreFlow explicitly uses importance sampling to reduce the variance of a fixed likelihood-weighted objective ([Song et al., 2021](references.md#ref-song2021maximum)). VDM and subsequent analyses similarly show that, when the objective is fixed in noise-level coordinates, the noise schedule can act as an importance-sampling distribution  ([Kingma et al., 2021](references.md#ref-kingma2021variational); [Kingma and Gao, 2023](references.md#ref-kingma2023understanding)). EDM ([Karras et al., 2022](references.md#ref-karras2022elucidating)), by contrast, specifies both a nonuniform training distribution over noise levels and a separate loss weighting, whose combination determines the effective training weight.

**(D) Time-Weighting Function $\omega(t)$.** For a fixed $p_{\text{time}}$, the weighting function $\omega(t)$ determines the relative contribution of different times to the training objective. Together with the previous paragraph, the effective weighting over time is therefore $p_{\text{time}}(t)\omega(t)$.

A simple baseline is uniform time sampling with $\omega(t)\equiv1$, as in the standard simplified ${\boldsymbol{\epsilon}}$-prediction and FM/RF objectives  ([Ho et al., 2020](references.md#ref-ho2020denoising); [Lipman et al., 2022](references.md#ref-lipman2022flow); [Liu, 2022](references.md#ref-liu2022rectified)). Other weightings emphasize different noise regimes and can also carry likelihood or variational interpretations. For example, for the score-matching objective, the likelihood weighting $\omega(t)=g^2(t)$ yields an upper bound on the negative log-likelihood under suitable regularity conditions  ([Song et al., 2021](references.md#ref-song2021maximum)). More generally, SNR-dependent weightings provide a common way to redistribute training emphasis across noise levels  ([Kingma et al., 2021](references.md#ref-kingma2021variational); [Kingma and Gao, 2023](references.md#ref-kingma2023understanding)).

**Monotonic Weighting and the ELBO-with-Augmentation View.** A useful way to describe the corruption level is through the *log signal-to-noise ratio* (log-SNR)

$$
\Lambda_t
:=
\log\frac{\alpha_t^2}{\sigma_t^2}.
$$

We use $\Lambda$ for the full log-SNR, distinguishing it from the half-log-SNR coordinate $\lambda_t=\log(\alpha_t/\sigma_t)=\tfrac12\Lambda_t$ used later in [Chapter 9](../09-numerical-solvers/index.md#ch-numerical-solvers). Since $\Lambda_t$ decreases as the forward process becomes noisier, after expressing the loss in a common prediction parameterization, let $w(\Lambda)$ denote the effective weight assigned to each log-SNR level.

A useful result of [Kingma and Gao (2023)](references.md#ref-kingma2023understanding) is that, if $w(\Lambda)$ is non-increasing in $\Lambda$ (equivalently, its weight does not decrease as the forward process becomes noisier), then the weighted diffusion objective can be rewritten, up to terms independent of the model parameters, as an average negative ELBO over Gaussian-perturbed data [^1].

Thus, monotonic weighting has a concrete probabilistic interpretation: rather than merely emphasizing some noise levels more than others, it can be viewed as averaging variational training over Gaussian-corrupted versions of the data at different corruption strengths.

**Conclusion.** For our purposes, the main lesson is that many apparent design choices in diffusion training can be understood through how they reshape the effective weighting over noise levels in the objective. This weighting in turn influences both the optimization landscape in practice and, in special cases such as monotonic weighting, the variational interpretation of the loss.

[^1]: To see why monotonicity matters, define the remaining variational cost from noise level $t$ onward as

    $$
    L(t;{\mathbf{x}}_0)
    :=
    \mathcal{D}_{\mathrm{KL}}
    \left(
    p({\mathbf{x}}_{[t,T]}|{\mathbf{x}}_0)
    \,\Vert\,
    p_{\boldsymbol{\phi}}({\mathbf{x}}_{[t,T]})
    \right),
    $$

    where ${\mathbf{x}}_{[t,T]}:=\{{\mathbf{x}}_s\}_{s\in[t,T]}$ denotes the remaining continuous diffusion path. This is the continuous-time counterpart of the negative ELBO in [Equation (2.2.13)](../02-variational-perspective/02-ddpm.md#eq-ddpm-elbo), starting from corruption level $t$; more precisely, it equals the expected negative ELBO of the perturbed sample ${\mathbf{x}}_t$, up to terms independent of $\boldsymbol{\phi}$.

    For ${\boldsymbol{\epsilon}}$-prediction, [Kingma and Gao (2023)](references.md#ref-kingma2023understanding) show that

    $$
    \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}L(t;{\mathbf{x}}_0)
    =
    \frac{1}{2}\frac{\mathop{}\!\mathrm{d}\Lambda_t}{\mathop{}\!\mathrm{d} t}
    {\mathbb{E}}_{{\boldsymbol{\epsilon}}}
    \left[
    \left\|
    {\boldsymbol{\epsilon}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)-{\boldsymbol{\epsilon}}
    \right\|_2^2
    \right].
    $$

    Hence the weighted diffusion loss is

    $$
    -\int_0^T
    w(\Lambda_t)
    \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}L(t;{\mathbf{x}}_0)\mathop{}\!\mathrm{d} t
    =
    w(\Lambda_0)L(0;{\mathbf{x}}_0)
    +
    \int_0^T
    \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}w(\Lambda_t)
    L(t;{\mathbf{x}}_0)\mathop{}\!\mathrm{d} t
    +
    C,
    $$

    where $C$ is independent of $\boldsymbol{\phi}$. If $w(\Lambda)$ is non-increasing in $\Lambda$, then $\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}w(\Lambda_t)\geq0$ because $\Lambda_t$ decreases with $t$. Thus, after normalization, the terms above form a nonnegative weighting over ELBOs at different Gaussian corruption levels, which gives the ELBO-with-Gaussian-augmentation interpretation.
