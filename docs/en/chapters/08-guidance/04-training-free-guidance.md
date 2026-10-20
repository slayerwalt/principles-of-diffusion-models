# 8.4 Training-Free Guidance {#sec-training-free-guidance}

Training-free guidance modifies the sampling dynamics of a fixed pre-trained diffusion model using an external condition, loss, or measurement objective, without retraining the diffusion model itself. The Bayes decomposition in [Equation (8.1.1)](01-prologue.md#eq-guidance-bayes) provides a useful motivation in score space: guidance adds a condition-dependent direction to the unconditional score. In practice, a task-specific loss can be used to construct corrections either to the predicted clean sample or to the score used during sampling.

We first introduce these two general forms in [Section 8.4.1](04-training-free-guidance.md#subsec-tfg). We then ask what distribution such surrogate-guided dynamics actually generate in [Section 8.4.2](04-training-free-guidance.md#subsec-guided-marginal-path), before turning to representative training-free approaches for inverse problems in [Section 8.4.3](04-training-free-guidance.md#subsec-inverse-problem).

**Setup and Notations.** Let ${\mathbf{c}}$ denote a conditioning variable. We assume access to a pre-trained diffusion model ${\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)$ expressed in score prediction[^1]. In addition, suppose we are given a task loss

$$
\ell(\cdot,{\mathbf{c}})\colon\mathbb{R}^D\to\mathbb{R}
$$

that measures disagreement with the desired condition, so that smaller $\ell({\mathbf{x}},{\mathbf{c}})$ indicates better agreement. Examples include a perceptual distance to a reference target, or a feature-space distance or negative similarity derived from a pre-trained representation such as CLIP ([Radford et al., 2021](references.md#ref-radford2021learning)).

Consider the standard linear–Gaussian forward noising kernel

$$
p_t(\cdot|{\mathbf{x}}_0)
:=
\mathcal N\bigl(\cdot;\alpha_t{\mathbf{x}}_0,\sigma_t^2{\mathbf{I}}\bigr).
$$

We borrow the DDIM update introduced later in [Equation (9.2.2)](../09-numerical-solvers/02-ddim.md#eq-ddim-update-all) as a concrete example:

<span id="eq-ddim-update-clean-noise-score"></span>

$$
\begin{aligned}
{\mathbf{x}}_{t\rightarrow t-1}
=
\alpha_{t-1}
\underbrace{\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)}_{\text{in data space}}
-
\sigma_{t-1}\sigma_t
\underbrace{\hat{{\mathbf{s}}}({\mathbf{x}}_t)}_{\text{in noise space}},
\end{aligned}
\tag{8.4.1}
$$

where $\hat{{\mathbf{x}}}_0({\mathbf{x}}_t):={\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)$ is the clean ${\mathbf{x}}$-prediction and $\hat{{\mathbf{s}}}({\mathbf{x}}_t):={\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)$ is the score prediction at noise level $t$.

## 8.4.1 Conceptual Framework for Training-Free Guidance {#subsec-tfg}

A common strategy is to use gradients of the task loss to modify either the predicted clean sample or the score term in the sampling update. To distinguish these two operations, define the data-space direction

$$
\mathcal G_t^{\mathrm{data}}({\mathbf{x}}_t,{\mathbf{c}})
:=
-
\left.
\nabla_{{\mathbf{x}}_0}\ell({\mathbf{x}}_0,{\mathbf{c}})
\right|_{{\mathbf{x}}_0=\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)}.
$$

For the noise-space correction, we reuse the guidance direction introduced in the Prologue,

$$
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})
:=
\nabla_{{\mathbf{x}}_t}\log{\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}}),
$$

for a positive guidance potential ${\mathbf{h}}_t$, specified below.

Using the DDIM update in [Equation (8.4.1)](04-training-free-guidance.md#eq-ddim-update-clean-noise-score), the two corrections can be written compactly as

<span id="eq-ddim-update-clean-noise-guide"></span>

$$
\begin{aligned}
{\mathbf{x}}_{t\rightarrow t-1}
=
\alpha_{t-1}
\underbrace{\left(
\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)
+
{\color{orange}
\eta_t^{\text{data}}
\mathcal G_t^{\text{data}}({\mathbf{x}}_t,{\mathbf{c}})}
\right)}_{\text{A. data space}}
-
\sigma_{t-1}\sigma_t
\underbrace{\left(
\hat{{\mathbf{s}}}({\mathbf{x}}_t)
+
{\color{orange}
\eta_t^{\text{latent}}
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})}
\right)}_{\text{B. noise space}},
\end{aligned}
\tag{8.4.2}
$$

where $\eta_t^{\text{data}},\eta_t^{\text{latent}}\geq0$ control the strengths of the data-space and noise-space corrections, respectively.

**A. Guidance in Data Space.** The data-space correction directly improves the current clean estimate by taking a local step that decreases the task loss:

$$
\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)
\;\longmapsto\;
\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)
+
\eta_t^{\text{data}}
\mathcal G_t^{\text{data}}({\mathbf{x}}_t,{\mathbf{c}}).
$$

Thus the clean prediction is first moved toward better agreement with the condition and then used in the subsequent sampling update. This correction can also be applied repeatedly when multiple optimization steps are desired.

Representative examples include MGPD ([He et al., 2023](references.md#ref-he2023manifold)) and UGD ([Bansal et al., 2023](references.md#ref-bansal2023universal)).

**B. Guidance in Noise Space.** A closely related strategy applies guidance directly to the score used during sampling. Following the potential view introduced in [Section 8.1](01-prologue.md#sec-prologue-guidance), a common construction evaluates the task loss on the current denoised estimate and chooses

$$
{\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
:=
\exp\!\left(
-\tau\,\ell\bigl(\hat{{\mathbf{x}}}_0({\mathbf{x}}_t),{\mathbf{c}}\bigr)
\right),
\qquad \tau>0.
$$

By the definition of $\mathcal G_t$ above, this gives

$$
\begin{aligned}
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})
=
\nabla_{{\mathbf{x}}_t}\log{\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
=
-\tau
\nabla_{{\mathbf{x}}_t}
\ell\bigl(\hat{{\mathbf{x}}}_0({\mathbf{x}}_t),{\mathbf{c}}\bigr).
\end{aligned}
$$

Accordingly, the score term in [Equation (8.4.2)](04-training-free-guidance.md#eq-ddim-update-clean-noise-guide) becomes

$$
\hat{{\mathbf{s}}}({\mathbf{x}}_t)
+
\eta_t^{\text{latent}}
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}}).
$$

Comparing this practical update with the population-level construction in [Equation (8.1.3)](01-prologue.md#eq-generic-guided-score), $\eta_t^{\text{latent}}$ plays the role of the generic guidance strength $w_t$, while $\hat{{\mathbf{s}}}({\mathbf{x}}_t)$ replaces the exact unconditional score by its learned estimate.

When ${\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})$ is proportional to the true conditional likelihood $p_t({\mathbf{c}}|{\mathbf{x}}_t)$ up to a factor independent of ${\mathbf{x}}_t$, $\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})$ recovers the Bayes guidance direction $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{c}}|{\mathbf{x}}_t)$. For a generic task loss, $\mathcal G_t$ instead defines a surrogate guidance direction.

Evaluating $\mathcal G_t$ requires differentiating through the ${\mathbf{x}}$-prediction:

$$
\mathcal G_t({\mathbf{x}}_t,{\mathbf{c}})
=
-\tau
\left[
\nabla_{{\mathbf{x}}_t}\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)
\right]^\top
\left.
\nabla_{{\mathbf{x}}_0}\ell({\mathbf{x}}_0,{\mathbf{c}})
\right|_{{\mathbf{x}}_0=\hat{{\mathbf{x}}}_0({\mathbf{x}}_t)},
$$

which may incur substantial computational cost in practice.

Representative examples include ([Yu et al., 2023](references.md#ref-yu2023freedom); [Chung et al., 2022](references.md#ref-chung2022diffusion); [Bansal et al., 2023](references.md#ref-bansal2023universal)).

The constructions above specify how guidance modifies each sampling step. They do not, by themselves, determine the marginal distributions obtained after these modified dynamics are integrated across all noise levels. We clarify this distinction next.

## 8.4.2 What Distribution Does Guided Sampling Produce? {#subsec-guided-marginal-path}

The potential view above has an exact fixed-time density interpretation at the population level. In particular, replacing the learned unconditional score estimate $\hat{{\mathbf{s}}}$ by the exact score, a positive potential ${\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}})$ and guidance strength $w_t\geq0$ define the ideal guided field

$$
\nabla_{{\mathbf{x}}}\log p_t({\mathbf{x}})
+
w_t\nabla_{{\mathbf{x}}}\log{\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}}).
$$

At each fixed $t$, this is the score of

<span id="eq-guided-fixed-time-tilt"></span>

$$
\begin{aligned}
\widetilde p_t^{\mathrm{tilt}}({\mathbf{x}}|{\mathbf{c}})
=
\frac{1}{Z_t({\mathbf{c}})}
p_t({\mathbf{x}}){\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}})^{w_t},
\qquad
Z_t({\mathbf{c}})
=
\int
p_t({\mathbf{x}}){\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}})^{w_t}\,\mathop{}\!\mathrm{d}{\mathbf{x}},
\end{aligned}
\tag{8.4.3}
$$

provided $0<Z_t({\mathbf{c}})<\infty$. The learned field used in practice approximates this ideal construction through the learned score estimate $\hat{{\mathbf{s}}}({\mathbf{x}}_t)$.

For the loss-based construction above,

$$
{\mathbf{h}}_t({\mathbf{x}}_t;{\mathbf{c}})
=
\exp\!\left[
-\tau\ell\bigl(\hat{{\mathbf{x}}}_0({\mathbf{x}}_t),{\mathbf{c}}\bigr)
\right],
$$

and $\eta_t^{\mathrm{latent}}$ plays the role of $w_t$.

This fixed-time identity does not determine the distribution generated after the guided dynamics are integrated across noise levels. Consequently, for a generic loss and guidance schedule, solving the guided PF-ODE from $T$ to $0$ does not in general produce samples distributed according to $\widetilde p_0^{\mathrm{tilt}}(\cdot|{\mathbf{c}})$.

The reason is cross-time consistency: densities at different noise levels cannot be chosen independently. For $\{\widetilde p_t^{\mathrm{tilt}}\}_t$ to form the marginal path associated with the same forward diffusion, they must also satisfy the corresponding Fokker-Planck equation:

$$
\partial_t \widetilde p_t^{\mathrm{tilt}}
+
\nabla_{{\mathbf{x}}} \cdot
\bigl(f(t){\mathbf{x}} \widetilde p_t^{\mathrm{tilt}}\bigr)
-
\frac{1}{2}g^2(t)
\Delta_{{\mathbf{x}}}\widetilde p_t^{\mathrm{tilt}}
=
0.
$$

When this condition holds, the corresponding PF-ODE

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t}
=
f(t){\mathbf{x}}(t)
-
\frac{1}{2}g^2(t)
\nabla_{{\mathbf{x}}}\log
\widetilde p_t^{\mathrm{tilt}}\left({\mathbf{x}}(t)|{\mathbf{c}}\right)
$$

has $\widetilde p_t^{\mathrm{tilt}}$ as its marginal path when initialized from the corresponding terminal distribution $\widetilde p_T^{\mathrm{tilt}}(\cdot|{\mathbf{c}})$.

The compatibility condition can be checked directly. Substituting [Equation (8.4.3)](04-training-free-guidance.md#eq-guided-fixed-time-tilt) into the Fokker-Planck equation and using the fact that $p_t$ itself satisfies the same equation gives

$$
\begin{aligned}
&
\partial_t \widetilde p_t^{\mathrm{tilt}}
+
\nabla_{{\mathbf{x}}} \cdot
\bigl(f(t){\mathbf{x}} \widetilde p_t^{\mathrm{tilt}}\bigr)
-
\frac{1}{2}g^2(t)
\Delta_{{\mathbf{x}}}\widetilde p_t^{\mathrm{tilt}}
\\
&=
\widetilde p_t^{\mathrm{tilt}}
\Bigg[
\partial_t
\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
+
f(t){\mathbf{x}}\cdot
\nabla_{{\mathbf{x}}}\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
-
g^2(t)
\nabla_{{\mathbf{x}}}\log p_t
\cdot
\nabla_{{\mathbf{x}}}\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
\\[-2pt]
&\hspace{3.4cm}
-
\frac{1}{2}g^2(t)
\left(
\Delta_{{\mathbf{x}}}\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
+
\left\|
\nabla_{{\mathbf{x}}}\log \bigl({\mathbf{h}}_t^{ w_t}\bigr)
\right\|^2
\right)
-
\frac{Z_t'}{Z_t}
\Bigg],
\end{aligned}
$$

where the arguments $({\mathbf{x}};{\mathbf{c}})$ are suppressed for readability. Thus, the tilted family is a valid marginal path only when the bracketed quantity vanishes. A generic task loss and guidance schedule need not satisfy this condition. The mismatch can therefore remain even with exact score evaluation and exact ODE integration: it comes from the guided dynamics themselves rather than from numerical discretization. This is an instance of the broader Fokker-Planck consistency requirement discussed in ([Lai et al., 2023](references.md#ref-lai2023fp)).

Two simple cases clarify the distinction. When $w_t\equiv0$, the tilt vanishes and the original unconditional path $p_t$ is recovered. For exact classifier guidance, under the standard setting where the forward diffusion corrupts ${\mathbf{x}}$ while leaving ${\mathbf{c}}$ fixed,

$$
{\mathbf{h}}_t({\mathbf{x}};{\mathbf{c}})
=
p_t({\mathbf{c}}|{\mathbf{x}}),
\qquad
w_t\equiv1.
$$

Bayes’ rule then gives

$$
\widetilde p_t^{\mathrm{tilt}}({\mathbf{x}}|{\mathbf{c}})
=
p_t({\mathbf{x}}|{\mathbf{c}}),
$$

so the tilted family is exactly the conditional marginal path. This explains the special role of guidance weight $1$ for the oracle classifier likelihood.

For a general guidance strength, for example

$$
p_t({\mathbf{x}}) p_t({\mathbf{c}}|{\mathbf{x}})^\omega,
$$

the expression still defines a valid fixed-time tilted density whenever it is normalizable, but it is not generally guaranteed to describe the marginals of the guided sampler. Special choices may satisfy the required compatibility condition, so $\omega\neq1$ alone does not imply inconsistency.

Correction-based approaches such as Feynman-Kac Correctors ([Skreta et al., 2025](references.md#ref-skreta2025feynman)) explicitly account for this cross-time mismatch when the goal is to follow a prescribed sequence of tilted distributions more faithfully.

## 8.4.3 Examples of Training-Free Approaches to Inverse Problems {#subsec-inverse-problem}

The principle introduced in [Section 8.4.1](04-training-free-guidance.md#subsec-tfg) has important applications in inverse problems. We begin with an overview of the background and then provide several concrete examples illustrating how to leverage pre-trained diffusion models for inference-time inverse problem solving.

**Background on Inverse Problems.** Let $\mathcal A$ denote a known forward operator, possibly nonlinear, and suppose that

<span id="eq-inverse-problem"></span>

$$
\begin{aligned}
{\mathbf{y}}
=
\mathcal A({\mathbf{x}}_0)
+
\sigma_{{\mathbf{y}}}{\mathbf{z}},
\qquad
{\mathbf{z}}\sim\mathcal N(\boldsymbol 0,{\mathbf{I}}).
\end{aligned}
\tag{8.4.4}
$$

The Bayesian objective is to characterize the posterior $p_0({\mathbf{x}}_0|{\mathbf{y}})$, which combines the measurement model with prior knowledge about plausible clean signals. Blind inverse problems extend this setting by treating unknown parameters of $\mathcal A$ as additional quantities to infer.

Classical inverse-problem methods commonly combine a data-fidelity term with an analytical or hand-designed prior in a variational or Bayesian formulation. Supervised restoration methods may instead learn a direct mapping from measurements to clean signals using paired data. Pre-trained diffusion models provide a third route: a learned prior over clean data can be reused across measurement operators, with measurement information introduced only at inference time.

**Pre-Trained Diffusion Models as Inverse Problem Solvers.** For the measurement model above, Bayes’ rule gives

<span id="eq-bayes-rule"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t|{\mathbf{y}})
=
\underbrace{\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)}_{\text{data score}}
+
\underbrace{\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{y}}|{\mathbf{x}}_t)}_{\text{measurement guidance}}.
\end{aligned}
\tag{8.4.5}
$$

The data score can be supplied by a pre-trained diffusion model, whereas the measurement term is generally unavailable in closed form. Training-free inverse solvers therefore introduce measurement information during sampling, but they need not all do so in the same mathematical form.

A useful distinction is between two common strategies. Some methods directly enforce measurement consistency by modifying or projecting the intermediate sample. Others approximate the likelihood score $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{y}}|{\mathbf{x}}_t)$ and add it to the data score through [Equation (8.4.5)](04-training-free-guidance.md#eq-bayes-rule).

**ILVR ([Choi et al., 2021](references.md#ref-choi2021ilvr)).** ILVR provides a simple example of the first strategy. Let $\phi_N$ denote its linear low-pass operator. At each reverse step, one first samples an unconditional proposal

$$
{\mathbf{x}}_{t-1}'
\sim
p_{\boldsymbol\theta}({\mathbf{x}}_{t-1}'|{\mathbf{x}}_t)
$$

and noises the reference image to the same level,

$$
{\mathbf{y}}_{t-1}
\sim
p_{t-1}({\mathbf{y}}_{t-1}|{\mathbf{y}}).
$$

The proposal is then refined according to

$$
{\mathbf{x}}_{t-1}
=
\phi_N({\mathbf{y}}_{t-1})
+
({\mathbf{I}}-\phi_N){\mathbf{x}}_{t-1}'.
$$

Thus, the prescribed low-frequency component comes from the noised reference, while the complementary component comes from the unconditional diffusion proposal. This is a sample-space consistency correction rather than an explicit approximation of $\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{y}}|{\mathbf{x}}_t)$.

**Diffusion Posterior Sampling (DPS) ([Chung et al., 2022](references.md#ref-chung2022diffusion)).** For a known nonlinear forward operator $\mathcal A$ with additive Gaussian measurement noise $\sigma_{{\mathbf{y}}}>0$, *Diffusion Posterior Sampling* (DPS) approximates

<span id="eq-dps-score"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{y}}|{\mathbf{x}}_t)
\approx
\nabla_{{\mathbf{x}}_t}
\log p\bigl({\mathbf{y}}|X_0=\hat{\mathbf{x}}_0({\mathbf{x}}_t)\bigr),
\end{aligned}
\tag{8.4.6}
$$

where $\hat{\mathbf{x}}_0({\mathbf{x}}_t):={\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_t]$ is the posterior mean of the clean sample given ${\mathbf{x}}_t$, typically estimated from the pre-trained diffusion model using Tweedie’s formula ([Equation (3.3.6)](../03-score-based-perspective/03-denoising-score-matching.md#eq-tweedie)).

This plug-in approximation replaces the posterior average of the measurement likelihood by its value at the posterior mean:

$$
\begin{aligned}
p_t({\mathbf{y}}|{\mathbf{x}}_t)
  &= \int p_t({\mathbf{y}}|{\mathbf{x}}_t, {\mathbf{x}}_0)  p({\mathbf{x}}_0|{\mathbf{x}}_t)  \mathrm{d}{\mathbf{x}}_0 \\
  &= \int p_t({\mathbf{y}}|{\mathbf{x}}_0)  p({\mathbf{x}}_0|{\mathbf{x}}_t)  \mathrm{d}{\mathbf{x}}_0
   \approx  p_t\bigl({\mathbf{y}}|X_0=\hat{\mathbf{x}}_0({\mathbf{x}}_t)\bigr),
\end{aligned}
$$

where we have used that ${\mathbf{y}}$ depends only on ${\mathbf{x}}_0$ (not on ${\mathbf{x}}_t$) given ${\mathbf{x}}_0$, and the approximation holds under the assumption that the posterior $p({\mathbf{x}}_0|{\mathbf{x}}_t)$ is tightly peaked around its mean.

Since

$$
p_t\bigl({\mathbf{y}}|X_0 = \hat{\mathbf{x}}_0({\mathbf{x}}_t)\bigr)
  = \mathcal{N}\bigl({\mathbf{y}};  \mathcal{A}(\hat{\mathbf{x}}_0({\mathbf{x}}_t)),  \sigma_{{\mathbf{y}}}^2 {\mathbf{I}}\bigr),
$$

we compute

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{y}}|{\mathbf{x}}_t)
  &\approx \nabla_{{\mathbf{x}}_t} \log \mathcal{N}\bigl({\mathbf{y}};  \mathcal{A}(\hat{\mathbf{x}}_0),  \sigma_{{\mathbf{y}}}^2 {\mathbf{I}}\bigr) \\
  &= -\frac{1}{2\sigma_{{\mathbf{y}}}^2}  \nabla_{{\mathbf{x}}_t} \bigl\| {\mathbf{y}} - \mathcal{A}(\hat{\mathbf{x}}_0) \bigr\|^2 \quad  \\
  &= \frac{1}{\sigma_{{\mathbf{y}}}^2}  \left[ \mathcal{J}_{\mathcal{A}}\big(\hat{\mathbf{x}}_0({\mathbf{x}}_t)\big) \cdot \nabla_{{\mathbf{x}}_t} \hat{\mathbf{x}}_0({\mathbf{x}}_t) \right]^\top \bigl( {\mathbf{y}} - \mathcal{A}\big(\hat{\mathbf{x}}_0({\mathbf{x}}_t)\big) \bigr),
\end{aligned}
$$

where $\mathcal{J}_{\mathcal{A}}\big(\hat{\mathbf{x}}_0({\mathbf{x}}_t)\big) := \nabla_{{\mathbf{x}}_0} \mathcal{A}({\mathbf{x}}) \big|_{{\mathbf{x}} = \hat{\mathbf{x}}_0({\mathbf{x}}_t)}$ denotes the Jacobian of the forward operator with respect to its input. This formula propagates the gradient through the score approximation pipeline, reflecting how the measurement likelihood changes with respect to perturbations in the noisy sample ${\mathbf{x}}_t$.

For linear inverse problems, this further simplifies to:

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{y}} | {\mathbf{x}}_t) \approx \bbox[2px,#d6e5c9]{\color{black}{\frac{1}{\sigma_{{\mathbf{y}}}^2}}} \,\,    \bbox[2px,#e7cfea]{\color{black}{\left[ {\mathbf{A}} \cdot \nabla_{{\mathbf{x}}_t} \hat{\mathbf{x}}_0({\mathbf{x}}_t) \right]^\top}} \left(\bbox[2px,#d5dcf2]{\color{black}{{\mathbf{y}} - {\mathbf{A}}(\hat{\mathbf{x}}_0({\mathbf{x}}_t))}}\right).
\end{aligned}
$$

A large body of work explores diffusion-based inverse problem solvers by proposing various approximations for $\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{y}} | {\mathbf{x}}_t)$. For a comprehensive overview, we refer readers to the survey by [Daras et al. (2024)](references.md#ref-daras2024survey).

**A Related Supervised Iterative-Restoration Viewpoint and Its Connection to the PF-ODE.** The discussion above solves inverse problems by combining a pre-trained diffusion prior with the Bayes-rule decomposition in [Equation (8.4.5)](04-training-free-guidance.md#eq-bayes-rule). More specifically, these methods use the score of the prior distribution together with the measurement model to guide the reconstruction process. At the same time, they also reflect a broader idea: instead of recovering the clean signal in one step, they gradually improve a corrupted input through many small updates. A similar high-level strategy also appears in supervised image restoration, but from a different starting point. Rather than using a pre-trained score model as a prior, these methods learn the refinement procedure directly from paired corrupted and clean examples.

A representative example is InDI ([Delbracio and Milanfar, 2023](references.md#ref-delbracio2023inversion)). Unlike the Bayes-rule-based approaches above, InDI begins with paired data $({\mathbf{y}},{\mathbf{x}}_0)$, where ${\mathbf{y}}$ is a corrupted observation and ${\mathbf{x}}_0$ is the corresponding clean target. Importantly, it does not assume in general that the corruption from ${\mathbf{x}}_0$ to ${\mathbf{y}}$ is known analytically or that it is Gaussian.

The basic idea is to connect the two endpoints by the interpolation path

$$
{\mathbf{x}}_t=(1-t){\mathbf{x}}_0+t{\mathbf{y}},\qquad t\in[0,1].
$$

This path should be understood only as a convenient trajectory from the clean target to the corrupted observation. It is not a Gaussian noising process in general. Along this path, for any $0\le s\le t$, one has the exact relation

$$
{\mathbf{x}}_s=\Bigl(1-\frac{s}{t}\Bigr){\mathbf{x}}_0+\frac{s}{t}{\mathbf{x}}_t.
$$

Taking conditional expectation given ${\mathbf{x}}_t$ yields

$$
\mathbb E[{\mathbf{x}}_s|{\mathbf{x}}_t]
=
\Bigl(1-\frac{s}{t}\Bigr)\mathbb E[{\mathbf{x}}_0|{\mathbf{x}}_t]
+
\frac{s}{t}{\mathbf{x}}_t.
$$

This shows that, to move from the current state ${\mathbf{x}}_t$ to a slightly less degraded state, it is enough to predict the clean target from ${\mathbf{x}}_t$.

Based on this idea, one trains a time-conditioned predictor ${\mathbf{F}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)$ from paired examples by minimizing the regression objective

$$
\min_{{\boldsymbol{\phi}}}\;
\mathbb E_{({\mathbf{x}}_0,{\mathbf{y}})\sim p({\mathbf{x}}_0,{\mathbf{y}}),  t\sim p(t)}
\Bigl\|
{\mathbf{F}}_{{\boldsymbol{\phi}}}\bigl((1-t){\mathbf{x}}_0+t{\mathbf{y}}, t\bigr)-{\mathbf{x}}_0
\Bigr\|_2^2,
$$

where $p({\mathbf{x}}_0,{\mathbf{y}})$ denotes the joint distribution of data pairs consisting of a clean target ${\mathbf{x}}_0$ and its corrupted observation ${\mathbf{y}}$, and $p(t)$ specifies how the interpolation time is sampled during training. That is, ${\mathbf{F}}_{{\boldsymbol{\phi}}}$ is trained to predict the clean target from an intermediate degraded state along the interpolation path. For this squared-error loss, the Bayes-optimal predictor is the conditional expectation

$$
{\mathbf{F}}^*({\mathbf{x}}_t,t)=\mathbb E[{\mathbf{x}}_0|{\mathbf{x}}_t].
$$

Therefore, if the model class is sufficiently expressive and the optimization is successful, the learned predictor approximates the average clean target associated with the intermediate state ${\mathbf{x}}_t$ at time $t$. In this sense, the training objective is similar in spirit to diffusion-model training: in both cases, one learns a time-conditioned regression from an intermediate corrupted state to a cleaner target. The difference is that diffusion models generate such training pairs synthetically from a prescribed forward corruption process, whereas here the endpoint pairs $({\mathbf{y}},{\mathbf{x}}_0)$ come directly from supervised restoration data.

At inference time, one starts from $\hat{{\mathbf{x}}}_1={\mathbf{y}}$ and repeatedly refines the estimate by

$$
\hat{{\mathbf{x}}}_{t-\Delta t}
=
\frac{\Delta t}{t}{\mathbf{F}}_{{\boldsymbol{\phi}}}(\hat{{\mathbf{x}}}_t,t)
+
\Bigl(1-\frac{\Delta t}{t}\Bigr)\hat{{\mathbf{x}}}_t.
$$

Hence, the final reconstruction is produced by gradual refinement rather than by a single direct prediction. In the continuous-time limit, this leads to InDI’s residual-flow ODE

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{{\mathbf{x}}_t-{\mathbf{F}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)}{t}.
$$

Now suppose we further specialize to the Gaussian denoising setting, where the corruption is assumed to take the known analytic form

$$
{\mathbf{y}}={\mathbf{x}}_0+\sigma{\boldsymbol{\epsilon}},
\qquad {\boldsymbol{\epsilon}}\sim\mathcal N(\boldsymbol 0,{\mathbf{I}}),
$$

with known noise level $\sigma$. Under this specialization, InDI’s residual-flow ODE reduces exactly to the probability-flow ODE. Indeed, the interpolation path becomes

$$
{\mathbf{x}}_t=(1-t){\mathbf{x}}_0+t{\mathbf{y}}
=
{\mathbf{x}}_0+t\sigma{\boldsymbol{\epsilon}},
$$

which is precisely the standard diffusion form

$$
{\mathbf{x}}_t=\alpha_t{\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}},
\qquad\text{with}\qquad
\alpha_t=1,\;\sigma_t=t\sigma.
$$

Therefore, the predictor ${\mathbf{F}}_{{\boldsymbol{\phi}}}({\mathbf{x}}_t,t)\approx \mathbb E[{\mathbf{x}}_0\mid {\mathbf{x}}_t]$ is now learning to denoise a Gaussian-corrupted intermediate state ${\mathbf{x}}_t$. In this case, the residual-flow ODE becomes

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}_t}{\mathop{}\!\mathrm{d} t}
=
\frac{{\mathbf{x}}_t-\mathbb E[{\mathbf{x}}_0\mid {\mathbf{x}}_t]}{t}
=
-\dot{\sigma}_t\sigma_t\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t),
$$

where the last equality follows from $\sigma_t=t\sigma$ together with Tweedie’s formula,

$$
\mathbb E[{\mathbf{x}}_0\mid {\mathbf{x}}_t]
=
{\mathbf{x}}_t+\sigma_t^2\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t).
$$

This is exactly the PF-ODE for this Gaussian denoising path, written in either denoiser form or score form. Thus, under the Gaussian denoising specialization, InDI suggests an alternative interpretation of diffusion modeling: one may view it as supervised denoiser learning on Gaussian-corrupted data, with the PF-ODE arising as the continuous-time limit of gradual refinement.

[^1]: Here, we adopt the score and ${\mathbf{x}}$-prediction parameterizations for simplicity; other parameterizations such as ${\boldsymbol{\epsilon}}$-prediction can be handled analogously.
