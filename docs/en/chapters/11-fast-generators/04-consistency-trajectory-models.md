# 11.4 General Flow Map: Consistency Trajectory Model {#sec-ctm}

Consistency Trajectory Model (CTM) ([Kim et al., 2024](references.md#ref-kim2023consistency)) is among the first methods to learn a *general* flow map ${\boldsymbol{\Psi}}_{s\to t}$.

**Setup of CTM in Practice.** Similar to the CM family, CTM originally follows the formulation of EDM ([Karras et al., 2022](references.md#ref-karras2022elucidating)) ([Section D.6](../../appendices/d-supplementary-proofs/index.md#sec-edm)), using the PF-ODE in ${\mathbf{x}}$-prediction form with the noise schedule $\alpha_t = 1$ and $\sigma_t = t$. Under this setup, the PF-ODE becomes

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(\tau)}{\mathop{}\!\mathrm{d} \tau}
    = \frac{{\mathbf{x}}(\tau) - {\mathbb{E}}[{\mathbf{x}}|{\mathbf{x}}(\tau)]}{\tau}.
\end{aligned}
$$

Starting from ${\mathbf{x}}_s$ at time $s$ and evolving backward to an ending time $t\le s$, the corresponding flow map (solution) can be written equivalently as

$$
\begin{aligned}
{\mathbf{x}}_s + \int_s^t
    \frac{{\mathbf{x}}_\tau - {\mathbb{E}}[{\mathbf{x}} |{\mathbf{x}}_\tau]}{\tau}\, \mathop{}\!\mathrm{d} \tau.
\end{aligned}
$$

CTM adopts an Euler-inspired parameterization: applying a single-step Euler solver to the PF-ODE[^1] yields

$$
{\mathbf{x}}_{s\to t}^{\mathrm{Euler}}
= {\mathbf{x}}_s - (s-t)\,\frac{{\mathbf{x}}_s - {\mathbb{E}}[{\mathbf{x}}|{\mathbf{x}}_s]}{s}
= \frac{t}{s}\,{\mathbf{x}}_s + \Big(1 - \frac{t}{s}\Big){\mathbb{E}}[{\mathbf{x}}|{\mathbf{x}}_s],
$$

where ${\mathbf{x}}_{s\to t}^{\mathrm{Euler}}$ approximates the solution at time $t$ given the state ${\mathbf{x}}_s$ at time $s$.

While the EDM setup provides a simple illustrative case, CTM allows broader noise schedules defined by an arbitrary linear Gaussian forward kernel $(\alpha_t, \sigma_t)$ and expresses the PF-ODE in ${\mathbf{v}}$-prediction form:

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
= {\mathbf{x}}_s + \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u.
$$

In the discussion that follows, we focus on this general formulation.

## 11.4.1 CTM Parametrization for Flexible Transition Learning {#ctm-parametrization}

Following the single-step Euler solver of the PF-ODE above, CTM rewrites the oracle flow map ${\boldsymbol{\Psi}}_{s\to t}$ as a convex combination of the input ${\mathbf{x}}_s$ and a residual function ${\mathbf{g}}^*$:

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
:= {\mathbf{x}}_s + \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u
= \frac{t}{s}\,{\mathbf{x}}_s
+ \frac{s-t}{s}
\underbrace{\Big[{\mathbf{x}}_s + \frac{s}{s-t} \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\Big]}_{=:~{\mathbf{g}}^*}.
\end{aligned}
$$

where the residual term $\mathbf{g}^*$ is defined as

<span id="eq-ctm-g-oracle"></span>

$$
\begin{aligned}
\mathbf{g}^*(\mathbf{x}_{s}, s, t) &:= {\mathbf{x}}_s + \frac{s}{s-t} \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\mathop{}\!\mathrm{d} u.
\end{aligned}
\tag{11.4.1}
$$

This motivates the neural parameterization

<span id="eq-ctm-para-1"></span>

$$
\begin{aligned}
{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t)
:= \frac{t}{s}\,{\mathbf{x}}_s + \frac{s-t}{s}\,{\mathbf{g}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t),
\end{aligned}
\tag{11.4.2}
$$

where ${\mathbf{g}}_{{\boldsymbol{\theta}}}$ is a neural network that aims at ${\mathbf{g}}_{{\boldsymbol{\theta}}} \approx {\mathbf{g}}^*$, hence ${\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t)$ is trained to approximate the oracle flow map,

$$
{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t) \approx {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s).
$$

Therefore, CTM naturally fits within the general consistency-mapping framework of [Equation (10.1.4)](01-prologue.md#eq-general-cm-loss-rep), which aligns the learned mapping with the oracle flow map.

Moreover, this formulation inherently satisfies the initial condition

$$
{\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,s)={\mathbf{x}}_s,
$$

without requiring any explicit enforcement during training.

**Advantages of CTM’s Parametrizations.** A crucial characteristic of ${\mathbf{g}}^*$ becomes evident when taking the limit as $t$ approaches $s$ (i.e., the same ending time as the starting time):

<span id="ctm-g-para"></span>

**Proposition 11.4.1: Properties of ${\mathbf{g}}^*$**

1.  **Recovering Diffusion Model:**

    $$
    {\mathbf{g}}^*({\mathbf{x}}_s, s, s)=\lim_{t\rightarrow s}\mathbf{g}^*(\mathbf{x}_{s},s,t)={\mathbf{x}}_s - s{\mathbf{v}}^*({\mathbf{x}}_s,s).
    $$

2.  **Integration Representation:**

    $$
    \mathbf{g}^*(\mathbf{x}_{s},s,t)={\mathbf{x}}_s - s{\mathbf{v}}^*({\mathbf{x}}_s,s)+\mathcal{O}(\vert t-s\vert).
    $$

**Proof for Proposition.**

From the definition of ${\mathbf{g}}^*$, we obtain

$$
\begin{aligned}
\lim_{t\rightarrow s}\mathbf{g}^*(\mathbf{x}_{s},s,t)=\mathbf{x}_{s}-s\lim_{t\rightarrow s}\frac{1}{t-s}\int_{s}^{t}{\mathbf{v}}^*(\mathbf{x}_{\tau}, \tau)\mathop{}\!\mathrm{d} \tau={\mathbf{x}}_s - s{\mathbf{v}}^*({\mathbf{x}}_s,s).
\end{aligned}
$$

This proved the first identity. For the second claim, from the Taylor expansion, we have

$$
\begin{aligned}
\mathbf{g}^*(\mathbf{x}_{s},s,t)&=\mathbf{x}_{s}-\frac{s}{s-t}\int_{t}^{s}{\mathbf{v}}^*({\mathbf{x}}_\tau, \tau)\mathop{}\!\mathrm{d} \tau\\
    &=\mathbf{x}_{s}-\frac{s}{s-t}\bigg[ (s-t){\mathbf{v}}^*({\mathbf{x}}_s, s) + \mathcal{O}((t-s)^{2}) \bigg]\\
    &={\mathbf{x}}_s - s{\mathbf{v}}^*({\mathbf{x}}_s,s)+\mathcal{O}(\vert t-s\vert).
\end{aligned}
$$

From this proposition, we can conclude that

1.  Estimating ${\mathbf{g}}^*$ enables approximating not only the finite $s$-to-$t$ transition (for $s \ge t$) but also the *infinitesimal* $s$-to-$s$ transition characterized by the instantaneous velocity ${\mathbf{v}}^*$.

2.  $\mathbf{g}^*(\mathbf{x}_{s},s,t)$ is interpreted as a denoiser-like quantity added with a residual term of the Taylor expansion.

Therefore, by leveraging CTM’s parameterization in [Equation (11.4.2)](04-consistency-trajectory-models.md#eq-ctm-para-1), learning $\mathbf{G}_{\boldsymbol{\theta}} \approx {\boldsymbol{\Psi}}_{s\to t}$ for $t<s$ corresponds, up to the nonzero factor $(s-t)/s$, to learning $\mathbf{g}_{\boldsymbol{\theta}} \approx \mathbf{g}^*$, enabling long-jump capability through $\mathbf{G}_{\boldsymbol{\theta}}$. Moreover, the diagonal value of $\mathbf{g}^*$ encodes the diffusion model’s instantaneous velocity through

$$
{\mathbf{v}}^*({\mathbf{x}}_s,s)
=
\frac{{\mathbf{x}}_s-{\mathbf{g}}^*({\mathbf{x}}_s,s,s)}{s},
$$

which can be directly supervised by the diffusion-model loss introduced below. Thus, CTM’s parameterization provides a single framework for learning both finite-time transitions and local diffusion dynamics, combining the long-jump capability of flow-map models with access to the instantaneous dynamics of diffusion models.

In the next two sections, we first present CTM’s consistency loss ([Section 11.4.2](04-consistency-trajectory-models.md#subsec-ctm-consistency)), which supports both distillation and training-from-scratch, and enforces the semigroup property to achieve $\mathbf{G}_{\boldsymbol{\theta}}(\cdot, s, t) \approx {\boldsymbol{\Psi}}_{s\to t}(\cdot, s, t)$. We then describe auxiliary losses ([Section 11.4.3](04-consistency-trajectory-models.md#subsec-ctm-auxiliary)) that arise naturally from the parametrization in [Equation (11.4.2)](04-consistency-trajectory-models.md#eq-ctm-para-1), including diffusion model loss and GAN loss, which further improve CTM’s performance significantly.

## 11.4.2 Consistency Loss in CTM {#subsec-ctm-consistency}

CTM aims to approximate the oracle solution map

$$
\mathbf{G}_{\boldsymbol{\theta}}(\cdot, s, t) \approx {\boldsymbol{\Psi}}_{s\to t}(\cdot, s, t),
$$

for any $s \ge t$. Since the oracle ${\boldsymbol{\Psi}}_{s\to t}$ is usually not available in closed form, CTM builds a feasible regression target by enforcing the *semigroup property* ([Equation (11.2.1)](02-discrete-consistency-models.md#eq-semi-group)): for any $s \ge u \ge t$,

$$
{\boldsymbol{\Psi}}_{u\to t}\circ {\boldsymbol{\Psi}}_{s\to u} = {\boldsymbol{\Psi}}_{s\to t}.
$$

Depending on whether a pre-trained diffusion model is available, the flow map ${\boldsymbol{\Psi}}_{s\to t}$ can be approximated in different ways. Throughout, we assume $s \ge u \ge t \in [0,T]$.

**Training via Distillation.** Assume access to a pre-trained diffusion model producing ${\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_s, s) \approx {\mathbf{v}}^*({\mathbf{x}}_s, s)$. Then the PF-ODE is approximated by the empirical dynamics

<span id="eq-ctm-teacher-pf-ode"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(\tau)}{\mathop{}\!\mathrm{d} \tau}
= {\mathbf{v}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_\tau, \tau).
\end{aligned}
\tag{11.4.3}
$$

CTM trains $\mathbf{G}_{\boldsymbol{\theta}}$ to match a numerical solver $\mathtt{Solver}_{s\to t}({\mathbf{x}}_s;\boldsymbol{\phi}^\times)$ applied to this empirical ODE, which serves as a computable proxy for the oracle:

$$
\mathbf{G}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s, t)
\approx \mathtt{Solver}_{s\to t}({\mathbf{x}}_s;\boldsymbol{\phi}^\times)
\approx {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s, s, t).
$$

With a strong teacher, the solver can recover ${\boldsymbol{\Psi}}_{s\to t}$ up to discretization error, so the optimal student closely matches the ground truth (see ([Kim et al., 2024](references.md#ref-kim2023consistency)), Propositions 3 and 4).

However, solving across the full interval $[t,s]$ during training loop can be costly when $s$ and $t$ are far apart. To improve efficiency and provide a smoother signal, CTM introduces *soft consistency matching*, which operationalizes the semigroup property. As illustrated in [Figure 11.5](04-consistency-trajectory-models.md#fig-ill-models), CTM compares two predictions at time $t$: the direct student output $\mathbf{G}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s, t)$, and a mixed teacher–student path that first advances the teacher from $s$ to a random $u \sim \mathcal{U}[t,s)$, then lets the student jump from $u$ to $t$:

$$
\mathbf{G}_{\boldsymbol{\theta}^-} \big(\mathtt{Solver}_{s\to u}({\mathbf{x}}_s;\boldsymbol{\phi}^\times),\, u,\, t\big).
$$

The student is trained to match this composite prediction:

<span id="eq-soft"></span>

$$
\begin{aligned}
\underbrace{\mathbf{G}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s, t)}_{\approx\, {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)}
 \approx
\underbrace{\mathbf{G}_{\boldsymbol{\theta}^-} \big(\mathtt{Solver}_{s\to u}({\mathbf{x}}_s;\boldsymbol{\phi}^\times), u, t\big)}_{\approx\, {\boldsymbol{\Psi}}_{u\to t}({\boldsymbol{\Psi}}_{s\to u}({\mathbf{x}}_s))},
\end{aligned}
\tag{11.4.4}
$$

where $\boldsymbol{\theta}^-$ is a stop gradient copy of $\mathbf{G}_{\boldsymbol{\theta}}$.

By varying $u$, CTM interpolates between global and local supervision:

-   **Global Consistency** ($u=t$): the teacher is solved over the full interval $s\to t$, so the target reduces to the full teacher transition at time $t$.

-   **Local Consistency** ($u=s-\Delta s$): the student learns from a short teacher step near $s$; when $t=0$, this reduces to consistency distillation.

To reinforce sample quality while aligning trajectories, both predictions are mapped to time $0$ by the stop gradient student and compared in a feature space metric $d$:

$$
\begin{aligned}
{\mathbf{x}}_{\mathrm{est}}({\mathbf{x}}_s,s,t) &:= \mathbf{G}_{\boldsymbol{\theta}^-} \big(\mathbf{G}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t),\, t,\, 0\big),\\
{\mathbf{x}}_{\mathrm{target}}({\mathbf{x}}_s,s,u,t) &:= \mathbf{G}_{\boldsymbol{\theta}^-} \big(
\mathbf{G}_{\boldsymbol{\theta}^-}(\mathtt{Solver}_{s\to u}({\mathbf{x}}_s;\boldsymbol{\phi}^\times), u, t),\, t,\, 0\big).
\end{aligned}
$$

The CTM consistency loss is

<span id="eq-ctm-consist-distill"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{consist}}(\boldsymbol{\theta}; \boldsymbol{\phi}^\times)
:=
\mathbb{E}_{s \in [0,T]}\mathbb{E}_{t \in [0,s]}\mathbb{E}_{u \in [t,s)}
\mathbb{E}_{{\mathbf{x}}_0}\mathbb{E}_{{\mathbf{x}}_s |{\mathbf{x}}_0}
\Big[ d\big({\mathbf{x}}_{\mathrm{est}}, {\mathbf{x}}_{\mathrm{target}}\big) \Big],
\end{aligned}
\tag{11.4.5}
$$

which encourages the student to match the empirical PF-ODE solution while preserving generation quality.

**Training from Scratch.** Leveraging CTM’s special parameterization ([Proposition 11.4.1](04-consistency-trajectory-models.md#ctm-g-para)(i)),

$$
{\mathbf{g}}^*({\mathbf{x}}_\tau, \tau, \tau)
= {\mathbf{x}}_\tau - \tau\,{\mathbf{v}}^*({\mathbf{x}}_\tau,\tau)
 \implies
{\mathbf{v}}^*({\mathbf{x}}_\tau,\tau)
= \frac{{\mathbf{x}}_\tau - {\mathbf{g}}^*({\mathbf{x}}_\tau, \tau, \tau)}{\tau}.
$$

We can therefore replace the oracle residual function ${\mathbf{g}}^*(\cdot,\tau,\tau)$ with CTM’s own estimate ${\mathbf{g}}_{\boldsymbol{\theta}^-}(\cdot,\tau,\tau)$ for $\tau \in (0,T]$, which yields a self-induced empirical PF-ODE:

<span id="eq-emp-pf-ode-ctm"></span>

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(\tau)}{\mathop{}\!\mathrm{d} \tau}
= \frac{{\mathbf{x}}(\tau) - {\mathbf{g}}_{\boldsymbol{\theta}^-} \left({\mathbf{x}}(\tau),\tau,\tau\right)}{\tau}.
\end{aligned}
\tag{11.4.6}
$$

We then solve this self-induced ODE to construct a tractable transition target:

$$
\mathbf{G}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s, t)
\approx \mathtt{Solver}_{s\to t}({\mathbf{x}}_s;\boldsymbol{\theta}^-).
$$

This target approaches the oracle flow map only insofar as the diagonal estimate ${\mathbf{g}}_{\boldsymbol{\theta}^-}(\cdot,\tau,\tau)$ accurately recovers the target local dynamics. The diffusion-model loss introduced below provides direct supervision for this local quantity.

As in the distillation case [Equation (11.4.4)](04-consistency-trajectory-models.md#eq-soft), full integration over $[t,s]$ can be costly when $s$ and $t$ are far apart. CTM therefore enforces the semigroup property to obtain a shorter supervision path:

$$
\underbrace{\mathbf{G}_{\boldsymbol{\theta}}({\mathbf{x}}_s, s, t)}_{\approx\, {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)}
 \approx
\underbrace{\mathbf{G}_{\boldsymbol{\theta}^-} \big(\mathtt{Solver}_{s\to u}({\mathbf{x}}_s;\boldsymbol{\theta}^-),\, u,\, t\big)}_{\approx\, {\boldsymbol{\Psi}}_{u\to t}({\boldsymbol{\Psi}}_{s\to u}({\mathbf{x}}_s))},
$$

where $u \sim \mathcal{U}[t,s)$ and $\boldsymbol{\theta}^-$ is a stop gradient copy of the student. The only change from distillation is that the external teacher ${\mathbf{v}}_{\boldsymbol{\phi}^\times}$ is replaced by the self-induced teacher ${\mathbf{g}}_{\boldsymbol{\theta}^-}$.

To couple trajectory matching with sample quality, both predictions are mapped to time $0$ using the stop gradient student and compared in feature space. The target without any pre-trained model is

$$
\hat{{\mathbf{x}}}_{\mathrm{target}}
:= \mathbf{G}_{\boldsymbol{\theta}^-} \big(
\mathbf{G}_{\boldsymbol{\theta}^-}(\mathtt{Solver}_{s\to u}({\mathbf{x}}_s;\boldsymbol{\theta}^-),\, u,\, t),\, t,\, 0\big),
$$

which replaces ${\mathbf{x}}_{\mathrm{target}}$ in [Equation (11.4.5)](04-consistency-trajectory-models.md#eq-ctm-consist-distill), and leads to:

<span id="eq-ctm-consist-scratch"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{consist}}(\boldsymbol{\theta}; {\boldsymbol{\theta}}^-)
:=
\mathbb{E}_{s \in [0,T]}\mathbb{E}_{t \in [0,s]}\mathbb{E}_{u \in [t,s)}
\mathbb{E}_{{\mathbf{x}}_0}\mathbb{E}_{{\mathbf{x}}_s |{\mathbf{x}}_0}
\Big[ d\big({\mathbf{x}}_{\mathrm{est}}, \hat{\mathbf{x}}_{\mathrm{target}}\big) \Big],
\end{aligned}
\tag{11.4.7}
$$

Conceptually, this is self-distillation within CTM: the model supplies its own short horizon teacher signals while the student learns the full transition.

<span id="fig-ill-models"></span>

![Consistency trajectory models compare direct and composite flow maps](../../assets/chapter-11/ctm-consistency.png)

**Figure 11.5:** **Illustration of CTM’s semigroup property.** For any $s \ge u \ge t$, CTM enforces ${\mathbf{G}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s, t) \approx {\mathbf{G}}_{{\boldsymbol{\theta}}^-} \big(\mathtt{Solver}_{s\to u}({\mathbf{x}}_s), u , t\big)$, i.e., a short solver segment $s \to u$ followed by a CTM “jump” to $t$ matches the direct CTM map $s\to t$. The solver may be a pre-trained diffusion or a CTM’s self-induced teacher.

Source: Adapted from [Kim et al. (2024)](references.md#ref-kim2023consistency).

## 11.4.3 Auxiliary Losses in CTM {#subsec-ctm-auxiliary}

(Self-)distillation can underperform the teacher because it optimizes only teacher generated targets, lacking direct supervision from real data. By contrast, CTM can naturally incorporate data driven regularizers, for example by augmenting its objective with denoising score matching and an adversarial (GAN) term ([Goodfellow et al., 2014](references.md#ref-goodfellow2014generative)), to better learn the flow map.

**Natural Integration of Diffusion Loss.** The diffusion–model loss (more precisely, the conditional flow matching loss; see [Equation (5.2.10)](../05-flow-based-perspective/02-flow-matching-framework.md#eq-fm-cfm)) integrates naturally into CTM and provides a fixed regression target that facilitates the learning of the flow map model. To see this, note that we have

$$
{\mathbf{v}}^*({\mathbf{x}}_s, s)
= \frac{{\mathbf{x}}_s - {\mathbf{g}}^*({\mathbf{x}}_s, s, s)}{s},
\qquad
{\mathbf{g}}^*({\mathbf{x}}_s, s, s) \approx {\mathbf{g}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s, s).
$$

This naturally induces a velocity parametrization through ${\mathbf{g}}_{{\boldsymbol{\theta}}}$:

$$
{\mathbf{v}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s)
:= \frac{1}{s}\bigl({\mathbf{x}}_s - {\mathbf{g}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s, s)\bigr).
$$

Using the linear Gaussian path

$$
{\mathbf{x}}_s = \alpha_s {\mathbf{x}}_0 + \sigma_s {\boldsymbol{\epsilon}},
\qquad
{\mathbf{x}}_0 \sim p_{\mathrm{data}},
{\boldsymbol{\epsilon}} \sim \mathcal N(0, \mathbf I),
$$

the diffusion model loss can be written as

<span id="eq-ctm-dsm"></span>

$$
\begin{aligned}
\mathcal L_{\mathrm{DM}}({\boldsymbol{\theta}})
&:=
{\mathbb{E}}_{{\mathbf{x}}_0, {\boldsymbol{\epsilon}}, s}
\Bigl[
w(s)
\bigl\|
 {\mathbf{v}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s) - \left(\alpha_s' {\mathbf{x}}_0 + \sigma_s' {\boldsymbol{\epsilon}}\right)
\bigr\|_2^2
\Bigr].
\end{aligned}
\tag{11.4.8}
$$

$\mathcal L_{\mathrm{DM}}$ improves accuracy when $t$ is close to $s$ by explicitly supervising small jumps along the trajectory. In this regime, the factor $1-\tfrac{t}{s}$ in [Equation (11.4.2)](04-consistency-trajectory-models.md#eq-ctm-para-1) approaches zero, which can weaken gradients and slow learning; $\mathcal L_{\mathrm{DM}}$ supplies a stronger local signal and stabilizes training.

Conceptually, [Equation (11.4.5)](04-consistency-trajectory-models.md#eq-ctm-consist-distill) and [Equation (11.4.7)](04-consistency-trajectory-models.md#eq-ctm-consist-scratch) enforce trajectory matching (zeroth order), while [Equation (11.4.8)](04-consistency-trajectory-models.md#eq-ctm-dsm) enforces slope matching (first order).

**(Optional) GAN Loss.** While consistency and diffusion model loss provide strong regression signals, they can yield overly smooth outputs. CTM therefore optionally adds an adversarial term to encourage sharper, more realistic samples by aligning the generator distribution with the data distribution. With a discriminator $D_{\boldsymbol{\zeta}}$ that distinguishes real ${\mathbf{x}}_{0}\sim p_{\mathrm{data}}$ from generated ${\mathbf{x}}_{\mathrm{est}}({\mathbf{x}}_{s},s,t)$, the objective is

$$
\begin{aligned}
&\mathcal{L}_{\mathrm{GAN}}(\boldsymbol{\theta}, \boldsymbol{\zeta}) := \\ &\mathbb{E}_{\mathbf{x}_0} \big[\log D_{\boldsymbol{\zeta}}(\mathbf{x}_0)\big] + \mathbb{E}_{s \in [0, T]} \mathbb{E}_{t \in [0, s]} \mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{\mathbf{x}_s \vert \mathbf{x}_0} \big[\log (1 - D_{\boldsymbol{\zeta}}( \mathbf{x}_{\mathrm{est}}(\mathbf{x}_s, s, t)))\big],
\end{aligned}
$$

where $D_{\boldsymbol{\zeta}}$ is maximized and ${\mathbf{G}}_{\boldsymbol{\theta}}$ is minimized. Intuitively, the discriminator acts as an adaptive perceptual distance that encourages realistic detail. Theoretically, the GAN term drives distributional matching (Jensen–Shannon divergence) between $p_{\mathrm{data}}$ and the model distribution induced by ${\mathbf{G}}_{\boldsymbol{\theta}}$ ([Goodfellow et al., 2014](references.md#ref-goodfellow2014generative)), which can raise fidelity beyond the teacher.

**Overall CTM Objective.** In summary, CTM unifies (self-)distillation, diffusion, and GAN losses into a single training framework:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CTM}}(\boldsymbol{\theta},\boldsymbol{\zeta})
:=  \mathcal{L}_{\mathrm{consist}}(\boldsymbol{\theta};  \boldsymbol{\phi}^\times/\boldsymbol{\theta}^-)
+ \lambda_{\mathrm{DM}} \mathcal{L}_{\mathrm{DM}}(\boldsymbol{\theta})
+ \lambda_{\mathrm{GAN}} \mathcal{L}_{\mathrm{GAN}}(\boldsymbol{\theta},\boldsymbol{\zeta}),
\end{aligned}
$$

where the teacher is either an external pre-trained model $\boldsymbol{\phi}^\times$ or the self-induced teacher $\boldsymbol{\theta}^-$. The regression style components $\mathcal{L}_{\mathrm{consist}}$ and $\mathcal{L}_{\mathrm{DM}}$ act as strong regularizers, while the optional GAN term improves fine scale detail without sacrificing stability ([Kim et al., 2024](references.md#ref-kim2024pagoda)).

## 11.4.4 Flexible Sampling with CTM {#flexible-sampling}

CTM learns the general flow map ${\boldsymbol{\Psi}}_{s\to t}$ for any $s>t$, which means it supports anytime to anytime transitions. This property enables flexible sampling strategies. For example, CTM proposes *$\gamma$ sampling*, where the hyperparameter $\gamma$ controls the stochasticity during generation. In addition, CTM can reuse standard inference techniques developed for diffusion models, such as ODE based solvers and exact likelihood computation.

In what follows, we fix a discrete time grid for sampling $T=\tau_0 > \tau_1 > \tau_2 > \cdots > \tau_{M}=0$.

<section class="book-algorithm" id="alg-gamma" aria-labelledby="alg-gamma-title" markdown="1">

<div class="book-algorithm__title" id="alg-gamma-title" markdown="span">
Algorithm 11: CTM’s $\gamma$-sampling
</div>

<div class="book-algorithm__meta" markdown="1">

**Input:** Trained CTM $\mathbf{G}_{\boldsymbol{\theta}^\times}$, $\gamma\in[0,1]$, $T=\tau_0&gt;\tau_1 &gt; \tau_2 &gt; \cdots &gt; \tau_{M}=0$.

</div>

<div class="book-algorithm__scroll" tabindex="0" role="region" aria-label="Algorithm 11 steps" markdown="1">
<div class="book-algorithm__lines" markdown="1">

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">1</div>
<div class="book-algorithm__statement" markdown="1">

Start from $\mathbf{x}_{\tau_{0}}\sim p_{\mathrm{prior}}=\mathcal{N}(\boldsymbol{0}, T^2{\mathbf{I}})$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">2</div>
<div class="book-algorithm__statement" markdown="1">

**for** $n=0$ to $M-1$ **do**

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">3</div>
<div class="book-algorithm__statement" markdown="1">

$\tilde{\tau}_{n+1}\leftarrow \sqrt{1-\gamma^{2}}\tau_{n+1}$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">4</div>
<div class="book-algorithm__statement" markdown="1">

Denoise $\mathbf{x}_{\tilde{\tau}_{n+1}}\leftarrow \mathbf{G}_{\boldsymbol{\theta}^\times}(\mathbf{x}_{\tau_{n}},\tau_{n},\tilde{\tau}_{n+1})$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 1" markdown="1">
<div class="book-algorithm__number">5</div>
<div class="book-algorithm__statement" markdown="1">

Diffuse $\mathbf{x}_{\tau_{n+1}}\leftarrow\mathbf{x}_{\tilde{\tau}_{n+1}}+\gamma \tau_{n+1}\boldsymbol{\epsilon}$, where $\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0}, {\mathbf{I}})$

</div>
</div>

<div class="book-algorithm__line" style="--algorithm-indent: 0" markdown="1">
<div class="book-algorithm__number">6</div>
<div class="book-algorithm__statement" markdown="1">

**end for**

</div>
</div>

</div>
</div>

<div class="book-algorithm__meta" markdown="1">

**Output:** $\mathbf{x}_{\tau_{M}}$

</div>

</section>

**Methodology of $\gamma$-Sampling.** CTM’s $\gamma$-sampling introduces a unified family of samplers that arises naturally from learning a general flow map model. It encompasses prior approaches, such as CM’s multistep sampling (see [Algorithm 9](02-discrete-consistency-models.md#alg-cm-sampling)) and time-stepping-style sampling, which is conceptually similar to ODE solvers. The parameter $\gamma$ directly controls the degree of semantic change during generation, making $\gamma$ sampling a flexible and task aware strategy for diverse downstream applications.

<span id="fig-gamma-sampling"></span>

![Gamma sampling interpolates between stochastic and deterministic transitions](../../assets/chapter-11/gamma-sampling.png)

**Figure 11.6:** **Illustration of $\gamma$-sampling with varying $\gamma$ value.** The procedure alternates between denoising with a network evaluation and adding noise in reverse, $(\tau_{n}\xrightarrow{\text{Denoise}} \sqrt{1-\gamma^{2}}\tau_{n+1}\xrightarrow{\text{Noisify}} \tau_{n+1})_{n=0}^{M-1}$. The leftmost panel illustrates $\gamma=1$, corresponding to the fully stochastic case. The rightmost panel shows $\gamma=0$, corresponding to the fully deterministic case. The middle panel depicts intermediate values $\gamma\in(0,1)$, which interpolate between these two extremes.

Source: Adapted from [Kim et al. (2024)](references.md#ref-kim2023consistency).

-   [Figure 11.6](04-consistency-trajectory-models.md#fig-gamma-sampling)-(Left): When $\gamma=1$, it coincides with the multistep sampling introduced in CM (i.e., a special flow map ${\boldsymbol{\Psi}}_{s\to 0}$), which is fully stochastic and results in semantic variation when the number of steps changes.

-   [Figure 11.6](04-consistency-trajectory-models.md#fig-gamma-sampling)-(Right): When $\gamma=0$, it reduces to fully deterministic time-stepping, which estimates the solution trajectory of the PF-ODE. A key distinction between $\gamma$ sampling with $\gamma=0$ and conventional time-stepping ODE-based sampling is that CTM avoids the discretization errors of numerical solvers.

-   [Figure 11.6](04-consistency-trajectory-models.md#fig-gamma-sampling)-(Middle): When $0 < \gamma < 1$, $\gamma$-sampling interpolates between the two extremes by allowing a controlled amount of stochasticity to be injected during sampling.

We highlight that values $\gamma\in[0,1)$ require the general flow map ${\boldsymbol{\Psi}}_{s\to t}$, whereas the special case $\gamma=1$ reduces to CM-style multistep sampling and requires only ${\boldsymbol{\Psi}}_{s\to0}$.

**Analysis of $\gamma$-Sampling.** CTM empirically observed that CM’s multistep sampling degrades in quality once the number of steps $M \geq 4$. To explain this phenomenon, CTM analyzed the underlying cause: when $\gamma \neq 0$, each neural “jump” introduces a small mismatch, and these mismatches accumulate as the model iteratively maps states toward time zero. This error accumulation explains why long multi-step runs can perform poorly. We formalize this idea in the following proposition.

<span id="ctm-gamma"></span>

**Proposition 11.4.2: (Informal) 2-step $\gamma$-sampling**

<span id="th-sampling-agg-error-2-steps"></span> Let $\tau\in(0,T)$ and $\gamma \in [0,1]$. Let $p_{\boldsymbol{\theta}^{*},2}$ denote as the density obtained from the $\gamma$-sampler with the optimal CTM, following the transition sequence $T\rightarrow\sqrt{1-\gamma^2}\tau\rightarrow \tau \rightarrow 0$, starting from $p_T$. Then

$$
\begin{aligned}
\mathcal D_{\mathrm{TV}}\big( p_{\mathrm{data}}, p_{\boldsymbol{\theta}^{*},2} \big)         = \mathcal{O}\left(\sqrt{T-\sqrt{1-\gamma^2}\tau+ \tau}  \right).
\end{aligned}
$$

Here, $\mathcal{D}_{\mathrm{TV}}$ denotes the total variation between distributions (see [Equation (1.1.4)](../01-deep-generative-modeling/01-what-is-deep-generative-modeling.md#eq-f-div)).

**Proof for Proposition.**

We refer the reader to Theorem 8 of [Kim et al. (2024)](references.md#ref-kim2023consistency) for the general case when the number of sampling steps is $M$.

The insights from the above theorem can be summarized as follows:

-   **When** $\gamma = 1$ (corresponding to CM’s multistep sampling)**:** The method performs iterative long-range transitions from $\tau_n$ to $0$ at each step $n$. This leads to error accumulation on the order of

    $$
    \begin{aligned}
    \mathcal{O}\left(\sqrt{T + \tau_1 + \cdots + \tau_M}\right).
    \end{aligned}
    $$

-   **When** $\gamma = 0$ (corresponding to CTM’s deterministic multistep sampling)**:** Such temporal overlap between transitions is eliminated. This avoids error accumulation and yields a tighter bound of $\mathcal{O}(\sqrt{T})$. Empirically, CTM with $\gamma = 0$ provides a favorable trade-off between sampling speed and sample quality: increasing the number of sampling steps improves generation quality without introducing instability.

**CTM Supports Diffusion Inference.** Since CTM learns the score function (or denoiser) directly through ${\mathbf{g}}_{\boldsymbol{\theta}}$, thanks to its parametrization in [Equation (11.4.2)](04-consistency-trajectory-models.md#eq-ctm-para-1), it is compatible with inference techniques originally developed for diffusion models. For instance, one can compute exact likelihoods ([Section 4.3.2](../04-score-sde-framework/03-training-and-sampling.md#subsec-sde-sample)) or apply advanced samplers such as DDIM or DPM ([Chapter 9](../09-numerical-solvers/index.md#ch-numerical-solvers)) for generation, by using ${\mathbf{g}}_{\boldsymbol{\theta}}(\cdot, s, s)$.

[^1]: For affine-in-time schedules, such as $(\alpha_t,\sigma_t)=(1,t)$ or $(1-t,t)$, the plain Euler step in diffusion time coincides exactly with the deterministic DDIM update. For general schedules, the two agree only to first order. See discussion in [Section 9.2.2](../09-numerical-solvers/02-ddim.md#subsec-ddim-classical-euler).
