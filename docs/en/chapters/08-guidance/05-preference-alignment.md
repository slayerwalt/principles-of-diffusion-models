# 8.5 From Reinforcement Learning to Direct Preference Optimization for Model Alignment {#sec-rlhf-dpo}

In the pursuit of aligning generative models with human intent, the prevailing paradigm has been Reinforcement Learning from Human Feedback (RLHF). While effective, RLHF is a complex, multi-stage process that can be unstable. This section introduces *Direct Preference Optimization (DPO)* ([Rafailov et al., 2023](../../references.md#ref-rafailov2023direct)), a more streamlined and stable method that reaches the same goal without explicit reward modeling or reinforcement learning. We then outline its extension to diffusion models via *Diffusion-DPO* ([Wallace et al., 2024](../../references.md#ref-wallace2024diffusion)).

## 8.5.1 The Motivation: Circumventing the Pitfalls of RLHF {#motivation-rlhf}

The goal of alignment is to steer a base, pre-trained model (e.g., an SFT model) toward outputs that humans prefer. RLHF proceeds in three stages. First, *supervised fine-tuning (SFT)* trains a base model on prompt–response pairs. Second, *reward modeling (RM)* fits a model on preference data consisting of prompts $\mathbf{c}$ and paired responses (a preferred “winner” $\mathbf{x}_w$ and a dispreferred “loser” $\mathbf{x}_l$), learning a scalar $r(\mathbf{c},\mathbf{x})$ with $r(\mathbf{c},\mathbf{x}_w)>r(\mathbf{c},\mathbf{x}_l)$. Third, *RL fine-tuning* optimizes the SFT model (policy $\pi$[^1]) with an algorithm such as PPO ([Schulman et al., 2017](../../references.md#ref-schulman2017proximal)), maximizing expected reward from $r$ while regularizing by a KL penalty that keeps $\pi$ close to the reference/SFT distribution.

Despite its impact, this pipeline suffers from drawbacks: the RL stage is unstable and computationally expensive because it is on-policy—each update requires freshly generated samples from the current model; it also requires training and hosting multiple large models (SFT, reward, and sometimes a value model); and it optimizes only a proxy for human preferences, so flaws in the reward model can be exploited. This motivates a central question:

<span id="question-8-5-1"></span>

**Question 8.5.1**

Can we eliminate explicit reward modeling and the unstable RL step, directly optimizing the model on preference data?

Direct Preference Optimization (DPO) streamlines alignment by replacing the multi-stage RLHF pipeline with a single, supervised-style step. Instead of training a separate reward model and running unstable RL algorithms like PPO, DPO directly fits the policy to preference pairs using a simple logistic loss, while staying close to a fixed reference model. The key insight is that the KL-regularized RLHF objective can be rewritten so that the log-likelihood ratio between the policy and the reference acts as an implicit reward. This preserves the same regularization toward the reference policy but avoids costly rollouts and explicit reward modeling.

In [Section 8.5.2](05-preference-alignment.md#subsec-rlhf-bradley-terry), we briefly review the RLHF pipeline and its reliance on large reward models and RL fine-tuning. In [Section 8.5.3](05-preference-alignment.md#subsec-dpo), we present DPO, originally proposed for language models, which circumvents reward model training and simplifies alignment fine-tuning. Finally, in [Section 8.5.4](05-preference-alignment.md#subsec-diffusion-dpo), we extend this idea to diffusion models, introducing Diffusion-DPO as a practical and stable alignment method in the generative modeling setting.

## 8.5.2 RLHF: Bradley-Terry View {#subsec-rlhf-bradley-terry}

**Short Introduction to RLHF.** RLHF begins with a learned judge: a reward model $r_{\boldsymbol\psi}$ that assigns a scalar preference score to candidate responses for the same prompt $\mathbf c$. The dataset $\mathcal D$ consists of pairs $(\tilde{\mathbf x},\mathbf x)$ annotated with a label $y$ indicating whether $\tilde{\mathbf x}$ is preferred over $\mathbf x$. The label can be binary $y\in\{0,1\}$ or a soft value $y\in[0,1]$ obtained by aggregating multiple raters. The training objective is a simple logistic loss

<span id="eq-rm-emp"></span>

$$
\begin{aligned}
    \mathcal L_{\mathrm{RM}}(\boldsymbol\psi)
=
- \mathbb E_{(\mathbf c,\tilde{\mathbf x},\mathbf x,y)\sim\mathcal D}
\Big[
y &\log \sigma \big(r_{\boldsymbol\psi}(\mathbf c,\tilde{\mathbf x})-r_{\boldsymbol\psi}(\mathbf c,\mathbf x)\big)
\\ &+(1-y)\log \big(1-\sigma \big(r_{\boldsymbol\psi}(\mathbf c,\tilde{\mathbf x})-r_{\boldsymbol\psi}(\mathbf c,\mathbf x)\big)\big)
\Big],
\end{aligned}
\tag{8.5.1}
$$

where $\sigma(u)=1/(1+e^{-u})$. In practice, preference pairs in $\mathcal D$ may originate from various sources: curated responses, model snapshots at different checkpoints, or generations from a pre-trained conditional diffusion model. A standard convention is to store them in an ordered format $(\text{winner}, \text{loser})$. Under this convention we simply set $y=1$, and [Equation (8.5.1)](05-preference-alignment.md#eq-rm-emp) reduces to the special case (with $\tilde{\mathbf{x}} = {\mathbf{x}}^w$ and ${\mathbf{x}} = {\mathbf{x}}^l$):

<span id="eq-rm-bt"></span>

$$
\mathcal L_{\mathrm{RM}}(\boldsymbol\psi)
=
- \mathbb E_{(\mathbf c,\mathbf x_w,\mathbf x_l)\sim\mathcal D}
\Big[\log \sigma \big(r_{\boldsymbol\psi}(\mathbf c,\mathbf x_w)-r_{\boldsymbol\psi}(\mathbf c,\mathbf x_l)\big)\Big].
\tag{8.5.2}
$$

**Bradley-Terry View and KL Connection.** It is standard to interpret

$$
p_{r_{\boldsymbol\psi}}(\tilde{\mathbf x}\succ \mathbf x|\mathbf c)
:= \sigma \big(r_{\boldsymbol\psi}(\mathbf c,\tilde{\mathbf x})-r_{\boldsymbol\psi}(\mathbf c,\mathbf x)\big)
$$

through the Bradley-Terry (BT) model ([Bradley and Terry, 1952](../../references.md#ref-bradley1952rank)), which converts two scalar scores into a win probability. This formulation highlights two key properties: (i) only the *difference* of scores matters (so $r_{\boldsymbol\psi}(\mathbf c,\cdot)$ is shift-invariant), and (ii) the loss pushes the predicted winner’s score above the loser’s score. To see (ii) intuitively, consider one pair with label $y\in\{0,1\}$ and define

$$
\Delta r := r_{\boldsymbol\psi}(\mathbf c,\tilde{\mathbf x})-r_{\boldsymbol\psi}(\mathbf c,\mathbf x),
\quad
p:=\sigma(\Delta r),\quad \sigma(u)=\tfrac{1}{1+e^{-u}}.
$$

The per-example logistic loss is

$$
\ell=-\big[y\log p+(1-y)\log(1-p)\big].
$$

Then

$$
\frac{\partial \ell}{\partial \Delta r}=\sigma(\Delta r)-y.
$$

Under gradient descent with step size $\eta>0$, the score gap updates as

$$
\Delta r \leftarrow \Delta r-\eta\big(\sigma(\Delta r)-y\big).
$$

Hence, if $y=1$ (“$\tilde{\mathbf x}$ wins”), then $\sigma(\Delta r)-1\le 0$, so $\Delta r$ increases (winner up, loser down); if $y=0$, $\Delta r$ decreases.

Each per-example term in [Equation (8.5.1)](05-preference-alignment.md#eq-rm-emp) can be viewed as the cross-entropy between the observed Bernoulli label and the model’s predicted win probability:

$$
-\big[y\log p_{r_{\boldsymbol\psi}}+(1-y)\log(1-p_{r_{\boldsymbol\psi}})\big]
= \mathcal D_{\mathrm{KL}} \big(\mathrm{Bern}(y) \big\| \mathrm{Bern}(p_{r_{\boldsymbol\psi}})\big)
+ \mathcal H \big(\mathrm{Bern}(y)\big),
$$

where $\mathcal H$ is the entropy of the target Bernoulli distribution. Averaging over the dataset $\mathcal D$ gives

<span id="eq-reward-loss"></span>

$$
\begin{aligned}
\mathcal L_{\mathrm{RM}}(\boldsymbol\psi)
= \mathbb E_{\mathcal D} \Big[\mathcal D_{\mathrm{KL}}\left(\mathrm{Bern}(y) \big\| \mathrm{Bern}(p_{r_{\boldsymbol\psi}})\right)\Big]
+ \underbrace{\mathbb E_{\mathcal D} \big[\mathcal H(\mathrm{Bern}(y))\big]}_{\text{independent of }\boldsymbol\psi}.
\end{aligned}
\tag{8.5.3}
$$

Thus, minimizing the logistic loss is equivalent to minimizing the KL divergence between the empirical Bernoulli distribution of human labels and the model’s predicted Bernoulli distribution. In the binary case ($y\in\{0,1\}$), this equivalence is exact; for soft labels ($y\in[0,1]$), the result holds up to an entropy constant offset. Intuitively, the reward model is trained to adjust its win probabilities until they align with the empirical human win rates observed in the dataset.

From this point onward, we adopt the most common convention where $\mathcal D$ stores pairs in an ordered format: $({\mathbf{x}}^w, {\mathbf{x}}^l, {\mathbf{c}}) \sim \mathcal D$. Under this convention, the label is always $y=1$, and the loss simplifies to the ordered form given in [Equation (8.5.2)](05-preference-alignment.md#eq-rm-bt), which we will use in the following discussion.

**KL Regularized Policy Optimization (with Fixed Reward).** With the fitted reward $r:=r_{\boldsymbol\psi^\times}$ trained via [Equation (8.5.2)](05-preference-alignment.md#eq-rm-bt), and a conditional pre-trained diffusion model $p_{\boldsymbol{\phi}^\times}(\mathbf x|\mathbf c)$, RLHF then adjusts a learnable policy $\pi_{\boldsymbol\theta}(\mathbf x|\mathbf c)$, usually fine-tuned on top of $p_{\boldsymbol{\phi}^\times}(\mathbf x|\mathbf c)$, toward higher-reward responses. At the same time, the policy is regularized to stay close to a reference model, taken as the pre-trained diffusion model $\pi_{\mathrm{ref}}(\mathbf x|\mathbf c):=p_{\boldsymbol{\phi}^\times}(\mathbf x|\mathbf c)$, using a $\mathcal{D}_{\mathrm{KL}}$ penalty:

<span id="eq-rlhf-policy-correct"></span>

$$
\max_{\boldsymbol\theta}
\mathbb E_{\mathbf c\sim p(\mathbf c)}
\Big[
\mathbb E_{\mathbf x\sim \pi_{\boldsymbol\theta}(\cdot|\mathbf c)}\big[r_{\boldsymbol\psi}(\mathbf c,\mathbf x)\big]
-\beta \mathcal{D}_{\mathrm{KL}} \big(\pi_{\boldsymbol\theta}(\cdot|\mathbf c) \big\| \pi_{\mathrm{ref}}(\cdot|\mathbf c)\big)
\Big],
\tag{8.5.4}
$$

which makes the two forces explicit: seek samples the judge prefers, but stay close to the pre-trained reference.

We remark that the reward objective in [Equation (8.5.2)](05-preference-alignment.md#eq-rm-bt) uses only labeled pairs and does not require that $\mathcal D$ be generated by the reference model (i,e., the pre-trained conditional diffusion model). While not required, collecting pairs from models close to the intended policy can reduce distribution shift and make the learned reward more reliable in the region where it will be used.

In summary, RLHF proceeds in two stages: first fit the reward $r^*$ by minimizing the loss in [Equation (8.5.2)](05-preference-alignment.md#eq-rm-bt) (equivalently, the expected binary $\mathcal{D}_{\mathrm{KL}}$ in [Equation (8.5.3)](05-preference-alignment.md#eq-reward-loss)); then optimize the policy $\pi^*$ by solving [Equation (8.5.4)](05-preference-alignment.md#eq-rlhf-policy-correct).

## 8.5.3 DPO Framework {#subsec-dpo}

**The Bridge from RLHF.** The KL-regularized policy objective in [Equation (8.5.4)](05-preference-alignment.md#eq-rlhf-policy-correct) has a simple closed-form solution for each prompt $\mathbf c$, given the fitted reward $r := r_{\boldsymbol\psi^\times}$, expressed in the following energy-based form ([Peters et al., 2010](../../references.md#ref-peters2010relative)):

<span id="eq-rlhf-closed-form"></span>

$$
\pi^*(\mathbf x|\mathbf c)
=
\frac{1}{Z(\mathbf c)} \pi_{\mathrm{ref}}(\mathbf x|\mathbf c)\exp(r(\mathbf c,\mathbf x)/\beta),
\tag{8.5.5}
$$

where $\pi_{\mathrm{ref}}(\mathbf x|\mathbf c):=p_{\boldsymbol{\phi}^\times}(\mathbf x|\mathbf c)$, and $Z(\mathbf c)$ is the partition function ensuring $\int \pi^*(\mathbf x|\mathbf c) \mathop{}\!\mathrm{d} \mathbf x=1$.

For smaller $\beta$, $\exp(r/\beta)$ becomes sharper, so $\pi^*$ concentrates on high reward regions: reward dominates, the policy moves farther from $\pi_{\mathrm{ref}}$, diversity decreases, and training may become unstable or prone to reward hacking. For larger $\beta$, $\exp(r/\beta)$ flattens, keeping $\pi^*$ closer to $\pi_{\mathrm{ref}}$: the KL term dominates, updates are conservative, diversity follows the reference, but reward gains are limited.

Since our aim is to fine-tune the policy directly (without training a separate reward model), [Equation (8.5.5)](05-preference-alignment.md#eq-rlhf-closed-form) lets us *define* an *implicit reward* from any policy. We introduce below:

**Defining an Implicit Reward Motivated by Inverting [Equation (8.5.5)](05-preference-alignment.md#eq-rlhf-closed-form).** [Equation (8.5.5)](05-preference-alignment.md#eq-rlhf-closed-form) suggests an immediate inversion: for any policy $\pi$ (with support contained in $\pi_{\mathrm{ref}}$), define

<span id="eq-implicit-reward"></span>

$$
r_{\pi}(\mathbf c,\mathbf x)
 =
\beta \log\frac{\pi(\mathbf x|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x|\mathbf c)}
 + \beta \log Z(\mathbf c).
\tag{8.5.6}
$$

Then [Equation (8.5.5)](05-preference-alignment.md#eq-rlhf-closed-form) holds with $\pi$ in place of $\pi^*$, i.e., $\pi$ would be the optimizer of [Equation (8.5.4)](05-preference-alignment.md#eq-rlhf-policy-correct) for the reward function $r_{\pi}$. In this sense, $r_{\pi}$ is an *implicit (policy-induced) reward*: it is identified up to the prompt-dependent constant $\beta\log Z(\mathbf c)$, which vanishes in any pairwise comparison such as in the BT model:

$$
r_{\pi}(\mathbf c,\mathbf x_w)-r_{\pi}(\mathbf c,\mathbf x_l)
=
\beta\Big(
\log\frac{\pi(\mathbf x_w|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_w|\mathbf c)}
-
\log\frac{\pi(\mathbf x_l|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_l|\mathbf c)}
\Big).
$$

This cancellation is exactly what makes the constant irrelevant for preference learning and leads directly to the DPO loss on log-probability differences.

**DPO’s Training Loss.** Plug the implicit reward [Equation (8.5.6)](05-preference-alignment.md#eq-implicit-reward) into the BT model of [Equation (8.5.2)](05-preference-alignment.md#eq-rm-bt) for a labeled pair $(\mathbf x_w,\mathbf x_l)$ under the same prompt $\mathbf c$. The constants $\log Z(\mathbf c)$ cancel between winner and loser, yielding a single logistic-loss objective on log-probability differences:

$$
\begin{aligned}
\mathcal L_{\mathrm{DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
&=
-\mathbb E_{(\mathbf c,\mathbf x_w,\mathbf x_l)\sim\mathcal D} \left[
\log \sigma \Big(
\beta\big(
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x_w|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_w|\mathbf c)}
-
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x_l|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_l|\mathbf c)}
\big)
\Big)
\right].
\end{aligned}
$$

In words: DPO pushes up the (temperature-scaled) advantage of the winner over the loser, measured as the difference of log-likelihood improvements over the reference:

$$
-\log \sigma \left(
  \beta
  \bigl[
    \text{log-ratio difference of }
    \tfrac{\pi_{\boldsymbol\theta}}{\pi_{\mathrm{ref}}}
    \text{ at } \mathbf x_w \text{ vs. } \mathbf x_l
  \bigr]
\right).
$$

This achieves the goal of RLHF in a single, stable classification-style fine-tuning stage, without training an explicit reward model.

## 8.5.4 Diffusion-DPO {#subsec-diffusion-dpo}

Throughout this subsection, we retain the policy notation from [Section 8.5.3](05-preference-alignment.md#subsec-dpo): $\pi_{\boldsymbol\theta}$ and $\pi_{\mathrm{ref}}$ denote the learned and reference diffusion policies, including their path and transition distributions, while $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ continues to denote the fixed forward noising process.

**Why Naive DPO Fails for Diffusion Models?** Evaluating the sample likelihood $\pi_{\boldsymbol{\theta}}(\mathbf x|\mathbf c)$ in diffusion models requires the instantaneous change-of-variables formula (divergence of the drift) of ODE solving (see [Equation (4.3.7)](../04-score-sde-framework/03-training-and-sampling.md#eq-score-sde-likelihood))[^2], which is computationally intensive. Moreover, differentiating through the entire sampling trajectory can suffer from vanishing or exploding gradients. To avoid these issues, Diffusion-DPO works at the *path* level. We take the discrete-time diffusion model (e.g., DDPM) as an illustrative example; the continuous-time diffusion model is analogous.

**Defining Pathwise Implicit Rewards.** Let a trajectory be $\mathbf x_{0:T}:=(\mathbf x_T,\ldots,\mathbf x_0)$ under the reverse-time Markov chain with conditionals $\pi(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)$. Here, $\mathbf x_T$ is the highest-noise state and $\mathbf x_0$ is the clean output.

Human preferences are defined on the final output through $r(\mathbf c,\mathbf x_0)$. Since $\mathbf x_0$ is part of every diffusion trajectory, the reward itself does not need to change when we move to path space. The useful step is instead to regularize the entire reverse process. By the chain rule for KL divergence, the KL divergence between the full path distributions is at least as large as the KL divergence between their endpoint marginals. This motivates the pathwise objective

$$
\max_{{\boldsymbol{\theta}}}\mathbb E_{\mathbf c\sim p(\mathbf c)}
\Big[
\mathbb E_{\mathbf x_{0:T}\sim\pi_{{\boldsymbol{\theta}}}(\cdot|\mathbf c)}
\big[r(\mathbf c,\mathbf x_0)\big]
-\beta
\mathcal D_{\mathrm{KL}}
\big(
\pi_{{\boldsymbol{\theta}}}(\cdot|\mathbf c)
\big\|
\pi_{\mathrm{ref}}(\cdot|\mathbf c)
\big)
\Big],
$$

where $\pi_{{\boldsymbol{\theta}}}(\cdot|\mathbf c)$ and $\pi_{\mathrm{ref}}(\cdot|\mathbf c)$ denote the corresponding path distributions. The reward term is unchanged, while the KL penalty now also controls how the latent denoising trajectories deviate from the reference.

At the distributional level, assuming the normalizer is finite, the optimizer for each prompt $\mathbf c$ has the energy-based form

<span id="eq-path-closed-form"></span>

$$
\pi^*(\mathbf x_{0:T}|\mathbf c)
=
\frac{1}{Z(\mathbf c)}
\pi_{\mathrm{ref}}(\mathbf x_{0:T}|\mathbf c)
\exp\big(r(\mathbf c,\mathbf x_0)/\beta\big).
\tag{8.5.7}
$$

As in ordinary DPO, we can invert this relation and define the *implicit path reward* induced by any policy $\pi$:

$$
R_{\pi}(\mathbf c,\mathbf x_{0:T})
:=
\beta\log
\frac{\pi(\mathbf x_{0:T}|\mathbf c)}
{\pi_{\mathrm{ref}}(\mathbf x_{0:T}|\mathbf c)}
+\beta\log Z(\mathbf c).
$$

At the pathwise optimum, $R_{\pi^*}(\mathbf c,\mathbf x_{0:T})=r(\mathbf c,\mathbf x_0)$. For a general policy, however, $R_{\pi}$ may also depend on the latent trajectory. Since the preference data compare only the clean endpoints, we therefore average this implicit path reward over trajectories associated with each endpoint.

**From Pathwise Implicit Rewards to DPO.** For a labeled endpoint pair $(\mathbf x_0^w,\mathbf x_0^l)$ under the same prompt $\mathbf c$, the Bradley–Terry margin is obtained by comparing the conditional averages of their implicit path rewards:

<span id="eq-diffusion-dpo"></span>

$$
\begin{aligned}
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
&:=
- \mathbb E_{(\mathbf c,\mathbf x_0^w,\mathbf x_0^l)\sim\mathcal D}
 \left[\log \sigma \big(\Delta R(\mathbf c;\boldsymbol\theta)\big)\right], \quad \text{where}\\
\Delta R(\mathbf c;\boldsymbol\theta)
&:=
\underbrace{\mathbb E_{\mathbf x_{1:T}^w\sim \pi_{\boldsymbol\theta}(\cdot | \mathbf x_0^w,\mathbf c)}
 \Big[R_{\pi_{\boldsymbol\theta}} \big(\mathbf c,(\mathbf x_0^w,\mathbf x_{1:T}^w)\big)\Big]}_{\text{winner path expectation}}
\\ &\quad\quad\quad-
\underbrace{\mathbb E_{\mathbf x_{1:T}^l\sim \pi_{\boldsymbol\theta}(\cdot | \mathbf x_0^l,\mathbf c)}
 \Big[R_{\pi_{\boldsymbol\theta}} \big(\mathbf c,(\mathbf x_0^l,\mathbf x_{1:T}^l)\big)\Big]}_{\text{loser path expectation}}.
\end{aligned}
\tag{8.5.8}
$$

Here, the expectation $\mathbb E_{\mathbf x_{1:T}\sim\pi_{\boldsymbol\theta}(\cdot|\mathbf x_0,\mathbf c)}[\cdot]$ means: given a fixed endpoint $\mathbf x_0$ (e.g., the winner $\mathbf x_0^w$) from the dataset, we take an expectation over latent denoising trajectories $\mathbf x_{1:T}$ under the model-induced conditional path distribution (the posterior over reverse-time trajectories) that, with kernels $\pi_{\boldsymbol\theta}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)$, could produce $\mathbf x_0$. Since these intermediate states are unobserved, we average the path reward over all such trajectories.

However, [Equation (8.5.8)](05-preference-alignment.md#eq-diffusion-dpo) is impractical for three practical reasons:

1.  **Endpoint Conditioning Induces an Intractable Path Posterior.** The term $\mathbb E_{\pi_{\boldsymbol\theta}(\mathbf x_{1:T}|\mathbf x_0,\mathbf c)}[\cdot]$ averages over reverse paths constrained to hit $\mathbf x_0$, whereas the sampler runs $\mathbf x_T \to \cdots \to \mathbf x_0$ without this constraint. Conditioning on the endpoint creates a diffusion-bridge posterior with generally no closed form and costly sampling.

2.  **Nested, $\boldsymbol\theta$-Coupled Expectations.** The loss $-\log\sigma(\Delta R(\mathbf c;\boldsymbol\theta))$ with

    $$
    \Delta R=\mathbb E_{\text{paths}|\mathbf x_0^w,\mathbf c}[R_{\pi_{\boldsymbol\theta}}]
    -\mathbb E_{\text{paths}|\mathbf x_0^l,\mathbf c}[R_{\pi_{\boldsymbol\theta}}]
    $$

    has both the path joint distribution and the integrand $R_{\pi_{\boldsymbol\theta}}$ depending on $\boldsymbol\theta$. Thus $\nabla_{\boldsymbol\theta}$ must differentiate through the sampling distribution, leading to REINFORCE/pathwise couplings and high-variance gradients.

3.  **Long Chains, Large Sums, and Expensive Backpropagation.** In $R_{\pi_{\boldsymbol\theta}}(\mathbf c,\mathbf x_{0:T})$, computing

    $$
    \beta \left[\log \pi_{\boldsymbol\theta}(\mathbf x_{0:T}|\mathbf c)
    -\log \pi_{\mathrm{ref}}(\mathbf x_{0:T}|\mathbf c)\right]
    $$

    requires $\mathcal O(T)$ per-step log-densities with $T \sim 10^2$–$10^3$, for both policy $\pi_{\boldsymbol\theta}$ and reference $\pi_{\mathrm{ref}}$, and for both winner/loser paths. Backpropagating through these stochastic chains (or bridge samplers) is memory and compute heavy and can be unstable; repeating this over many samples per pair and across all triplets pushes training beyond practical budgets.

**Toward a Tractable Surrogate for [Equation (8.5.8)](05-preference-alignment.md#eq-diffusion-dpo).** To make this computable, we apply a key mathematical insight. By leveraging properties of diffusion models and applying Jensen’s inequality, we can optimize a tractable upper bound on this loss. This transforms the problem from evaluating an entire path’s likelihood to evaluating an expectation over the individual, single-step transitions within the path:

Because $-\log\sigma(\cdot)$ is convex, Jensen’s inequality yields an upper bound by moving the inner expectations outside the nonlinear function:

$$
\begin{aligned}
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
\le
-\mathbb E_{(\mathbf c,\mathbf x_0^w,\mathbf x_0^l)\sim\mathcal D}
  \mathbb E_{\substack{\mathbf x^w_{1:T}\sim \pi_{\boldsymbol\theta}(\cdot | \mathbf x_0^w,\mathbf c)\\[1pt]
                       \mathbf x^l_{1:T}\sim \pi_{\boldsymbol\theta}(\cdot | \mathbf x_0^l,\mathbf c)}}
\left[
\log \sigma \Big(
R_{\pi_{\boldsymbol\theta}}(\mathbf c,\mathbf x^w_{0:T})
-
R_{\pi_{\boldsymbol\theta}}(\mathbf c,\mathbf x^l_{0:T})
\Big)
\right].
\end{aligned}
$$

Using the implicit-reward identity $R_{\pi_{\boldsymbol\theta}}=\beta\log\frac{\pi_{\boldsymbol\theta}}{\pi_{\mathrm{ref}}}+\beta\log Z(\mathbf c)$ and cancellation of the constant between winner and loser, the bound becomes

<span id="eq-dpo-path-bound"></span>

$$
\begin{aligned}
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
& \le
-\mathbb E
\left[
\log \sigma \Big(
\beta\big(
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x^w_{0:T}| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x^w_{0:T}| \mathbf c)}
-
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x^l_{0:T}| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x^l_{0:T}| \mathbf c)}
\big)
\Big)
\right].
\end{aligned}
\tag{8.5.9}
$$

**A Tractable Surrogate (Stepwise Form).** We now exploit the Markov property of the reverse process to decompose the upper bound of $\mathcal L_{\mathrm{Diff\text{-}DPO}}$. This allows us to express the path-level preference as a sum of per-step contributions, converting the intractable pathwise loss into a tractable single-step estimator. The resulting form yields a DPO-style log-sigmoid loss whose inner margin reduces to a DSM-style MSE difference. Concretely, for the reverse chain,

$$
\begin{aligned}
\pi_{\boldsymbol\theta}(\mathbf x_{0:T}| \mathbf c)
&=\pi_{\boldsymbol\theta}(\mathbf x_T| \mathbf c) \prod_{t=1}^{T}
\pi_{\boldsymbol\theta}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c),
\\
\pi_{\mathrm{ref}}(\mathbf x_{0:T}| \mathbf c)
&=\pi_{\mathrm{ref}}(\mathbf x_T| \mathbf c) \prod_{t=1}^{T}
\pi_{\mathrm{ref}}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c).
\end{aligned}
$$

Hence

$$
\frac{\pi_{\boldsymbol\theta}(\mathbf x_{0:T}| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_{0:T}| \mathbf c)}
=
\frac{\pi_{\boldsymbol\theta}(\mathbf x_T| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_T| \mathbf c)}
\prod_{t=1}^{T}
\frac{\pi_{\boldsymbol\theta}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c)}
     {\pi_{\mathrm{ref}}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c)}.
$$

If the prior at time $T$ is the same for both models, $\pi_{\boldsymbol\theta}(\mathbf x_T| \mathbf c)=\pi_{\mathrm{ref}}(\mathbf x_T| \mathbf c)$, then the first factor equals $1$, and taking logs yields

$$
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x_{0:T}| \mathbf c)}{\pi_{\mathrm{ref}}(\mathbf x_{0:T}| \mathbf c)}
=\sum_{t=1}^{T}
\log\frac{\pi_{\boldsymbol\theta}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c)}
         {\pi_{\mathrm{ref}}(\mathbf x_{t-1}| \mathbf x_t,\mathbf c)}.
$$

It follows that the bound in [Equation (8.5.9)](05-preference-alignment.md#eq-dpo-path-bound) can be written as

$$
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
  \le  - {\mathbb{E}} \left[\log\sigma \Big(\beta \sum_{t=1}^T \Delta_t\Big)\right],
$$

where each per-step contribution is

$$
\Delta_t
= \log\frac{\pi_{\boldsymbol\theta}(\mathbf x^w_{t-1}|\mathbf x^w_t,\mathbf c)}
               {\pi_{\mathrm{ref}}(\mathbf x^w_{t-1}|\mathbf x^w_t,\mathbf c)}
- \log\frac{\pi_{\boldsymbol\theta}(\mathbf x^l_{t-1}|\mathbf x^l_t,\mathbf c)}
               {\pi_{\mathrm{ref}}(\mathbf x^l_{t-1}|\mathbf x^l_t,\mathbf c)}.
$$

To reduce the sum over timesteps to a single randomly selected transition, observe that $\beta\sum_{s=1}^T\Delta_s = {\mathbb{E}}_{t\sim\mathcal U\{1,\dots,T\}}[\beta T\Delta_t]$. Since $-\log\sigma(\cdot)$ is convex, Jensen’s inequality therefore gives, after taking the existing outer expectations,

$$
\mathcal L_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
\leq
-{\mathbb{E}}\left[
\log\sigma\bigl(\beta T\Delta_t\bigr)
\right],
$$

where the expectation is over $(\mathbf c,\mathbf x_0^w,\mathbf x_0^l)\sim\mathcal D$, $t\sim\mathcal U\{1,\dots,T\}$, and $\mathbf x_{t-1:t}^w\sim \pi_{\boldsymbol\theta}(\cdot|\mathbf x_0^w,\mathbf c)$ and $\mathbf x_{t-1:t}^l\sim \pi_{\boldsymbol\theta}(\cdot|\mathbf x_0^l,\mathbf c)$. Although the bound now depends on only one transition, sampling these endpoint-conditioned transition pairs remains intractable.

Following Diffusion-DPO, we approximate the intractable model-induced conditional path using the fixed forward noising process. For each clean endpoint $\mathbf x_0$, we first sample $\mathbf x_t \sim p_t(\mathbf x_t|\mathbf x_0)$. Conditioned on $(\mathbf x_0,\mathbf x_t)$, the forward process induces the tractable posterior $p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)$. Averaging the reverse-transition log-ratio under this posterior gives

$$
\begin{aligned}
&{\mathbb{E}}_{p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}
\left[
\log
\frac{
\pi_{\boldsymbol\theta}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
}{
\pi_{\mathrm{ref}}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
}
\right]
\\&=
-
\mathcal D_{\mathrm{KL}}
\left(
p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)
\,\middle\|\,
\pi_{\boldsymbol\theta}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
\right)+
\mathcal D_{\mathrm{KL}}
\left(
p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)
\,\middle\|\,
\pi_{\mathrm{ref}}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
\right).
\end{aligned}
$$

For Gaussian reverse kernels with the same prescribed reverse variance, these KL divergences reduce to weighted denoising regression errors. Under $\boldsymbol\epsilon$-prediction,

<span id="eq-dpo-gaussian-log-ratio"></span>

$$
\begin{aligned}
{\mathbb{E}}_{p(\mathbf x_{t-1}|\mathbf x_t,\mathbf x_0)}
\left[
\log
\frac{
\pi_{\boldsymbol\theta}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
}{
\pi_{\mathrm{ref}}(\mathbf x_{t-1}|\mathbf x_t,\mathbf c)
}
\right]
=
-\lambda_t
\left[
\left\|
\hat{\boldsymbol\epsilon}_{\boldsymbol\theta}(\mathbf x_t,t,\mathbf c)
-\boldsymbol\epsilon
\right\|^2
-
\left\|
\hat{\boldsymbol\epsilon}_{\mathrm{ref}}(\mathbf x_t,t,\mathbf c)
-\boldsymbol\epsilon
\right\|^2
\right],
\end{aligned}
\tag{8.5.10}
$$

where $\mathbf x_t = \alpha_t\mathbf x_0 + \sigma_t\boldsymbol\epsilon$ with $\boldsymbol\epsilon\sim\mathcal N(\mathbf 0,\mathbf I)$, and $\lambda_t>0$ collects the schedule-dependent weighting factors.

For notation simplicity, define for any noised sample $(\mathbf x_t,\boldsymbol\epsilon)$:

$$
\Delta\mathrm{MSE}(\mathbf x_t;\boldsymbol\epsilon)
:= \big\|\hat{\boldsymbol\epsilon}_{\boldsymbol\theta}(\mathbf x_t,t,\mathbf c)-\boldsymbol\epsilon\big\|^2
 - \big\|\hat{\boldsymbol\epsilon}_{\mathrm{ref}}(\mathbf x_t,t,\mathbf c)-\boldsymbol\epsilon\big\|^2.
$$

This yields the practical single-step Diffusion-DPO surrogate for $\mathcal L_{\mathrm{Diff\text{-}DPO}}$:

$$
\begin{aligned}
&\tilde{\mathcal L}_{\mathrm{Diff\text{-}DPO}}(\boldsymbol\theta;\pi_{\mathrm{ref}})
    :=
    \\&-\mathbb E_{\substack{(\mathbf c,\mathbf x_0^w,\mathbf x_0^l)\sim\mathcal D\\
    t\sim\mathcal U\{1,\dots,T\}, \boldsymbol\epsilon^w,\boldsymbol\epsilon^l\sim\mathcal N(\mathbf 0,\mathbf I)}}
    \Big[
    \log\sigma \Big(
    \beta T\lambda_t\big(\Delta\mathrm{MSE}(\mathbf x_t^l;\boldsymbol\epsilon^l)-\Delta\mathrm{MSE}(\mathbf x_t^w;\boldsymbol\epsilon^w)\big)
    \Big)
    \Big],
\end{aligned}
$$

where $\mathbf x_t^w=\alpha_t\mathbf x_0^w+\sigma_t\boldsymbol\epsilon^w$ and $\mathbf x_t^l=\alpha_t\mathbf x_0^l+\sigma_t\boldsymbol\epsilon^l$. In practice, one may also use shared noise $\boldsymbol\epsilon^w=\boldsymbol\epsilon^l$ as a variance-reduction variant; this does not change the overall DPO-style form of the objective, but slightly alters the stochastic estimator.

Intuitively, minimizing $\tilde{\mathcal L}_{\mathrm{Diff\text{-}DPO}}$ increases the reference-normalized winner-over-loser margin. Equivalently, it encourages the model to reduce denoising error on the winner more than on the loser, measured relative to the frozen reference model. The reference model does not contribute a direct gradient term, but it remains inside the margin within the $\log\sigma(\cdot)$ nonlinearity, so it still shapes the DPO-style per-example weighting.

[^1]: A policy maps a prompt/history (state) to a distribution over responses/actions.

[^2]: In discrete-time diffusion models (e.g., DDPM), evaluating $\pi_{\boldsymbol\theta}(\mathbf x_0| \mathbf c)$ requires marginalizing over the latent reverse trajectory $\mathbf x_{1:T}$.
