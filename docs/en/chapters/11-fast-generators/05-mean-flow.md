# 11.5 General Flow Map: Mean Flow {#sec-mf}

Just as diffusion models admit many equivalent parameterizations and training objectives, a general flow map ${\boldsymbol{\Psi}}_{s\to t}$ can also be learned in multiple plausible ways. In this section, we introduce *Mean Flow* (MF) ([Geng et al., 2025](references.md#ref-geng2025mean)), a later representative of the general flow map family ${\boldsymbol{\Psi}}_{s\to t}$ that illustrates an alternative yet principled perspective on how such maps can be effectively learned.

## 11.5.1 Modeling and Training of Mean Flow {#modeling-and-training}

In contrast to CM and CTM, which build on the EDM framework, MF is based on the flow matching formulation ($\alpha_t = 1-t$ and $\sigma_t = t$ for $t\in[0,1]$). Rather than directly parameterizing the flow map, MF learns the *average drift* over an interval $[t,s]$ (with $t<s$):

$$
{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s, t)  \approx
{\mathbf{h}}^*({\mathbf{x}}_s, s, t) := \frac{1}{t-s}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u.
$$

The corresponding oracle loss is

<span id="eq-mf-oracle"></span>

$$
\begin{aligned}
\mathbb{E}_{t<s} \mathbb{E}_{{\mathbf{x}}_s\sim p_s}\Big[w(s)
\|{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t) - {\mathbf{h}}^*({\mathbf{x}}_s,s,t)\|_2^2\Big].
\end{aligned}
\tag{11.5.1}
$$

In particular, when $s \to t$, the loss function reduces to the flow matching loss:

<span id="eq-mf-to-fm"></span>

$$
\begin{aligned}
\mathbb{E}_{t} \mathbb{E}_{{\mathbf{x}}_t\sim p_t}\Big[w(t)
\|{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_t,t,t) - {\mathbf{v}}^*({\mathbf{x}}_t,t)\|_2^2\Big],
\end{aligned}
\tag{11.5.2}
$$

learning the instantaneous velocity. We will see later in [Section 11.5.3](05-mean-flow.md#subsec-equivalence-ctm-mf) that MF remains consistent with the general objective in [Equation (10.1.4)](01-prologue.md#eq-general-cm-loss-rep), but approaches it from a different (while equivalent) perspective. Since the oracle regression target ${\mathbf{h}}^*({\mathbf{x}}_s,s,t)$ does not admit a closed form in general, MF constructs a surrogate by exploiting an identity obtained from differentiating

$$
(t-s)\,{\mathbf{h}}^*({\mathbf{x}}_s,s,t) = \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u
$$

with respect to $s$. This yields the *MF identity*:

<span id="eq-mf-identity"></span>

$$
\begin{aligned}
    {\mathbf{h}}^*({\mathbf{x}}_s,s,t)
&= {\mathbf{v}}^*({\mathbf{x}}_s,s) - (s-t) \tfrac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{h}}^*({\mathbf{x}}_s,s,t) \\
&= {\mathbf{v}}^*({\mathbf{x}}_s,s) - (s-t)\Big[(\partial_{{\mathbf{x}}}{\mathbf{h}}^*)({\mathbf{x}}_s,s,t) {\mathbf{v}}^*({\mathbf{x}}_s,s)
+ \partial_s {\mathbf{h}}^*({\mathbf{x}}_s,s,t)\Big],
\end{aligned}
\tag{11.5.3}
$$

where the second line applies the chain rule together with

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{x}}_s = {\mathbf{v}}^*({\mathbf{x}}_s,s).
$$

Motivated by this identity, MF replaces the intractable oracle with a stop-gradient surrogate, leading to the practical training objective

<span id="eq-mf-practice"></span>

$$
\begin{aligned}
\mathcal{L}_{\mathrm{MF}}({\boldsymbol{\theta}})
:= \mathbb{E}_{t<s} \mathbb{E}_{{\mathbf{x}}_s\sim p_s}\Big[w(s)\,
\|{\mathbf{h}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s,t) - {\mathbf{h}}_{{\boldsymbol{\theta}}^-}^{\mathrm{tgt}}({\mathbf{x}}_s,s,t)\|_2^2\Big],
\end{aligned}
\tag{11.5.4}
$$

where the regression target is defined as

$$
\begin{aligned}
{\mathbf{h}}_{{\boldsymbol{\theta}}^-}^{\mathrm{tgt}}&({\mathbf{x}}_s,s,t)
:= \\ &{\mathbf{v}}^*({\mathbf{x}}_s,s) - (s-t)\Big[\underbrace{(\partial_{{\mathbf{x}}}{\mathbf{h}}_{{\boldsymbol{\theta}}^-})({\mathbf{x}}_s,s,t) {\mathbf{v}}^*({\mathbf{x}}_s,s)
+ \partial_s {\mathbf{h}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s,t)}_{\mathrm{JVP}}\Big].
\end{aligned}
$$

In practice, the oracle velocity ${\mathbf{v}}^*$ cannot be computed in closed form and must instead be approximated. Two common strategies are available: relying on a pre-trained diffusion model (*distillation*) or constructing a direct estimator from data (*training from scratch*). Regardless of the choice, one ultimately needs to compute a *Jacobian–vector product* (JVP) of the target network ${\mathbf{h}}_{{\boldsymbol{\theta}}^-}$:

$$
\left[\partial_{\mathbf{x}} {\mathbf{h}}_{{\boldsymbol{\theta}}^-}, \partial_s {\mathbf{h}}_{{\boldsymbol{\theta}}^-}, \partial_t {\mathbf{h}}_{{\boldsymbol{\theta}}^-}\right]^\top \cdot \left[{\mathbf{v}}^*, 1, 0\right]
$$

**Distillation.** Use a pre-trained diffusion model with a flow matching backbone, ${\mathbf{v}}_{{\boldsymbol{\phi}}^\times} \approx {\mathbf{v}}^*$.

**Training from scratch.** Use the one point conditional velocity $\alpha_s'{\mathbf{x}}_0 + \sigma_s'{\boldsymbol{\epsilon}}$, obtained from the forward noise injection ${\mathbf{x}}_s=\alpha_s{\mathbf{x}}_0+\sigma_s{\boldsymbol{\epsilon}}$ with ${\boldsymbol{\epsilon}}\sim\mathcal N(\boldsymbol{0},{\mathbf{I}})$. This gives an unbiased single sample estimate of the instantaneous drift at level $s$ when evaluated at paired $({\mathbf{x}}_0,{\boldsymbol{\epsilon}})$.

## 11.5.2 Sampling of Mean Flow {#sampling}

Once a MF ${\mathbf{h}}_{{\boldsymbol{\theta}}^\times}$ is trained, it naturally recovers a proxy of the flow map. For any starting point ${\mathbf{x}}_s$, the map from $s$ to $t$ is (approximately) given by

$$
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s) = {\mathbf{x}}_s + (t-s)\,{\mathbf{h}}^*({\mathbf{x}}_s,s,t)
  \approx   {\mathbf{x}}_s + (t-s)\,{\mathbf{h}}_{{\boldsymbol{\theta}}^\times}({\mathbf{x}}_s,s,t).
$$

This enables both one-step and multi-step sampling. For example, drawing ${\mathbf{x}}_T\sim p_{\mathrm{prior}}$, the one-step generation of a clean sample is

$$
{\mathbf{x}}_0  \leftarrow  {\mathbf{x}}_T - T\,{\mathbf{h}}_{{\boldsymbol{\theta}}^\times}({\mathbf{x}}_T,T,0).
$$

Alternatively, multi-step generation can be performed by preparing a time grid and applying the map sequentially, in the same time-stepping manner used in CTM. Since MF learns a general flow map, it also supports $\gamma$-sampling as in CTM, where a controllable hyperparameter $\gamma$ injects stochasticity into the sampling process.

## 11.5.3 Relationship between CTM and MF {#subsec-equivalence-ctm-mf}

At first sight CTM and MF may appear unrelated. In fact, both provide different parameterizations of the same oracle flow map ${\boldsymbol{\Psi}}_{s\to t}$, and their corresponding squared oracle regression errors are related by a time-dependent weighting ([Hu et al., 2025](references.md#ref-hu2025cmt)).

**Relationship of Parameterizations.** Both methods operate under the same general framework but represent the learned function in distinct ways. The flow map can be written equivalently as

$$
\begin{aligned}
{\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)
& = {\mathbf{x}}_s + \int_s^t {\mathbf{v}}^*({\mathbf{x}}_u, u)\mathop{}\!\mathrm{d} u \\
&= \frac{t}{s}{\mathbf{x}}_s
   + \frac{s-t}{s}\underbrace{\Bigg[{\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u\Bigg]}_{\approx\,{\mathbf{g}}_{\boldsymbol{\theta}}} \\
&= {\mathbf{x}}_s + (t-s)\underbrace{\Bigg[\frac{1}{t-s}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\,\mathop{}\!\mathrm{d} u\Bigg]}_{\approx\,{\mathbf{h}}_{\boldsymbol{\theta}}}.
\end{aligned}
$$

Here, the first is the definition of the flow map, the second form highlights the CTM parametrization through ${\mathbf{g}}_{\boldsymbol{\theta}}$ (see [Equation (11.4.1)](04-consistency-trajectory-models.md#eq-ctm-g-oracle) and [Equation (11.4.2)](04-consistency-trajectory-models.md#eq-ctm-para-1)), while the last highlights the MF parametrization through ${\mathbf{h}}_{\boldsymbol{\theta}}$.

**Relationship of Training Loss.** Given the above reinterpretation of the oracle flow map ${\boldsymbol{\Psi}}_{s\to t}$ in terms of the CTM parametrization

$$
{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t) \approx {\mathbf{g}}^*({\mathbf{x}}_s,s,t):={\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u
$$

and the MF parametrization

$$
{\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t) \approx {\mathbf{h}}^*({\mathbf{x}}_s,s,t):=\frac{1}{t-s}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u,
$$

we now show that the training losses of CTM and MF are in fact equivalent.

Consider the relation

$$
{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t) := {\mathbf{x}}_s - s {\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t),
$$

and take $d({\mathbf{x}},{\mathbf{y}}):=\|{\mathbf{x}}-{\mathbf{y}}\|^2$ as an example. Substituting into [Equation (10.1.4)](01-prologue.md#eq-general-cm-loss-rep) and viewing ${\mathbf{G}}_{\boldsymbol{\theta}}$ as CTM’s flow-map parameterization ([Equation (11.4.2)](04-consistency-trajectory-models.md#eq-ctm-para-1)) gives

<span id="eq-g-para"></span>
<span id="eq-h-para"></span>

$$
\begin{align*}
&~d\big({\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t), {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\big)  \\ = &~\left\lVert{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t) - {\boldsymbol{\Psi}}_{s\to t}({\mathbf{x}}_s)\right\rVert^2  \\ = &~\left\lVert\left(\frac{t}{s}{\mathbf{x}}_s + \frac{s-t}{s} {\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\right)-\left(\frac{t}{s}{\mathbf{x}}_s + \frac{s-t}{s}\Big[{\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\Big]\right)\right\rVert^2  \\ = &~\left(\frac{s-t}{s}\right)^2\left\lVert{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)-\left({\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\right)\right\rVert^2 \tag{11.5.5} \\ = &~\left(\frac{s-t}{s}\right)^2\left\lVert\left({\mathbf{x}}_s - s {\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\right)-\left({\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\right)\right\rVert^2  \\ = &~\left(\frac{s-t}{s}\right)^2\left\lVert\left({\mathbf{x}}_s - s {\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\right)-\left({\mathbf{x}}_s + \frac{s}{s-t}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u) \mathop{}\!\mathrm{d} u\right)\right\rVert^2  \\ = &~\left(s-t\right)^2\left\lVert {\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)- \left(\frac{1}{t-s}\int_s^t {\mathbf{v}}^*({\mathbf{x}}_u,u)\mathop{}\!\mathrm{d} u \right)\right\rVert^2 \tag{11.5.6}
\end{align*}
$$

Hence,

$$
\begin{aligned}
\frac{1}{s^2} \Bigg\|{\mathbf{g}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)-{\mathbf{g}}^*({\mathbf{x}}_s,s,t) \Bigg\|^2 = \Bigg\|{\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)-{\mathbf{h}}^*({\mathbf{x}}_s,s,t) \Bigg\|^2.
\end{aligned}
$$

Putting aside the specific algorithm used to learn the flow map, the CTM and MF losses share the same mathematical form, differing only by a weighting function. Moreover, setting $t=0$ in either case recovers the CM setting (${\boldsymbol{\Psi}}_{s\to 0}$), where each state maps directly to the clean data.

**Auxiliary Loss in Practice.** In CTM, training is performed with the consistency loss in [Equation (11.4.7)](04-consistency-trajectory-models.md#eq-ctm-consist-scratch) jointly with its self-defined diffusion model loss in [Equation (11.4.8)](04-consistency-trajectory-models.md#eq-ctm-dsm). A similar strategy is adopted in MF. As shown in [Equation (11.5.2)](05-mean-flow.md#eq-mf-to-fm), when $s \to t$, the MF loss reduces to the standard flow matching objective. In practice, MF controls the ratio between pairs with $s \neq t$ and those with $s = t$; consequently, the overall optimization becomes a mixture of the MF objective in [Equation (11.5.4)](05-mean-flow.md#eq-mf-practice) and the flow matching objective in [Equation (11.5.2)](05-mean-flow.md#eq-mf-to-fm).

Both parametrizations are able to provide a smooth transition from diffusion-model training, which learns instantaneous velocity with a fixed regression target, to flow-map learning, which employs a stop-gradient pseudo-regression target.

**Both CTM and MF Parameterizations Enable Flexible Inference.** Both CTM (${\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)$) and MF (${\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)$) aim to approximate the underlying flow map ${\boldsymbol{\Psi}}_{s\to t}$:

$$
{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\approx{\boldsymbol{\Psi}}_{s\to t},\quad  \text{and}
\quad
{\mathbf{x}}_s+(t-s){\mathbf{h}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,t)\approx{\boldsymbol{\Psi}}_{s\to t}.
$$

Since both models learn an explicit mapping between any two time steps, they naturally support CTM’s $\gamma$-sampling and remain compatible with inference techniques originally developed for diffusion models, such as guidance ([Chapter 8](../08-guidance/index.md#ch-guidance)), exact likelihood computation ([Equation (4.3.7)](../04-score-sde-framework/03-training-and-sampling.md#eq-score-sde-likelihood)), and accelerated sampling with higher-order solvers ([Chapter 9](../09-numerical-solvers/index.md#ch-numerical-solvers)). This compatibility arises because their parameterizations recover the instantaneous diffusion drift in the infinitesimal limit $t\to s$:

$$
{\mathbf{g}}^*({\mathbf{x}}_s,s,s)={\mathbf{x}}_s-s\,{\mathbf{v}}^*({\mathbf{x}}_s,s),
\quad  \text{and}
\quad
{\mathbf{h}}^*({\mathbf{x}}_s,s,s)={\mathbf{v}}^*({\mathbf{x}}_s,s).
$$

This property is not shared by specialized flow map formulations ${\boldsymbol{\Psi}}_{s\to0}$, such as those in the CM family. Thus, both CTM and MF can be regarded as flexible and general flow map formulations that generalize diffusion-based inference to direct time-to-time mappings.

**Conclusion.**

The relationship between CTM and MF is analogous to the relationship between different parametrizations of diffusion models ([Section 6.3](../06-unified-perspective/03-equivalent-parameterizations.md#sec-equivalent-parametrizations)): at the oracle level, they encode the same underlying flow map. Under squared-error regression and the parametrization relation above, their oracle regression errors differ only by a known time-dependent scaling. Their practical training objectives, however, remain different because they use different surrogate targets and optimization procedures.

Although CTM and MF are related in this way, they differ conceptually and algorithmically in how the target is learned. The CTM target is defined in the “integral” form, as it acts directly on the flow map. The MF target, on the other hand, is defined in the “differential” form, as it relies on finite differences of the flow map. Because a target in the integral form is not directly tractable, CTM must solve an ODE during training. In contrast, MF leverages the MF identity (see [Equation (11.5.3)](05-mean-flow.md#eq-mf-identity)) to construct a tractable target in the differential form. This relationship is analogous to that between energy-based and score-based methods: the former learns a probability density itself (in the integral form), while the latter matches the gradient of that density (in the differential form). In practice, CTM and MF may behave differently due to factors such as loss weighting, network architecture, and optimization dynamics, which can cause one method to perform better than the other under certain conditions.

This perspective suggests that CTM and MF are not the only viable formulations. Other parameterizations of the flow map could also enable efficient and stable training, potentially leading to new standalone generative models. Exploring these alternatives may further enrich the landscape of diffusion models and their flow map extensions, pushing the boundaries of few-step generation.
