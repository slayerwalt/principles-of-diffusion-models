# 8.2 Classifier Guidance {#sec-cg}

## 8.2.1 Foundation of Classifier Guidance {#foundation-of-classifier-guidance}

Let ${\mathbf{c}}$ denote a conditioning variable drawn from a distribution $p({\mathbf{c}})$, such as a class label, caption, or other auxiliary information. Our goal is to draw samples from $p_0({\mathbf{x}} | {\mathbf{c}})$. In diffusion-based conditional generation, we realize this goal by running the reverse-time dynamics whose time marginals are $p_t(\cdot | {\mathbf{c}})$. The drift of these dynamics depends on the conditional score

$$
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t | {\mathbf{c}}), \quad t \in [0,T].
$$

Hence a standard and effective route[^1] is to estimate this quantity.

A fundamental insight, based on Bayes’ rule, is that the conditional score can be decomposed as:

<span id="eq-cg-bayes"></span>

$$
\begin{aligned}
\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t | {\mathbf{c}})
&= \nabla_{{\mathbf{x}}_t} \log \left( \frac{p_t({\mathbf{x}}_t)   p_t({\mathbf{c}} | {\mathbf{x}}_t)}{p({\mathbf{c}})} \right)  \\
&= \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t) + \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{c}} | {\mathbf{x}}_t) - \nabla_{{\mathbf{x}}_t} \log p({\mathbf{c}})  \\
&= \underbrace{\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t)}_{\text{unconditional score}}
+ \underbrace{\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{c}} | {\mathbf{x}}_t)}_{\text{classifier gradient}},
\end{aligned}
\tag{8.2.1}
$$

where $p_t({\mathbf{c}} | {\mathbf{x}}_t)$ indicates a probability of $\mathbf{c}$ conditioned on $\mathbf{x}_{t}$ which predicts the condition ${\mathbf{c}}$ from the noisy input ${\mathbf{x}}_t$ at time $t$.

This decomposition[^2] motivates the *Classifier Guidance* (CG) approach proposed by [Dhariwal and Nichol (2021)](../../references.md#ref-dhariwal2021diffusion). To separate the population identity from the learned classifier used in practice, first consider the oracle quantities. For a guidance scale $\omega\geq0$, define at each fixed time $t$

<span id="eq-cg-density-omega"></span>

$$
\begin{aligned}
\widetilde p_t^{\mathrm{CG},\omega}({\mathbf{x}}_t|{\mathbf{c}})
\propto
p_t({\mathbf{x}}_t)p_t({\mathbf{c}}|{\mathbf{x}}_t)^\omega.
\end{aligned}
\tag{8.2.2}
$$

Its score is

<span id="eq-cg-omega"></span>

$$
\begin{aligned}
{\mathbf{s}}^{\mathrm{CG}}({\mathbf{x}}_t,t,{\mathbf{c}};\omega)
&:=
\nabla_{{\mathbf{x}}_t}
\log\widetilde p_t^{\mathrm{CG},\omega}({\mathbf{x}}_t|{\mathbf{c}})
\\
&=
\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{x}}_t)
+
\omega\nabla_{{\mathbf{x}}_t}\log p_t({\mathbf{c}}|{\mathbf{x}}_t).
\end{aligned}
\tag{8.2.3}
$$

Thus $\omega$ scales the classifier-guidance direction. When $\omega=1$, Bayes’ rule gives $\widetilde p_t^{\mathrm{CG},1}({\mathbf{x}}_t|{\mathbf{c}})=p_t({\mathbf{x}}_t|{\mathbf{c}})$, and [Equation (8.2.3)](02-classifier-guidance.md#eq-cg-omega) recovers the exact conditional score. For a general $\omega$, $\widetilde p_t^{\mathrm{CG},\omega}$ provides a fixed-time tempered interpretation of the guided field; whether it also forms the marginal path of the guided sampler is discussed in [Section 8.4.2](04-training-free-guidance.md#subsec-guided-marginal-path).

The scalar $\omega \ge 0$ modulates the influence of the classifier:

-   $\omega = 1$: recovers the true conditional score $\nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t |{\mathbf{c}})$.

-   $\omega > 1$: amplifies the classifier signal, typically increasing conditional fidelity (often at the expense of diversity).

-   $0 \le \omega < 1$: down-weights the classifier signal, typically increasing sample diversity while weakening conditioning.

**Practical Approximation in CG.** In practice, CG is a training-free method (w.r.t. the diffusion model) for steering a pre-trained unconditional diffusion model,

$$
{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t ,t) \approx \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{x}}_t).
$$

CG is applied only at sampling time, without modifying the diffusion model itself. To enable this, a time-dependent classifier $p_{{\boldsymbol{\psi}}}({\mathbf{c}} |{\mathbf{x}}_t,t)$ is trained separately to predict the condition ${\mathbf{c}}$ from noisy inputs ${\mathbf{x}}_t$ at different noise levels $t$. The classifier is trained in a standard way by minimizing the cross-entropy loss:

<span id="eq-cg-classifier-training"></span>

$$
\begin{aligned}
\mathbb{E}_{t \sim \mathcal{U}[0,T], ({\mathbf{x}}, {\mathbf{c}}) \sim p_{\text{data}},  \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}
\Big[ -\log p_{{\boldsymbol{\psi}}}({\mathbf{c}} | {\mathbf{x}}_t,t) \Big],
\end{aligned}
\tag{8.2.4}
$$

where $({\mathbf{x}},{\mathbf{c}})\sim p_{\text{data}}$ denotes paired labeled data, and ${\mathbf{x}}_t = \alpha_t {\mathbf{x}} + \sigma_t \boldsymbol{\epsilon}$ is the noisy input at time $t$. The classifier must be trained across noise levels and typically receives $t$, or an equivalent noise-level representation, as input so that it can approximate $p_t({\mathbf{c}}|{\mathbf{x}}_t)$ throughout the diffusion path.

After training, the classifier provides scores that serves as a surrogate for the true likelihood gradient:

$$
\nabla_{{\mathbf{x}}_t} \log p_{{\boldsymbol{\psi}}^\times}({\mathbf{c}} |{\mathbf{x}}_t,t) \approx \nabla_{{\mathbf{x}}_t} \log p_t({\mathbf{c}} |{\mathbf{x}}_t).
$$

## 8.2.2 Inference with CG {#inference-with-cg}

At inference time, the learned classifier gradient $\nabla_{{\mathbf{x}}_t}\log p_{{\boldsymbol{\psi}}^\times}({\mathbf{c}}|{\mathbf{x}}_t,t)$ is added to the unconditional score estimate and scaled by $\omega$, yielding a learned approximation to the oracle field in [Equation (8.2.3)](02-classifier-guidance.md#eq-cg-omega):

$$
\begin{aligned}
{\mathbf{s}}^{\mathrm{CG}}_{\mathrm{model}}({\mathbf{x}}_t,t,{\mathbf{c}};\omega)
:=
\underbrace{{\mathbf{s}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}_t,t)}_{\text{uncond. direction}}
+
\omega
\underbrace{\nabla_{{\mathbf{x}}_t}\log
p_{{\boldsymbol{\psi}}^\times}({\mathbf{c}}|{\mathbf{x}}_t,t)}_{\text{guidance direction}}
\approx
{\mathbf{s}}^{\mathrm{CG}}({\mathbf{x}}_t,t,{\mathbf{c}};\omega).
\end{aligned}
$$

The unconditional score estimate in the reverse-time SDE or PF-ODE is then replaced by ${\mathbf{s}}^{\mathrm{CG}}_{\mathrm{model}}({\mathbf{x}}_t,t,{\mathbf{c}};\omega)$, thereby steering the generative trajectory toward samples that align with the condition ${\mathbf{c}}$.

## 8.2.3 Advantages and Limitations {#advantages-and-limitations}

CG provides a simple and flexible mechanism for conditional generation, allowing for explicit control over the strength of conditioning via $\omega$. It can be used with any pre-trained unconditional diffusion model, requiring only an additional classifier for conditioning.

However, the approach has notable limitations:

-   **Training Cost:** The classifier must be trained to operate across all noise levels, which is computationally expensive.

-   **Robustness:** Classifiers must generalize well to severely corrupted inputs ${\mathbf{x}}_t$, especially for large $t$, which can be challenging.

-   **Separate Training:** Since the classifier is trained independently of the diffusion model, it may not align perfectly with the learned data distribution.

[^1]: One could in principle obtain $p_0({\mathbf{x}} | {\mathbf{c}})$ from an unconditional generator via rejection or importance sampling if $p({\mathbf{c}} | {\mathbf{x}})$ were available and well calibrated. This is rarely practical for high-dimensional or rare conditions.

[^2]: In the last identity, $\nabla_{{\mathbf{x}}_t}\log p({\mathbf{c}})=0$ because $p({\mathbf{c}})$ does not depend on ${\mathbf{x}}_t$.
