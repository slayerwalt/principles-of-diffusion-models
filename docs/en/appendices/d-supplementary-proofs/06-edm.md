# D.6 (Optional) Elucidating Diffusion Model (EDM) {#sec-edm}

We introduce specific criteria for neural network parameterization design in the ${\mathbf{x}}$-prediction model, as proposed in Elucidating Diffusion Models (EDM) ([Karras et al., 2022](../../references.md#ref-karras2022elucidating)). EDM provides a simple recipe that makes the training process easier to optimize and more reliable. The ${\mathbf{x}}$-prediction model is expressed as a time-dependent skip connection combined with a scaled residual ([Equation (D.6.1)](06-edm.md#eq-d-parametrization)). The central idea is to normalize both the inputs and the regression targets to unit variance at all times, and to adjust the skip path so that residual errors are not amplified as time evolves. This recipe has become a widely adopted default in diffusion model implementations and extends naturally to flow map learning, especially the family of Consistency Models.

## D.6.1 Criteria for Neural Network ${\mathbf{x}}_{\boldsymbol{\phi}}$ Design {#subsec-d-design-edm}

EDM considers a parametrization of the ${\mathbf{x}}$-prediction model, denoted with a slight abuse of notation as ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t)$[^1], in the following form:

<span id="eq-d-parametrization"></span>

$$
\begin{aligned}
{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t) := c_{\mathrm{skip}}(t) {\mathbf{x}} + c_{\mathrm{out}}(t) {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t) {\mathbf{x}}, c_{\mathrm{noise}}(t)\right).
\end{aligned}
\tag{D.6.1}
$$

Here, $c_{\mathrm{skip}}(t)$, $c_{\mathrm{out}}(t)$, $c_{\mathrm{in}}(t)$, and $c_{\mathrm{noise}}(t)$ are time-dependent functions. They are chosen to enhance stability and performance during training, based on practical considerations that will be introduced shortly.

Plugging this in [Equation (6.2.5)](../../chapters/06-unified-perspective/02-training-objectives.md#eq-general-dm-loss-d), then the objective function becomes after straightforward algebraic manipulation[^2]:

<span id="eq-loss-f"></span>

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)c_{\mathrm{out}}^2(t)\left\lVert {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t){\mathbf{x}}_t, c_{\mathrm{noise}}(t)\right) - {\mathbf{x}}_{\mathrm{tgt}}(t)\right\rVert_2^2\right].
\end{aligned}
\tag{D.6.2}
$$

Here, the regression target ${\mathbf{x}}_{\mathrm{tgt}}(t)$ is obtained as:

$$
{\mathbf{x}}_{\mathrm{tgt}}(t) = \frac{\left(1-c_{\mathrm{skip}}(t)\alpha_t\right){\mathbf{x}}_0-c_{\mathrm{skip}}(t)\sigma_t \boldsymbol{\epsilon}}{c_{\mathrm{out}}(t)}.
$$

By incorporating the standard deviation of $p_{\text{data}}$, denoted as $\sigma_{\mathrm{d}}$, EDM proposes the following design criterion for network parameterization, which may be described as the *unit variance criterion*.

**Unit Variance of Input.**

$$
\begin{aligned}
&\text{Var}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[ c_{\mathrm{in}}(t){\mathbf{x}}_t\right] = 1
    \\ \Longleftrightarrow &~c_{\mathrm{in}}^2(t) \text{Var}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[ \alpha_t {\mathbf{x}}_0 +\sigma_t \boldsymbol{\epsilon} \right] = 1
    \\ \Longleftrightarrow &~c_{\mathrm{in}}(t)  = \frac{1}{\sqrt{\sigma_{\mathrm{d}}^2 \alpha_t^2 + \sigma_t^2}},
\end{aligned}
$$

taking the positive root.

**Unit Variance of Training Target.**

<span id="eq-c-out"></span>

$$
\begin{aligned}
&\text{Var}_{{\mathbf{x}}_0,\boldsymbol{\epsilon}}\left[ {\mathbf{x}}_{\mathrm{tgt}}(t) \right] = 1
    \\ \Longleftrightarrow &~c_{\mathrm{out}}^2(t)  = \left(1-c_{\mathrm{skip}}(t)\alpha_t\right)^2 \sigma_{\mathrm{d}}^2 + c_{\mathrm{skip}}^2(t)\sigma_t^2,
\end{aligned}
\tag{D.6.3}
$$

with centered data (${\mathbb{E}}[{\mathbf{x}}_0]=\boldsymbol{0}$).

**Minimize the Error Amplification from ${\mathbf{F}}_{\boldsymbol{\phi}}$ to ${\mathbf{x}}_{\boldsymbol{\phi}}$.** EDM aims to mitigate the amplification of the network’s learning error from ${\mathbf{F}}_{\boldsymbol{\phi}}$ to ${\mathbf{x}}_{\boldsymbol{\phi}}$. This is achieved by selecting $c_{\mathrm{skip}}$ to minimize $c_{\mathrm{out}}$:

$$
c_{\mathrm{skip}}^* \in \operatorname*{arg\,min}_{c_{\mathrm{skip}}}  c_{\mathrm{out}}^2.
$$

Using the standard approach of solving $\frac{\partial c_{\mathrm{out}}}{\partial c_{\mathrm{skip}}} = 0$ for the critical point $c_{\mathrm{skip}}^*$, we obtain

$$
\begin{aligned}
c_{\mathrm{skip}}^*(t) = \frac{\alpha_t \sigma_{\mathrm{d}}^2 }{\alpha_t^2 \sigma_{\mathrm{d}}^2  + \sigma_t^2}.
\end{aligned}
$$

Substituting this into [Equation (D.6.3)](06-edm.md#eq-c-out), the optimal value is given by

$$
\begin{aligned}
c_{\mathrm{out}}^*(t) = \pm \frac{\sigma_t \sigma_{\mathrm{d}} }{\sqrt{\alpha_t^2 \sigma_{\mathrm{d}}^2  + \sigma_t^2}}.
\end{aligned}
$$

By convention, we use the nonnegative branch for the output scale:

$$
c_{\mathrm{out}}^*(t) = \frac{\sigma_t \sigma_{\text{d}}}
{\sqrt{\alpha_t^2 \sigma_{\text{d}}^2 + \sigma_t^2}}
\;\;\;(\ge 0),
$$

which ensures $c_{\mathrm{out}}(0)=0$ and $c_{\mathrm{out}}(t)\to\sigma_{\text{d}}$ as $\sigma_t$ is sufficiently large, yielding the intuitive limits ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,0)\approx {\mathbf{x}}_t$ and ${\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t,t)\approx \sigma_{\text{d}}\,{\mathbf{F}}_{\boldsymbol{\phi}}(\cdot)$.

We summarize these coefficients as follows:

Denoting $R_t := \alpha_t^2 \sigma_{\mathrm{d}}^2+ \sigma_t^2$, we have the following selections:

<span id="eq-edm-coeff"></span>

$$
\begin{aligned}
            c_{\mathrm{in}}(t)  = \frac{1}{\sqrt{R_t}}, \quad
            c_{\mathrm{skip}}(t) = \frac{\alpha_t \sigma_{\mathrm{d}}^2 }{R_t}, \quad
            c_{\mathrm{out}}(t) =  \frac{\sigma_t \sigma_{\mathrm{d}} }{\sqrt{R_t}}.
        \end{aligned}
\tag{D.6.4}
$$

With [Equation (D.6.4)](06-edm.md#eq-edm-coeff), the regression target ${\mathbf{x}}_{\mathrm{tgt}}(t)$ simplifies to

$$
{\mathbf{x}}_{\mathrm{tgt}}(t) = \frac{1}{\sigma_{\mathrm{d}}}\frac{\sigma_t {\mathbf{x}}_0 - \alpha_t \sigma_{\mathrm{d}}^2\boldsymbol{\epsilon} }{\sqrt{R_t}}.
$$

Additionally, substituting these expressions into [Equation (D.6.1)](06-edm.md#eq-d-parametrization) and [Equation (D.6.2)](06-edm.md#eq-loss-f) allows us to simplify both the parametrization and the loss function, yielding:

$$
\begin{aligned}
{\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t)=\frac{\alpha_t \sigma_{\mathrm{d}}^2 }{R_t} {\mathbf{x}} + \frac{\sigma_t \sigma_{\mathrm{d}} }{\sqrt{R_t}}{\mathbf{F}}_{\boldsymbol{\phi}}\left(\frac{1}{\sqrt{R_t}}{\mathbf{x}}, c_{\mathrm{noise}}(t)\right),
\end{aligned}
$$

and

$$
\begin{aligned}
\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)\frac{\sigma_t^2  }{R_t}\left\lVert \sigma_{\mathrm{d}} {\mathbf{F}}_{\boldsymbol{\phi}}\left(\frac{1}{\sqrt{R_t}}{\mathbf{x}}_t, c_{\mathrm{noise}}(t)\right) - \left(\frac{\sigma_t {\mathbf{x}}_0 - \alpha_t \sigma_{\mathrm{d}}^2\boldsymbol{\epsilon} }{\sqrt{R_t}}\right)\right\rVert_2^2\right].
\end{aligned}
$$

With this parameterization and the conditions $\alpha_0 \approx 1$ and $\sigma_0 \approx 0$, we observe that

$$
c_{\mathrm{skip}}(0) \approx 1 \quad \text{and} \quad c_{\mathrm{out}}(0) \approx 0.
$$

**Selection of $c_{\mathrm{noise}}(t)$.** It provides a noise-level embedding to ${\mathbf{F}}_{\boldsymbol{\phi}}$; any one-to-one mapping of the noise level $\sigma_t$ (e.g., $c_{\mathrm{noise}}(t)=\log \sigma_t$) is suitable.

## D.6.2 A Common EDM Special Case: $\alpha_t = 1$, $\sigma_t = t$ {#common-edm-special-case}

We consider the simplified noise schedule used in EDM, where $\alpha_t = 1$ and $\sigma_t = t$, which also appears in the discussion of CTM in [Section 11.4](../../chapters/11-fast-generators/04-consistency-trajectory-models.md#sec-ctm). Under this setting, the forward process becomes

$$
{\mathbf{x}}_t = {\mathbf{x}}_0 + t\boldsymbol{\epsilon},
\quad
\text{with }
{\mathbf{x}}_0 \sim p_{\text{data}},
\,\,
\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}),
$$

corresponding to the perturbation kernel

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
=
\mathcal{N}({\mathbf{x}}_t; {\mathbf{x}}_0, t^2{\mathbf{I}}).
$$

The resulting marginal density is

$$
p_t({\mathbf{x}})
=
\int
\mathcal{N}({\mathbf{x}};{\mathbf{x}}_0,t^2{\mathbf{I}})
p_{\mathrm{data}}({\mathbf{x}}_0)
\mathop{}\!\mathrm{d}{\mathbf{x}}_0
=
\bigl[
p_{\mathrm{data}}
*
\mathcal{N}(\boldsymbol{0},t^2{\mathbf{I}})
\bigr]({\mathbf{x}}).
$$

Thus, the exact terminal marginal is $p_T=p_{\mathrm{data}}*\mathcal{N}(\boldsymbol{0},T^2{\mathbf{I}})$. When $T$ is sufficiently large relative to the scale of the data, this marginal is commonly approximated by $\mathcal{N}(\boldsymbol{0},T^2{\mathbf{I}})$, which is used as the sampling prior:

$$
p_{\mathrm{prior}}({\mathbf{x}}_T)
:=
\mathcal{N}({\mathbf{x}}_T;\boldsymbol{0},T^2{\mathbf{I}}).
$$

Under this setup, the PF-ODE based on ${\mathbf{x}}$-prediction ${\mathbf{x}}_{\boldsymbol{\phi}^\times}$ (see [Equation (6.3.2)](../../chapters/06-unified-perspective/03-equivalent-parameterizations.md#eq-clean-pf-ode)) simplifies to

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{x}}(t)}{\mathop{}\!\mathrm{d} t} = \frac{{\mathbf{x}}(t) - {\mathbf{x}}_{\boldsymbol{\phi}^\times}({\mathbf{x}}(t), t)}{t}.
\end{aligned}
$$

Substituting this formulation into [Equation (D.6.4)](06-edm.md#eq-edm-coeff), the neural network parameterization coefficients become

<span id="eq-edm-simple-coefficients"></span>

$$
\begin{aligned}
c_{\mathrm{in}}(t) = \frac{1}{\sqrt{\sigma_{\mathrm{d}}^2 + t^2}}, \quad
    c_{\mathrm{skip}}(t) = \frac{\sigma_{\mathrm{d}}^2}{\sigma_{\mathrm{d}}^2 + t^2}, \quad
    c_{\mathrm{out}}(t) = \pm \frac{t \sigma_{\mathrm{d}}}{\sqrt{\sigma_{\mathrm{d}}^2 + t^2}}.
\end{aligned}
\tag{D.6.5}
$$

From these expressions, we observe:

-   When $t \approx 0$, the noise level is negligible, so $c_{\mathrm{skip}} \approx 1$ and $c_{\mathrm{out}} \approx 0$. In this limit, the skip path dominates and the network essentially passes through its input,

    $$
    {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t) \approx {\mathbf{x}}.
    $$

-   When $t \gg 0$, the input is heavily corrupted by noise, so $c_{\mathrm{skip}} \approx 0$ and $c_{\mathrm{out}} \approx \sigma_{\mathrm{d}}$. In this regime, the skip path vanishes and the model output is determined entirely by the learned residual,

    $$
    {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}, t) \approx \sigma_{\mathrm{d}} {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t){\mathbf{x}}, c_{\mathrm{noise}}(t)\right),
    $$

    meaning that the network ${\mathbf{F}}_{\boldsymbol{\phi}}$ predicts a scaled proxy of the clean signal from a normalized noisy input; at high noise levels the model output is therefore determined entirely by the learned denoising function.

In short, the parameterization smoothly interpolates from an identity map at small $t$ to a scaled residual predictor on standardized inputs at large $t$.

[^1]: As discussed, all four prediction types are equivalent and can be reduced to the ${\mathbf{x}}$-prediction case. EDM adopts this formulation, which is both well-studied and naturally aligned with the goal of generating clean samples of flow map models.

[^2]: We start from the ${\mathbf{x}}$-prediction diffusion loss by substituting the parameterization given in [Equation (D.6.1)](06-edm.md#eq-d-parametrization):

    $$
    \begin{aligned}
          &~\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)\left\lVert {\mathbf{x}}_{\boldsymbol{\phi}}({\mathbf{x}}_t, t) - {\mathbf{x}}_0\right\rVert_2^2\right]
          \\= &~\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)\Big\Vert c_{\mathrm{skip}}(t) \underbrace{(\alpha_t {\mathbf{x}}_0+\sigma_t{\boldsymbol{\epsilon}})}_{{\mathbf{x}}_t} + c_{\mathrm{out}}(t) {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t) {\mathbf{x}}_t, c_{\mathrm{noise}}(t)\right) - {\mathbf{x}}_0\Big\Vert_2^2\right]
            \\= &~\mathbb{E}_{{\mathbf{x}}_0, \boldsymbol{\epsilon}, t} \left[\omega(t)c_{\mathrm{out}}^2(t)\Big\Vert  {\mathbf{F}}_{\boldsymbol{\phi}}\left(c_{\mathrm{in}}(t) {\mathbf{x}}, c_{\mathrm{noise}}(t)\right) - \underbrace{\left(\frac{\left(1-c_{\mathrm{skip}}(t)\alpha_t\right){\mathbf{x}}_0-c_{\mathrm{skip}}(t)\sigma_t \boldsymbol{\epsilon}}{c_{\mathrm{out}}(t)}\right)}_{{\mathbf{x}}_{\mathrm{tgt}}(t)}\Big\Vert_2^2\right]
            \\ = &~\href{../06-edm/#eq-loss-f}{\text{Equation (D.6.2)}}.
        \end{aligned}
    $$
