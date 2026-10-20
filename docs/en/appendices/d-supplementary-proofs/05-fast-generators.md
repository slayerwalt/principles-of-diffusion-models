# D.5 Theoretical Supplement: Learning Fast Diffusion-Based Generators {#app-sec-flow-map}

## D.5.1 Knowledge Distillation Loss as an Instance of the General Framework [Equation (10.1.4)](../../chapters/10-distillation/01-prologue.md#eq-general-cm-loss) {#knowledge-distillation-framework}

We begin with the oracle KD loss

$$
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}({\boldsymbol{\theta}})
={\mathbb{E}}_{{\mathbf{x}}_T\sim p_T}\,
\big\|{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T,T,0)-{\boldsymbol{\Psi}}_{T\to 0}({\mathbf{x}}_T)\big\|_2^2,
$$

where $p_T = p_{\mathrm{prior}}$. For the deterministic ODE flow map ${\boldsymbol{\Psi}}$ (which satisfies the semigroup property and is bijective along trajectories), the marginals satisfy the pushforward identities

$$
p_t
= {\boldsymbol{\Psi}}_{0\to t}\,\sharp\,p_{\mathrm{data}}
= {\boldsymbol{\Psi}}_{T\to t}\,\sharp\,p_{\mathrm{prior}}.
$$

Hence,

$$
{\boldsymbol{\Psi}}_{s\to T}\,\sharp\,p_s=p_T
\quad\text{and}\quad
{\boldsymbol{\Psi}}_{T\to 0}\circ{\boldsymbol{\Psi}}_{s\to T}={\boldsymbol{\Psi}}_{s\to 0}.
$$

Changing variables ${\mathbf{x}}_T={\boldsymbol{\Psi}}_{s\to T}({\mathbf{x}}_s)$ with ${\mathbf{x}}_s\sim p_s$ gives

$$
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}({\boldsymbol{\theta}})
= {\mathbb{E}}_{{\mathbf{x}}_s\sim p_s}
\Big\|{\mathbf{G}}_{\boldsymbol{\theta}} \big({\boldsymbol{\Psi}}_{s\to T}({\mathbf{x}}_s),T,0\big)
- {\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)\Big\|_2^2.
$$

Define the pulled-back student $\widetilde{{\mathbf{G}}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,0):= {\mathbf{G}}_{\boldsymbol{\theta}}({\boldsymbol{\Psi}}_{s\to T}({\mathbf{x}}_s),T,0)$ to express the same loss in the unified flow-map form (at $t=0$):

$$
\begin{aligned}
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}({\boldsymbol{\theta}})
= {\mathbb{E}}_{{\mathbf{x}}_s\sim p_s}
\big\|\widetilde{{\mathbf{G}}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,0)-{\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)\big\|_2^2.
\end{aligned}
$$

This derivation relies on change of variables through the oracle flow and the semigroup property.

$\blacksquare$

**Transport Interpretation.** [Proposition 10.2.1](../../chapters/10-distillation/02-distribution-based-distillation.md#vsd-grad) also admits a distributional interpretation. For

$$
\hat{\mathbf{x}}_t
=
\alpha_t{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})+\sigma_t{\boldsymbol{\epsilon}},
$$

an infinitesimal parameter perturbation induces the data-space displacement $(\partial_{\boldsymbol{\theta}}\hat{\mathbf{x}}_t)\,\mathop{}\!\mathrm{d}{\boldsymbol{\theta}}$. Accordingly, the VSD gradient can be written as

$$
\nabla_{{\boldsymbol{\theta}}}\mathcal{L}_{\mathrm{VSD}}
=
{\mathbb{E}}\!\left[
\omega(t)
\big(\partial_{\boldsymbol{\theta}}\hat{\mathbf{x}}_t\big)^\top
\left(
\nabla_{{\mathbf{x}}}\log p_t^{{\boldsymbol{\theta}}}(\hat{\mathbf{x}}_t)
-
\nabla_{{\mathbf{x}}}\log p_t(\hat{\mathbf{x}}_t)
\right)
\right].
$$

Thus, the score mismatch provides the data-space gradient signal, while the generator Jacobian maps this signal back to parameter space; gradient descent then updates the parameters in the opposite direction. This is the parametric counterpart of the distribution-level transport view in [Section B.3](../b-density-evolution/03-wasserstein-gradient-flows.md#sec-wgf-distribution-level-training).

## D.5.2 [Theorem 11.2.1](../../chapters/11-fast-generators/02-discrete-consistency-models.md#thm-ct-cm): CM Equals CT up to $\mathcal{O}(\Delta s^2)$ {#cm-ct-equivalence}

***Proof:***

**Step 1: DDIM Update (with Oracle Score) Is the Conditional Mean.**

$$
\begin{aligned}
\hat{\mathbf{x}}_{s'}
&:= \frac{\alpha_{s'}}{\alpha_s} {\mathbf{x}}_s
+ \sigma_s^2 \left(\frac{\alpha_{s'}}{\alpha_s}-\frac{\sigma_{s'}}{\sigma_s}\right)\nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s)
\\
&= \frac{\alpha_{s'}}{\alpha_s}\bigl({\mathbf{x}}_s+\sigma_s^2\nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s)\bigr)
- \sigma_{s'}\sigma_s \nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s)
\\
&= \alpha_{s'} {\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_s]
+ \sigma_{s'}\Bigl(\frac{{\mathbf{x}}_s-\alpha_s {\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_s]}{\sigma_s}\Bigr)
\\
&= \alpha_{s'} {\mathbb{E}}[{\mathbf{x}}_0|{\mathbf{x}}_s] + \sigma_{s'} {\mathbb{E}}[{\boldsymbol{\epsilon}}|{\mathbf{x}}_s]
\\
&= {\mathbb{E}}[\alpha_{s'}{\mathbf{x}}_0+\sigma_{s'}{\boldsymbol{\epsilon}}|{\mathbf{x}}_s]
\\
&= {\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s].
\end{aligned}
$$

Here we use Tweedie’s formula

$$
{\mathbb{E}}[{\mathbf{x}}_0 |{\mathbf{x}}_s]=\frac{{\mathbf{x}}_s+\sigma_s^2\nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s)}{\alpha_s},
$$

together with the shared-noise coupling

$$
{\mathbf{x}}_s=\alpha_s{\mathbf{x}}_0+\sigma_s{\boldsymbol{\epsilon}},
\qquad
{\mathbf{x}}_{s'}=\alpha_{s'}{\mathbf{x}}_0+\sigma_{s'}{\boldsymbol{\epsilon}}
\quad (s'=s-\Delta s),
$$

in the third and fourth equalities. We also note that, more precisely, $\hat{\mathbf{x}}_{s'}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s]$ should be read as $\hat{\mathbf{x}}_{s'}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s, s]$, since $s'=s-\Delta s$ is deterministically fixed once $s$ is sampled.

**Step 2: Expand CT Around the Conditional Mean $\hat{\mathbf{x}}_{s'}$.**

Recall that $s$ is sampled first and we set $s':=s-\Delta s$ deterministically.

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CT}}
&= {\mathbb{E}}_{s,{\mathbf{x}}_s}{\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s} \Big[w(s) d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_{s'},s')\big)\Big]
\\
&\overset{\text{(1)}}{=} {\mathbb{E}}_{s,{\mathbf{x}}_s}{\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s} \Big[
w(s) d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\hat{\mathbf{x}}_{s'},s')\big)
\\[-2pt]
&\qquad\qquad\qquad\qquad
+ w(s) \partial_2 d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s),{\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\hat{\mathbf{x}}_{s'},s')\big)\big[\partial_1{\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\hat{\mathbf{x}}_{s'},s')({\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'})\big]
\\[-2pt]
&\qquad\qquad\qquad\qquad
+ w(s) \mathcal O \big(\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2\big)\Big]
\\
&\overset{\text{(2)}}{=} {\mathbb{E}}_{s,{\mathbf{x}}_s} \Big[w(s) d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\hat{\mathbf{x}}_{s'},s')\big)\Big]
\\
&\quad+ {\mathbb{E}}_{s,{\mathbf{x}}_s} \Big[w(s) \partial_2 d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s),{\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\hat{\mathbf{x}}_{s'},s')\big)\big[\partial_1{\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\hat{\mathbf{x}}_{s'},s') {\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s}({\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'})\big]\Big]
\\
&\quad+ {\mathbb{E}}_{s,{\mathbf{x}}_s}{\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s} \Big[w(s) \mathcal{O} \big(\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2\big)\Big]
\\
&\overset{\text{(3)}}{=} {\mathbb{E}}_{s,{\mathbf{x}}_s} \Big[w(s) d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\hat{\mathbf{x}}_{s'},s')\big)\Big]
+ {\mathbb{E}}_{s,{\mathbf{x}}_s}{\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s} \Big[w(s) \mathcal{O} \big(\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2\big)\Big]
\\
&= \mathcal{L}_{\mathrm{CM}}
+ \mathcal{O} \big({\mathbb{E}}\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2\big)
\\
&\overset{\text{(4)}}{=} \mathcal{L}_{\mathrm{CM}} + \mathcal{O}(\Delta s^2).
\end{aligned}
$$

Here, (1) applies a second-order Taylor expansion of

$$
h({\mathbf{x}}') := d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}',s')\big)
$$

around $\hat{\mathbf{x}}_{s'}$ in its second argument. (2) uses the *tower property* (a.k.a. the law of total expectation):

$$
{\mathbb{E}}_{s,{\mathbf{x}}_s}{\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s}[\cdot]
={\mathbb{E}}_{s,{\mathbf{x}}_s}[\cdot],
$$

and notes that inside ${\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s}$, the terms

$$
w(s),\ {\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s),\ \hat{\mathbf{x}}_{s'},\ s'
$$

do not depend on the random variable ${\mathbf{x}}_{s'}$ (since $s' = s-\Delta s$ and $\hat{\mathbf{x}}_{s'}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s]$ are fixed once $(s,{\mathbf{x}}_s)$ is fixed). Hence the only ${\mathbf{x}}_{s'}$-dependence is through $({\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'})$, and the inner expectations reduce to ${\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s}({\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'})$ and ${\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s}\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2$. (3) uses

$$
{\mathbb{E}}[{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s]=\boldsymbol 0
\quad\text{because}\quad
\hat{\mathbf{x}}_{s'}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s].
$$

For (4), note that

$$
{\mathbb{E}}\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2
={\mathbb{E}}\,\operatorname{Var}({\mathbf{x}}_{s'}|{\mathbf{x}}_s,s)
\le {\mathbb{E}}\|{\mathbf{x}}_{s'}-{\mathbf{x}}_s\|^2.
$$

Moreover,

$$
{\mathbf{x}}_{s'}-{\mathbf{x}}_s=(\alpha_{s'}-\alpha_s){\mathbf{x}}_0+(\sigma_{s'}-\sigma_s){\boldsymbol{\epsilon}},
$$

and for a smooth scheduler, $\alpha_{s'}-\alpha_s=\mathcal{O}(\Delta s)$ and $\sigma_{s'}-\sigma_s=\mathcal{O}(\Delta s)$, hence

$$
{\mathbb{E}}\|{\mathbf{x}}_{s'}-{\mathbf{x}}_s\|^2=\mathcal{O}(\Delta s^2),
$$

which yields

$$
{\mathbb{E}}\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2=\mathcal{O}(\Delta s^2).
$$

$\blacksquare$

## D.5.3 [Proposition 11.3.1](../../chapters/11-fast-generators/03-continuous-consistency-models.md#continuous-time-ct): Continuous-Time Limit of the CT Gradient {#continuous-time-ct-gradient}

***Proof:***

We simplify the notation by working with [Equation (11.3.2)](../../chapters/11-fast-generators/03-continuous-consistency-models.md#eq-cm-discrete) in the form

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CM}}^{\Delta s}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-)
:=
{\mathbb{E}} \left[\omega(s)\big\|
{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s) -
{\mathbf{f}}_{{\boldsymbol{\theta}}^-}\big({\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s),\, s-\Delta s\big)
\big\|_2^2\right].
\end{aligned}
$$

For notational simplicity, write

$$
\tilde{\mathbf{x}}_{s-\Delta s}:={\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s).
$$

Define

$$
\delta {\mathbf{f}}
:= {\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s)-{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s).
$$

Then

$$
\begin{aligned}
\big\|{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s)- {\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\tilde{\mathbf{x}}_{s-\Delta s}, s-\Delta s)\big\|_2^2
=
\Big\|\delta {\mathbf{f}}
+
\Big({\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s, s)-{\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\tilde{\mathbf{x}}_{s-\Delta s}, s-\Delta s)\Big)\Big\|_2^2.
\end{aligned}
$$

Applying a Taylor expansion at $({\mathbf{x}}_s,s)$ gives

$$
\begin{aligned}
{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s) -
{\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\tilde{\mathbf{x}}_{s-\Delta s}, s-\Delta s)
=
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\,\Delta s
+ \mathcal{O}(|\Delta s|^2),
\end{aligned}
$$

where we use the first-order expansion of the oracle flow map,

$$
{\mathbf{x}}_s - {\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)
= {\mathbf{v}}^*({\mathbf{x}}_s,s)\,\Delta s + \mathcal{O}(|\Delta s|^2),
$$

together with the total differentiation identity

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
=
\partial_s {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
+
\big(\partial_{{\mathbf{x}}}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\big)\,{\mathbf{v}}^*({\mathbf{x}}_s,s).
$$

Therefore,

$$
\begin{aligned}
\big\|{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s)- {\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\tilde{\mathbf{x}}_{s-\Delta s}, s-\Delta s)\big\|_2^2
=
\|\delta {\mathbf{f}}\|_2^2
+
2\,\delta {\mathbf{f}}^\top \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\,\Delta s
+
\mathcal{O}(|\Delta s|^2).
\end{aligned}
$$

Now differentiate with respect to ${\boldsymbol{\theta}}$. Since ${\boldsymbol{\theta}}^-$ is treated as fixed with respect to $\nabla_{{\boldsymbol{\theta}}}$ (i.e., no gradient is taken through the stop-gradient target), there is no gradient through ${\mathbf{f}}_{{\boldsymbol{\theta}}^-}$. Hence

$$
\begin{aligned}
\nabla_{{\boldsymbol{\theta}}}\mathcal{L}_{\mathrm{CM}}^{\Delta s}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-)
&=
\nabla_{{\boldsymbol{\theta}}}{\mathbb{E}}\!\left[\omega(s)\|\delta {\mathbf{f}}\|_2^2\right]\\
&\qquad\qquad
+
2\,{\mathbb{E}}\!\left[
\omega(s)\,
\nabla_{{\boldsymbol{\theta}}}{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s)^\top
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
\right]\Delta s
+
\mathcal{O}(|\Delta s|^2).
\end{aligned}
$$

At the comparison point where the target network is the stop-gradient copy of the online network, i.e., ${\boldsymbol{\theta}}={\boldsymbol{\theta}}^-$, we have

$$
\delta {\mathbf{f}}
=
{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s)-{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
=
0,
$$

and therefore

$$
\nabla_{{\boldsymbol{\theta}}}{\mathbb{E}}\!\left[\omega(s)\|\delta {\mathbf{f}}\|_2^2\right]=0.
$$

Thus,

$$
\begin{aligned}
\left.
\nabla_{{\boldsymbol{\theta}}}\mathcal{L}_{\mathrm{CM}}^{\Delta s}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-)
\right|_{{\boldsymbol{\theta}}={\boldsymbol{\theta}}^-}
&=
\left.
2\,{\mathbb{E}}\!\left[
\omega(s)\,
\nabla_{{\boldsymbol{\theta}}}{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s)^\top
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
\right]
\right|_{{\boldsymbol{\theta}}={\boldsymbol{\theta}}^-}\Delta s
+
\mathcal{O}(|\Delta s|^2).
\end{aligned}
$$

Dividing by $\Delta s$ and taking the limit yields

$$
\begin{aligned}
\left.
\lim_{\Delta s\to 0}\frac{1}{\Delta s}\nabla_{{\boldsymbol{\theta}}}\mathcal{L}_{\mathrm{CM}}^{\Delta s}({\boldsymbol{\theta}},{\boldsymbol{\theta}}^-)
\right|_{{\boldsymbol{\theta}}={\boldsymbol{\theta}}^-}
&=
\left.
\nabla_{{\boldsymbol{\theta}}}{\mathbb{E}}\!\left[
2\,\omega(s)\,
{\mathbf{f}}_{{\boldsymbol{\theta}}}^\top({\mathbf{x}}_s,s)\,
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
\right]
\right|_{{\boldsymbol{\theta}}={\boldsymbol{\theta}}^-}.
\end{aligned}
$$

This proves the identity at ${\boldsymbol{\theta}}={\boldsymbol{\theta}}^-$.

$\blacksquare$
