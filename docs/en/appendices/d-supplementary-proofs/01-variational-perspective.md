# D.1 Variational Perspective {#app-sec-var-proof}

## D.1.1 [Theorem 2.2.1](../../chapters/02-variational-perspective/02-ddpm.md#thm-equiv-marginal-kl): Equivalence Between Marginal and Conditional KL Minimization {#subsec-proof-equiv-marginal-kl}

***Proof.*** **Derivation of [Equation (2.2.3)](../../chapters/02-variational-perspective/02-ddpm.md#eq-kl-matching).**

We start by expanding the right-hand side expectation:

$$
\begin{aligned}
&\mathbb{E}_{p({\mathbf{x}}_0, {\mathbf{x}}_i)}\left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0) \| p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right]
    \\
    =& \int \int p({\mathbf{x}}_0, {\mathbf{x}}_i)
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0) \| p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \mathop{}\!\mathrm{d}{\mathbf{x}}_i.
\end{aligned}
$$

By the definition of KL divergence,

$$
\begin{aligned}
&\mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0) \| p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    = \int p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \mathop{}\!\mathrm{d}{\mathbf{x}}_{i-1}.
\end{aligned}
$$

Substituting this into the expectation, we have

$$
\begin{aligned}
&\int \int \int p({\mathbf{x}}_0, {\mathbf{x}}_i)  p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \mathop{}\!\mathrm{d}{\mathbf{x}}_{i-1} \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \mathop{}\!\mathrm{d}{\mathbf{x}}_i.
\end{aligned}
$$

Using the chain rule of probability,

$$
\begin{aligned}
p({\mathbf{x}}_0, {\mathbf{x}}_i)
    &= p({\mathbf{x}}_i)   p({\mathbf{x}}_0|{\mathbf{x}}_i),
\end{aligned}
$$

we rewrite the integral as

$$
\begin{aligned}
&\int p({\mathbf{x}}_i) \int p({\mathbf{x}}_0|{\mathbf{x}}_i) \int p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \mathop{}\!\mathrm{d}{\mathbf{x}}_{i-1} \mathop{}\!\mathrm{d}{\mathbf{x}}_0 \mathop{}\!\mathrm{d}{\mathbf{x}}_i.
\end{aligned}
$$

This allows us to express the expectation in nested form:

$$
\begin{aligned}
&\mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right]
    \right]
    \right].
\end{aligned}
$$

Next, we apply the decomposition of the logarithm:

$$
\begin{aligned}
\log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    =\log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    + \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}.
\end{aligned}
$$

Substituting this back into the expectation gives two terms:

$$
\begin{aligned}
&\mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)}{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right]
    \right]
    \right]
    \\
    &+
    \mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right]
    \right]
    \right].
\end{aligned}
$$

Since the second logarithmic term does not depend on ${\mathbf{x}}_0$, by the law of total probability

$$
\begin{aligned}
\mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right]
    \right]
    = \mathbb{E}_{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)} \left[
    \log \frac{p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}{p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)}
    \right].
\end{aligned}
$$

Similarly, the first term is the KL divergence

$$
\begin{aligned}
\mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)  \|  p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right].
\end{aligned}
$$

Putting it all together, we obtain the decomposition:

$$
\begin{aligned}
&\mathbb{E}_{p({\mathbf{x}}_0, {\mathbf{x}}_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)  \|  p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right]
    \\
    =& \mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i, {\mathbf{x}}_0)  \|  p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right]
    \right]
    \\
    &+
    \mathbb{E}_{p({\mathbf{x}}_i)} \left[
    \mathcal{D}_{\text{KL}} \big(p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)  \|  p_\phi({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)\big)
    \right].
\end{aligned}
$$

**Proof of Optimality.** To prove:

$$
p^*({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)
=
p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i)
=
\mathbb{E}_{p({\mathbf{x}}_0|{\mathbf{x}}_i)}
\left[
p({\mathbf{x}}_{i-1}|{\mathbf{x}}_i,{\mathbf{x}}_0)
\right],
\qquad
{\mathbf{x}}_i \sim p_i.
$$

The first identity follows from the fact that the KL divergence $\mathcal{D}_{\mathrm{KL}}(p \| p_{\boldsymbol{\phi}})$ is minimized when $p^* = p$, assuming the parameterization is sufficiently expressive. The second identity follows directly from the law of total probability. $\blacksquare$

## D.1.2 [Theorem 2.2.3](../../chapters/02-variational-perspective/02-ddpm.md#thm-ddpm-elbo): ELBO of Diffusion Model {#app-vlb-proof}

***Proof.*** For notational simplicity, we denote

$$
{\mathbf{x}}_{1:L}:=({\mathbf{x}}_1,\ldots,{\mathbf{x}}_L).
$$

**Step 1: Apply Jensen’s Inequality.** The marginal log-likelihood is

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)
=
\log \int p_{\boldsymbol{\phi}}({\mathbf{x}}_0,{\mathbf{x}}_{1:L})\,\mathop{}\!\mathrm{d} {\mathbf{x}}_{1:L},
$$

where the joint generative distribution is

$$
p_{\boldsymbol{\phi}}({\mathbf{x}}_0,{\mathbf{x}}_{1:L})
=
p_{\text{prior}}({\mathbf{x}}_L)\prod_{i=1}^L p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i).
$$

We introduce the forward noising distribution

$$
p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)
=
\prod_{i=1}^L p({\mathbf{x}}_i| {\mathbf{x}}_{i-1}),
$$

and rewrite

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)
=
\log \int
p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)
\frac{p_{\boldsymbol{\phi}}({\mathbf{x}}_0,{\mathbf{x}}_{1:L})}{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\,\mathop{}\!\mathrm{d} {\mathbf{x}}_{1:L}.
$$

Applying Jensen’s inequality yields

$$
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)
\ge
\mathbb{E}_{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\left[
\log
\frac{p_{\boldsymbol{\phi}}({\mathbf{x}}_0,{\mathbf{x}}_{1:L})}{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\right]
=: \mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi}),
$$

and hence

$$
-\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0)
\le
-\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi}).
$$

**Step 2: Expand the ELBO.** Substituting the factorizations into the ELBO gives

$$
\begin{aligned}
\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi})
&=
\mathbb{E}_{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\Bigg[
\log p_{\text{prior}}({\mathbf{x}}_L)
+
\sum_{i=1}^L \log p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i)
-
\sum_{i=1}^L \log p({\mathbf{x}}_i| {\mathbf{x}}_{i-1})
\Bigg].
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
-\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi})
&=
\mathbb{E}_{p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)}
\Bigg[
\log \frac{p({\mathbf{x}}_L| {\mathbf{x}}_0)}{p_{\text{prior}}({\mathbf{x}}_L)}
+
\sum_{i=2}^L
\log \frac{p({\mathbf{x}}_{i-1}| {\mathbf{x}}_i,{\mathbf{x}}_0)}{p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i)}
-
\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0| {\mathbf{x}}_1)
\Bigg].
\end{aligned}
$$

Here we used the identity

$$
p({\mathbf{x}}_{1:L}| {\mathbf{x}}_0)
=
p({\mathbf{x}}_L| {\mathbf{x}}_0)\prod_{i=2}^L p({\mathbf{x}}_{i-1}| {\mathbf{x}}_i,{\mathbf{x}}_0),
$$

which follows from the Markov structure of the forward process.

Now group the terms according to their dependence:

$$
\begin{aligned}
-\mathcal{L}_{\text{ELBO}}({\mathbf{x}}_0;\boldsymbol{\phi})
&=
\underbrace{
\mathbb{E}_{p({\mathbf{x}}_L| {\mathbf{x}}_0)}
\left[
\log \frac{p({\mathbf{x}}_L| {\mathbf{x}}_0)}{p_{\text{prior}}({\mathbf{x}}_L)}
\right]
}_{\mathcal{L}_{\text{prior}}({\mathbf{x}}_0)}
+
\underbrace{
\mathbb{E}_{p({\mathbf{x}}_1| {\mathbf{x}}_0)}
\left[
-\log p_{\boldsymbol{\phi}}({\mathbf{x}}_0| {\mathbf{x}}_1)
\right]
}_{\mathcal{L}_{\text{recon.}}({\mathbf{x}}_0;\boldsymbol{\phi})}
\\
&\quad+
\underbrace{
\sum_{i=2}^L
\mathbb{E}_{p({\mathbf{x}}_i| {\mathbf{x}}_0)}
\left[
\mathcal{D}_{\mathrm{KL}}
\big(
p({\mathbf{x}}_{i-1}| {\mathbf{x}}_i,{\mathbf{x}}_0)
\|
p_{\boldsymbol{\phi}}({\mathbf{x}}_{i-1}| {\mathbf{x}}_i)
\big)
\right]
}_{\mathcal{L}_{\text{diffusion}}({\mathbf{x}}_0;\boldsymbol{\phi})}.
\end{aligned}
$$

This proves the claimed decomposition. $\blacksquare$
