# Notations {#notations}

## Numbers and Arrays {#notation-numbers-and-arrays}

| Notation | Meaning |
| --- | --- |
| $a$ | A scalar. |
| ${\mathbf{a}}$ | A column vector (e.g., ${\mathbf{a}}\in\mathbb{R}^D$). |
| ${\mathbf{A}}$ | A matrix (e.g., ${\mathbf{A}}\in\mathbb{R}^{m\times n}$). |
| ${\mathbf{A}}^\top$ | Transpose of ${\mathbf{A}}$. |
| $\operatorname{Tr}({\mathbf{A}})$ | Trace of ${\mathbf{A}}$. |
| ${\mathbf{I}}_D$ | Identity matrix of size $D\times D$. |
| ${\mathbf{I}}$ | Identity matrix; dimension implied by context. |
| $\mathrm{diag}({\mathbf{a}})$ | Diagonal matrix with diagonal entries given by ${\mathbf{a}}$. |
| $\boldsymbol{\phi},\,\boldsymbol{\theta}$ | Learnable neural network parameters. |
| $\boldsymbol{\phi}^{\times},\,\boldsymbol{\theta}^{\times}$ | Parameters after training (fixed during inference). |
| $\boldsymbol{\phi}^{*},\,\boldsymbol{\theta}^{*}$ | Optimal parameters of an optimization problem. |

## Calculus {#notation-calculus}

| Notation | Meaning |
| --- | --- |
| $\dfrac{\partial {\mathbf{y}}}{\partial {\mathbf{x}}}$ | Partial derivatives of ${\mathbf{y}}$ w.r.t. ${\mathbf{x}}$ (componentwise). |
| $\dfrac{\mathop{}\!\mathrm{d} {\mathbf{y}}}{\mathop{}\!\mathrm{d} {\mathbf{x}}}$ or $\mathrm{D}{\mathbf{y}}({\mathbf{x}})$ | Total (Fréchet) derivative of ${\mathbf{y}}$ w.r.t. ${\mathbf{x}}$. |
| $\nabla_{\mathbf{x}} y$ | Gradient of scalar $y:\mathbb{R}^D \to\mathbb{R}$; a column in $\mathbb{R}^D$. |
| $\dfrac{\partial {\mathbf{F}}}{\partial {\mathbf{x}}}$ or $\nabla_{\mathbf{x}} {\mathbf{F}}$ | Jacobian of ${\mathbf{F}}:\mathbb{R}^n \to\mathbb{R}^m$; shape $m\times n$. |
| $\nabla\cdot{\mathbf{y}}$ | Divergence of a vector field ${\mathbf{y}}:\mathbb{R}^D \to\mathbb{R}^D$; a scalar. |
| $\nabla^2_{\mathbf{x}} f({\mathbf{x}})$ or ${\mathbf{H}}(f)({\mathbf{x}})$ | Hessian of $f:\mathbb{R}^D \to\mathbb{R}$; shape $D\times D$. |
| $\displaystyle\int f({\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}$ | Integral of $f$ over the domain of ${\mathbf{x}}$. |

## Probability and Information Theory {#notation-probability-and-information-theory}

| Notation | Meaning |
| --- | --- |
| $p({\mathbf{x}})$ | Density/distribution over a continuous vector ${\mathbf{x}}$. |
| $\Pr({\mathbf{x}} = {\mathbf{a}})$, or informally $\Pr({\mathbf{x}})$ | Point probability in the discrete setting. |
| $\Pr({\mathbf{x}} = {\mathbf{a}} \vert {\mathbf{y}} = {\mathbf{b}})$, or informally $\Pr({\mathbf{x}} \vert {\mathbf{y}})$ | Conditional point probability in the discrete setting. |
| $p_{\mathrm{data}}$ | Data distribution. |
| $p_{\mathrm{prior}}$ | Prior distribution (e.g., standard normal). |
| $p_{\mathrm{src}}$ | Source distribution. |
| $p_{\mathrm{tgt}}$ | Target distribution. |
| ${\mathbf{a}} \sim p$ | Random vector ${\mathbf{a}}$ is distributed as $p$. |
| ${\mathbb{E}}_{{\mathbf{x}}\sim p}\big[{\mathbf{f}}({\mathbf{x}})\big]$ | Expectation of ${\mathbf{f}}({\mathbf{x}})$ under $p({\mathbf{x}})$. |
| ${\mathbb{E}}\big[{\mathbf{f}}({\mathbf{x}})\,\vert \,{\mathbf{z}}\big]$, or ${\mathbb{E}}_{{\mathbf{x}}\sim p(\cdot\vert {\mathbf{z}})}\big[{\mathbf{f}}({\mathbf{x}})\big]$ | Conditional expectation of ${\mathbf{f}}({\mathbf{x}})$ given ${\mathbf{z}}$, with ${\mathbf{x}}$ distributed as $p(\cdot\vert {\mathbf{z}})$. |
| $\mathrm{Var}\big({\mathbf{f}}({\mathbf{x}})\big)$ | Variance under $p({\mathbf{x}})$. |
| $\mathrm{Cov}\big({\mathbf{f}}({\mathbf{x}}),{\mathbf{g}}({\mathbf{x}})\big)$ | Covariance under $p({\mathbf{x}})$. |
| $\mathcal{D}_{\mathrm{KL}}\left(p\,\Vert\, q\right)$ | Kullback–Leibler divergence from $q$ to $p$. |
| $\boldsymbol{\epsilon}\sim\mathcal{N}(\boldsymbol{0},{\mathbf{I}})$ | Standard normal sample. |
| $\mathcal{N}({\mathbf{x}};{\boldsymbol{\mu}},{\boldsymbol{\Sigma}})$ | Gaussian over ${\mathbf{x}}$ with mean ${\boldsymbol{\mu}}$ and covariance ${\boldsymbol{\Sigma}}$. |

## Notational Conventions. {#notational-conventions}

We collect several conventions used throughout this book.

### I. Random Variables and Their Realizations.

We use the same symbol for a random vector and its realized value. This convention, common in deep learning and generative modeling, keeps notation compact. The intended meaning is determined by context.

For densities, $p({\mathbf{x}})$ denotes the functional form of the distribution rather than evaluation at a particular sample. When evaluation at a specific point is intended, we state so explicitly. Conditional densities are read by context: in $p({\mathbf{x}}|{\mathbf{y}})$, fixing ${\mathbf{y}}$ gives a density in ${\mathbf{x}}$, while fixing ${\mathbf{x}}$ gives a function of ${\mathbf{y}}$.

### II. Conditional Expectations.

The expression ${\mathbb{E}}[{\mathbf{f}}({\mathbf{x}})|{\mathbf{z}}]$ denotes a function of ${\mathbf{z}}$, giving the expected value of ${\mathbf{f}}({\mathbf{x}})$ conditional on ${\mathbf{z}}$. When conditioning on a specific realized value, we write ${\mathbb{E}}[{\mathbf{f}}({\mathbf{x}})|{\mathbf{Z}} = {\mathbf{z}}]$. Equivalently, this can be expressed as an integral with respect to the conditional distribution:

$$
{\mathbb{E}}_{{\mathbf{x}} \sim p(\cdot|{\mathbf{z}})}[{\mathbf{f}}({\mathbf{x}})]
= \int {\mathbf{f}}({\mathbf{x}})\, p({\mathbf{x}}|{\mathbf{z}})\, \mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

This distinction clarifies whether ${\mathbf{z}}$ is treated as a variable defining a function ${\mathbf{z}} \mapsto {\mathbb{E}}[{\mathbf{f}}({\mathbf{x}})|{\mathbf{z}}]$, or as a fixed value at which that function is evaluated.

### III. Implicit Time Indices.

Since many objects in this book are indexed by time, writing every time variable explicitly can make formulas heavy. When the relevant time indices are clear from context, we leave them implicit. For example,

$$
{\mathbb{E}}[{\mathbf{x}}_s | {\mathbf{x}}_t]
\quad\text{is shorthand for}\quad
{\mathbb{E}}[{\mathbf{x}}_s | {\mathbf{x}}_t,\, s,\, t],
$$

meaning the average of ${\mathbf{x}}_s$ at time $s$ given ${\mathbf{x}}_t$ at time $t$, with $s$ and $t$ understood from the surrounding discussion. Similarly, $p({\mathbf{x}}_s | {\mathbf{x}}_t)$ denotes the conditional distribution of ${\mathbf{x}}_s$ given ${\mathbf{x}}_t$, with both time indices left implicit.

### IV. Gaussian Perturbation and Equality in Distribution.

For a fixed index $t$, we frequently use the Gaussian perturbation kernel

$$
p_t({\mathbf{x}}_t|{\mathbf{x}}_0)
= \mathcal{N}\big({\mathbf{x}}_t;\, \alpha_t {\mathbf{x}}_0,\, \sigma_t^2 {\mathbf{I}}\big),
$$

where ${\mathbf{x}}_0 \sim p_{\mathrm{data}}$, $\alpha_t$ and $\sigma_t$ are deterministic scalars. We equivalently write this as the identity *in distribution*

$$
{\mathbf{x}}_t \stackrel{d}{=} \alpha_t {\mathbf{x}}_0 + \sigma_t {\boldsymbol{\epsilon}},
\qquad
{\boldsymbol{\epsilon}} \sim \mathcal{N}(\boldsymbol{0}, {\mathbf{I}}),
$$

where ${\boldsymbol{\epsilon}}$ is independent of ${\mathbf{x}}_0$. The symbol “$\stackrel{d}{=}$” means the two sides have the same probability law, so that

$$
{\mathbb{E}}[\phi({\mathbf{x}}_t)] = {\mathbb{E}}[\phi(\alpha_t {\mathbf{x}}_0 + \sigma_t {\boldsymbol{\epsilon}})]
$$

for any test function $\phi$. For brevity, we will simply write ${\mathbf{x}}_t = \alpha_t {\mathbf{x}}_0 + \sigma_t {\boldsymbol{\epsilon}}$, understood either as an equality in distribution or as a sample realization depending on context; this shorthand is used throughout.
