# C.1 Itô’s Formula: The Chain Rule for Random Processes {#sec-ito-formula}

Standard calculus does not directly apply to stochastic processes because Wiener processes are not differentiable in the classical sense. Instead, we use Itô’s calculus, which provides rules for working with stochastic integrals.

## C.1.1 Motivation: Why Do We Need a Special Chain Rule? {#motivation-special-chain-rule}

Consider a deterministic time-varying function ${\mathbf{y}}_t$ that evolves smoothly with time $t$ (e.g., an ODE). If we have a function ${\mathbf{h}}({\mathbf{y}}_t, t)$, the usual chain rule tells us:

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{h}}}{\mathop{}\!\mathrm{d} t} = \frac{\partial {\mathbf{h}}}{\partial t}  + \nabla_{{\mathbf{y}}}{\mathbf{h}}  \frac{\mathop{}\!\mathrm{d} {\mathbf{y}}_t}{\mathop{}\!\mathrm{d} t} .
$$

Here, $\nabla_{{\mathbf{y}}}{\mathbf{h}}$ is the Jacobian of ${\mathbf{h}}$. This works perfectly for deterministic paths ${\mathbf{y}}_t$.

<span id="question-c-1-1"></span>

**Question C.1.1**

But what happens if ${\mathbf{x}}_t$ is a stochastic process, say, driven by an SDE

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = {\mathbf{f}}({\mathbf{x}}_t, t)\mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t
$$

as in [Equation (A.2.3)](../a-differential-equations/02-stochastic-differential-equations.md#eq-sde)? What SDE does the process ${\mathbf{h}}({\mathbf{x}}_t, t)$ satisfy?

**Why the Ordinary Chain Rule Fails?** Naïvely applying the classical chain rule yields

$$
\mathop{}\!\mathrm{d} {\mathbf{h}} = \frac{\partial {\mathbf{h}}}{\partial t} \mathop{}\!\mathrm{d} t + \nabla_{{\mathbf{y}}}{\mathbf{h}}  \cdot \mathop{}\!\mathrm{d} {\mathbf{x}}_t.
$$

However, Brownian increments scale as $\mathcal{O}(\sqrt{\mathop{}\!\mathrm{d} t})$, while their quadratic variation is encoded in Itô calculus by

$$
(\mathop{}\!\mathrm{d} {\mathbf{w}}_t)^2 = \mathop{}\!\mathrm{d} t.
$$

Thus, second-order terms in $\mathop{}\!\mathrm{d} {\mathbf{w}}_t$ *do not* vanish in stochastic calculus, unlike classical calculus where $(\mathop{}\!\mathrm{d} t)^2$ terms are negligible.

**Example: Simple Example–$h(x_t) = x_t^2$**

To see the intuition, let us consider the simple real-valued function $h(x_t) = x_t^2 \in \mathbb{R}$ where the random variable $x_t \in \mathbb{R}$ satisfies

$$
\mathop{}\!\mathrm{d} x_t = \sigma \mathop{}\!\mathrm{d} w_t, \quad
x_0=0,
$$

with a constant $\sigma>0$. If we try the classical chain rule,

$$
\mathop{}\!\mathrm{d} h = 2 x_t \mathop{}\!\mathrm{d} x_t = 2 x_t \sigma \mathop{}\!\mathrm{d} w_t.
$$

If this were true, the expectation of $h(x_t)$ would be constant in time because

$$
\mathbb{E}[\mathop{}\!\mathrm{d} h ] =  2 \sigma \mathbb{E}[x_t  \mathop{}\!\mathrm{d} w_t]= 2 \sigma \mathbb{E}[x_t]  \underbrace{\mathbb{E}[\mathop{}\!\mathrm{d} w_t]}_{=0}=0.
$$

But we know from classical Brownian motion properties (see [Equation (A.2.4)](../a-differential-equations/02-stochastic-differential-equations.md#eq-wiener-gaussian)) that

$$
\mathbb{E}[x_t^2] = \sigma^2 t,
$$

which grows linearly in time. So the ordinary chain rule misses an important term.

## C.1.2 Deriving 1D Itô’s Formula from Taylor Expansion {#subsec-1d-ito}

**Deterministic Chain Rule via Taylor Expansion.** To understand why the classical chain rule fails for stochastic processes defined by SDEs, we first revisit it in the deterministic setting using Taylor expansion. We consider the scalar case: $y_t \in \mathbb{R}$ and $h(\cdot,\cdot)\in \mathbb{R}$. Formally treating $\mathop{}\!\mathrm{d} y_t = y_{t+\mathop{}\!\mathrm{d} t} - y_t$, with $\mathop{}\!\mathrm{d} t \approx 0$, we expand:

$$
\begin{aligned}
&h(y_{t+\mathop{}\!\mathrm{d} t}, t+\mathop{}\!\mathrm{d} t)
- h(y_t, t) \\
= ~&\frac{\partial h}{\partial t} \mathop{}\!\mathrm{d} t
+ \frac{\partial h}{\partial y} \mathop{}\!\mathrm{d} y_t  + \frac{1}{2} \left({\color{gray}\frac{\partial^2 h}{\partial y^2}  (\mathop{}\!\mathrm{d} y_t)^2}
+ 2 {\color{gray}\frac{\partial^2 h}{\partial t \partial y} \mathop{}\!\mathrm{d} t \mathop{}\!\mathrm{d} y_t}
+  {\color{gray}\frac{\partial^2 h}{\partial t^2} (\mathop{}\!\mathrm{d} t)^2}
\right)
+ \mathcal{O}(\mathop{}\!\mathrm{d} t^3),
\end{aligned}
$$

Here, $\mathop{}\!\mathrm{d} t \mathop{}\!\mathrm{d} y_t = \left(\frac{\mathop{}\!\mathrm{d} y_t}{\mathop{}\!\mathrm{d} t}\right)(\mathop{}\!\mathrm{d} t)^2 = \mathcal{O}(\mathop{}\!\mathrm{d} t^2)$, and similarly $(\mathop{}\!\mathrm{d} t)^2 = \mathcal{O}(\mathop{}\!\mathrm{d} t^2)$. Therefore, all the gray parts are ignorable, and the full differential is:

$$
\mathop{}\!\mathrm{d} h
= \frac{\partial h}{\partial t} \mathop{}\!\mathrm{d} t
+ \frac{\partial h}{\partial y} \mathop{}\!\mathrm{d} y_t
+ \mathcal{O}(\mathop{}\!\mathrm{d} t^2).
$$

**Itô’s Formula via Stochastic Taylor Expansion.** Now consider a stochastic process $x_t \in \mathbb{R}$ governed by the SDE:

$$
\mathop{}\!\mathrm{d} x_t = f(x_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} w_t,
$$

where $w_t$ is standard Brownian motion. We aim to compute the differential of a scalar-valued function $h(x_t, t)$.

Using the stochastic Taylor expansion ([Kloeden et al., 1992](references.md#ref-kloeden1992stochastic)), which retains second-order terms in $\mathop{}\!\mathrm{d} x_t$, we have:

$$
\begin{aligned}
&h(x_{t+\mathop{}\!\mathrm{d} t}, t+\mathop{}\!\mathrm{d} t)
- h(x_t, t) \\
=~&\frac{\partial h}{\partial t} \mathop{}\!\mathrm{d} t
+ \frac{\partial h}{\partial x} \mathop{}\!\mathrm{d} x_t
+ \frac{1}{2} \left( \frac{\partial^2 h}{\partial x^2} (\mathop{}\!\mathrm{d} x_t)^2
+ 2{\color{gray}{\frac{\partial^2 h}{\partial t \partial x} \mathop{}\!\mathrm{d} t \mathop{}\!\mathrm{d} x_t}}
+ {\color{gray}{ \frac{\partial^2 h}{\partial t^2} (\mathop{}\!\mathrm{d} t)^2}} \right)
+ \cdots
\end{aligned}
$$

**Negligible Cross Terms.** By the scaling property of Brownian motion ([Equation (A.2.4)](../a-differential-equations/02-stochastic-differential-equations.md#eq-wiener-gaussian)),

$$
\mathop{}\!\mathrm{d} w_t = \mathcal{O}(\sqrt{\mathop{}\!\mathrm{d} t}) \quad \Rightarrow \quad \mathop{}\!\mathrm{d} t \cdot \mathop{}\!\mathrm{d} w_t = \mathcal{O}((\mathop{}\!\mathrm{d} t)^{3/2}).
$$

Therefore,

$$
\begin{aligned}
\mathop{}\!\mathrm{d} t \cdot \mathop{}\!\mathrm{d} x_t
&= \mathop{}\!\mathrm{d} t \left(f \mathop{}\!\mathrm{d} t + g \mathop{}\!\mathrm{d} w_t\right)
= f (\mathop{}\!\mathrm{d} t)^2 + g \cdot \mathop{}\!\mathrm{d} t \cdot \mathop{}\!\mathrm{d} w_t
= \mathcal{O}((\mathop{}\!\mathrm{d} t)^{3/2}).
\end{aligned}
$$

So, the gray terms are negligible: $\mathcal{O}((\mathop{}\!\mathrm{d} t)^{3/2})$ or smaller.

**Second-Order Term $(\mathop{}\!\mathrm{d} x_t)^2$.** Expanding using the SDE:

$$
\begin{aligned}
(\mathop{}\!\mathrm{d} x_t)^2
&= \left( f \mathop{}\!\mathrm{d} t + g \mathop{}\!\mathrm{d} w_t \right)^2 \\
&= f^2 (\mathop{}\!\mathrm{d} t)^2 + 2fg \mathop{}\!\mathrm{d} t \mathop{}\!\mathrm{d} w_t + g^2 (\mathop{}\!\mathrm{d} w_t)^2 \\
&= g^2(t) \mathop{}\!\mathrm{d} t,
\end{aligned}
$$

where the last equality uses the Itô differential rules $(\mathop{}\!\mathrm{d} t)^2=0$, $\mathop{}\!\mathrm{d} t\,\mathop{}\!\mathrm{d} w_t=0$, and $(\mathop{}\!\mathrm{d} w_t)^2=\mathop{}\!\mathrm{d} t$, with the last identity understood in the quadratic-variation sense.

Combining terms, we obtain the differential:

$$
\mathop{}\!\mathrm{d} h(x_t, t)
= \frac{\partial h}{\partial t} \mathop{}\!\mathrm{d} t
+ \frac{\partial h}{\partial x} \mathop{}\!\mathrm{d} x_t
+ \frac{1}{2} \frac{\partial^2 h}{\partial x^2} g^2(t) \mathop{}\!\mathrm{d} t.
$$

Substituting $\mathop{}\!\mathrm{d} x_t = f(x_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} w_t$ yields:

$$
\mathop{}\!\mathrm{d} h(x_t, t)
= \left( \frac{\partial h}{\partial t} + f \frac{\partial h}{\partial x}
+ \frac{1}{2} g^2 \frac{\partial^2 h}{\partial x^2} \right) \mathop{}\!\mathrm{d} t
+ g \frac{\partial h}{\partial x} \mathop{}\!\mathrm{d} w_t.
$$

This is the 1D version of *Itô’s formula*.

**Example: Simple Example–$h(x_t) = x_t^2$**

We revisit the simple example: $h(x_t) = x_t^2$, where the stochastic process $x_t \in \mathbb{R}$ satisfies

$$
\mathop{}\!\mathrm{d} x_t = \sigma \mathop{}\!\mathrm{d} w_t,
$$

with a constant $\sigma > 0$. Applying Itô’s formula correctly to $h(x_t) = x_t^2$, we obtain:

$$
\mathop{}\!\mathrm{d} h(x_t) = \mathop{}\!\mathrm{d} (x_t^2) = 2 x_t \mathop{}\!\mathrm{d} x_t + \sigma^2 \mathop{}\!\mathrm{d} t.
$$

Substituting $\mathop{}\!\mathrm{d} x_t = \sigma \mathop{}\!\mathrm{d} w_t$, this becomes:

$$
\mathop{}\!\mathrm{d}(x_t^2) = 2 x_t \sigma \mathop{}\!\mathrm{d} w_t + \sigma^2 \mathop{}\!\mathrm{d} t.
$$

## C.1.3 Itô’s Formula: The Chain Rule for SDEs {#subsec-ito-summary}

We summarize the one-dimensional Itô’s formula derived above. Using similar arguments, the result extends naturally to the multi-dimensional setting. While we omit the detailed derivation, we state the general formula for completeness.

Finally, we illustrate an application of Itô’s formula by deriving the *Itô product rule*, which enables computation of $\mathop{}\!\mathrm{d}({\mathbf{x}}_t^\top {\mathbf{y}}_t)$ for stochastic processes ${\mathbf{x}}_t$ and ${\mathbf{y}}_t$.

**$1$D Itô’s Formula.** Let $x_t \in \mathbb{R}$ be a stochastic process satisfying the SDE:

$$
\mathop{}\!\mathrm{d} x_t = f(x_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} w_t.
$$

For a scalar function $h \colon \mathbb{R} \times [0, T] \to \mathbb{R}$, the process $h(x_t, t)$ satisfies:

$$
\mathop{}\!\mathrm{d} h(x_t, t)
= \left( \frac{\partial h}{\partial t} + f \frac{\partial h}{\partial x}
+ \frac{1}{2} g^2 \frac{\partial^2 h}{\partial x^2} \right) \mathop{}\!\mathrm{d} t
+ g \frac{\partial h}{\partial x} \mathop{}\!\mathrm{d} w_t.
$$

**Multidimensional Itô’s Formula with Scalar Output.** Let ${\mathbf{x}}_t \in \mathbb{R}^D$ satisfy the SDE:

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = {\mathbf{f}}({\mathbf{x}}_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t,
$$

where ${\mathbf{f}} \colon \mathbb{R}^D \times [0, T] \to \mathbb{R}^D$, $g \colon [0, T] \to \mathbb{R}$, and ${\mathbf{w}}_t \in \mathbb{R}^D$ is a $D$-dimensional Brownian motion. Let $h \colon \mathbb{R}^D \times [0, T] \to \mathbb{R}$ be a scalar-valued function. Then $h({\mathbf{x}}_t, t)$ satisfies:

<span id="eq-ito-scalar"></span>

$$
\begin{aligned}
\mathop{}\!\mathrm{d} h({\mathbf{x}}_t, t)
= \left( \frac{\partial h}{\partial t}
+ \nabla_{{\mathbf{x}}} h^\top {\mathbf{f}}
+ \frac{1}{2} g^2(t) \operatorname{Tr} \left( \nabla^2_{{\mathbf{x}}} h \right) \right) \mathop{}\!\mathrm{d} t
+ g(t) \nabla_{{\mathbf{x}}} h^\top \mathop{}\!\mathrm{d} {\mathbf{w}}_t,
\end{aligned}
\tag{C.1.1}
$$

where $\nabla_{{\mathbf{x}}} h \in \mathbb{R}^D$ is the gradient and $\nabla^2_{{\mathbf{x}}} h \in \mathbb{R}^{D \times D}$ is the Hessian matrix of $h$ with respect to ${\mathbf{x}}$.

**Example: Itô’s Product Rule**

Let ${\mathbf{x}}_t, {\mathbf{y}}_t \in \mathbb{R}^D$ be vector-valued stochastic processes governed by the SDEs:

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}}_t &= {\mathbf{a}}({\mathbf{x}}_t, t) \mathop{}\!\mathrm{d} t + b(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t, \\
\mathop{}\!\mathrm{d} {\mathbf{y}}_t &= {\mathbf{c}}({\mathbf{y}}_t, t) \mathop{}\!\mathrm{d} t + d(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t,
\end{aligned}
$$

where ${\mathbf{a}}, {\mathbf{c}}: \mathbb{R}^D \times [0, T] \to \mathbb{R}^D$ are vector fields, and $b(t), d(t) \in \mathbb{R}$ are scalar-valued functions. Here, ${\mathbf{w}}_t \in \mathbb{R}^D$ denotes a standard $D$-dimensional Brownian motion.

We aim to derive the SDE for the scalar-valued process

$$
z(t) := {\mathbf{x}}_t^\top {\mathbf{y}}_t.
$$

Applying the multivariate Itô formula to the bilinear function $h({\mathbf{x}}, {\mathbf{y}}) := {\mathbf{x}}^\top {\mathbf{y}}$, we obtain:

$$
\mathop{}\!\mathrm{d}({\mathbf{x}}^\top {\mathbf{y}}) = (\mathop{}\!\mathrm{d} {\mathbf{x}})^\top {\mathbf{y}} + {\mathbf{x}}^\top \mathop{}\!\mathrm{d} {\mathbf{y}} + \operatorname{Tr}\left[ \mathop{}\!\mathrm{d} {\mathbf{x}} \cdot (\mathop{}\!\mathrm{d} {\mathbf{y}})^\top \right].
$$

The Itô correction term is computed as:

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}} \cdot (\mathop{}\!\mathrm{d} {\mathbf{y}})^\top &= b(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t \cdot \left[ d(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t \right]^\top \\&= b(t) d(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t \cdot \mathop{}\!\mathrm{d} {\mathbf{w}}_t^\top \\&= b(t)d(t) \mathop{}\!\mathrm{d} t \cdot {\mathbf{I}}_D.
\end{aligned}
$$

Thus,

$$
\operatorname{Tr}\left[ \mathop{}\!\mathrm{d} {\mathbf{x}} \cdot (\mathop{}\!\mathrm{d} {\mathbf{y}})^\top \right] = b(t)d(t) \operatorname{Tr}({\mathbf{I}}_D) \mathop{}\!\mathrm{d} t = D  b(t)d(t) \mathop{}\!\mathrm{d} t.
$$

Putting everything together, the resulting SDE is:

<span id="eq-ito-product"></span>

$$
\begin{aligned}
        \mathop{}\!\mathrm{d}({\mathbf{x}}^\top {\mathbf{y}}) &= (\mathop{}\!\mathrm{d} {\mathbf{x}})^\top {\mathbf{y}} + {\mathbf{x}}^\top \mathop{}\!\mathrm{d} {\mathbf{y}} + D  b(t)d(t) \mathop{}\!\mathrm{d} t
\end{aligned}
\tag{C.1.2}
$$

## C.1.4 Itô’s Formula’s Application: Derivation of Fokker-Planck Equation {#subsec-rigorous-proof-fpe}

In this section, we apply Itô’s formula from [Equation (C.1.1)](01-ito-formula.md#eq-ito-scalar) to derive the Fokker–Planck equation, a PDE that characterizes the time evolution of the probability density $p_t({\mathbf{x}})$ associated with the $D$-dimensional diffusion process defined by the SDE in [Equation (A.2.3)](../a-differential-equations/02-stochastic-differential-equations.md#eq-sde).

**Step 1: Apply Itô’s Formula.** Let $\phi({\mathbf{x}}, t)$ be a smooth test function $\phi \colon \mathbb{R}^D \times [0, T] \to \mathbb{R}$. By Itô’s Formula:

$$
\mathop{}\!\mathrm{d} \phi({\mathbf{x}}_t, t) = \left( \frac{\partial \phi}{\partial t} + \nabla_{{\mathbf{x}}} \phi^\top {\mathbf{f}}({\mathbf{x}}_t, t) + \frac{1}{2} g^2(t) \operatorname{Tr}[\nabla_{{\mathbf{x}}}^2 \phi] \right) \mathop{}\!\mathrm{d} t + g(t) \nabla_{{\mathbf{x}}} \phi^\top \mathop{}\!\mathrm{d} {\mathbf{w}}_t.
$$

**Step 2: Take Expectation.** Taking expectation and using that the Itô integral has zero expectation under the standard integrability conditions:

$$
\mathbb{E}[\mathop{}\!\mathrm{d} \phi({\mathbf{x}}_t, t)] = \mathbb{E} \left[ \left( \frac{\partial \phi}{\partial t} + \nabla_{{\mathbf{x}}} \phi^\top {\mathbf{f}}({\mathbf{x}}_t, t) + \frac{1}{2} g^2(t) \operatorname{Tr}[\nabla_{{\mathbf{x}}}^2 \phi] \right) \mathop{}\!\mathrm{d} t \right].
$$

**Step 3: Express Expectation via Density.** This expectation can be written as:

$$
\mathbb{E}[\mathop{}\!\mathrm{d} \phi({\mathbf{x}}_t, t)] = \int \left( \frac{\partial \phi}{\partial t} + \nabla_{{\mathbf{x}}} \phi^\top {\mathbf{f}}({\mathbf{x}}, t) + \frac{1}{2} g^2(t) \operatorname{Tr}[\nabla_{{\mathbf{x}}}^2 \phi] \right) p_t({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \mathop{}\!\mathrm{d} t.
$$

**Step 4: Integrate by Parts.** Use integration by parts (divergence theorem) in $\mathbb{R}^D$:

$$
\begin{aligned}
\int \nabla_{{\mathbf{x}}} \phi^\top {\mathbf{f}} p_t \mathop{}\!\mathrm{d} {\mathbf{x}} &= -\int \phi \nabla_{{\mathbf{x}}} \cdot ({\mathbf{f}} p_t) \mathop{}\!\mathrm{d} {\mathbf{x}}, \\
    \int \operatorname{Tr}[\nabla_{{\mathbf{x}}}^2 \phi] p_t  \mathop{}\!\mathrm{d} {\mathbf{x}} &= \int \phi \Delta p_t \mathop{}\!\mathrm{d} {\mathbf{x}}.
\end{aligned}
$$

**Step 5: Substitute and Rearrange.** Note that

$$
\mathbb{E}[\phi({\mathbf{x}}_t,t)] = \int \phi({\mathbf{x}},t) p_t({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

Differentiating in time, we have

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\mathbb{E}[\phi({\mathbf{x}}_t,t)]
= \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\int \phi({\mathbf{x}},t) p_t({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}}
= \int\Bigl(\partial_t\phi({\mathbf{x}},t) p_t({\mathbf{x}}) + \phi({\mathbf{x}},t) \partial_t p_t({\mathbf{x}})\Bigr)\mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

Therefore,

$$
\mathbb{E}[\mathop{}\!\mathrm{d} \phi({\mathbf{x}}_t,t)]
= \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\mathbb{E}[\phi({\mathbf{x}}_t,t)] \mathop{}\!\mathrm{d} t
= \int\Bigl(\partial_t\phi({\mathbf{x}},t) p_t({\mathbf{x}}) + \phi({\mathbf{x}},t) \partial_t p_t({\mathbf{x}})\Bigr)\mathop{}\!\mathrm{d}{\mathbf{x}} \mathop{}\!\mathrm{d} t.
$$

Equating this expression with Step 3, the terms $\int \partial_t\phi\,p_t\,\mathop{}\!\mathrm{d}{\mathbf{x}}\,\mathop{}\!\mathrm{d} t$ cancel. Applying Step 4 to the remaining spatial terms gives

$$
\int
\phi({\mathbf{x}},t)
\left[
\frac{\partial p_t}{\partial t}
+
\nabla_{{\mathbf{x}}}\cdot\bigl({\mathbf{f}}({\mathbf{x}},t)p_t({\mathbf{x}})\bigr)
-
\frac{1}{2}g^2(t)\Delta p_t({\mathbf{x}})
\right]
\mathop{}\!\mathrm{d}{\mathbf{x}}
=
0.
$$

Since this holds for every smooth test function $\phi$, we obtain the Fokker–Planck equation

$$
\frac{\partial p_t({\mathbf{x}})}{\partial t}
=
-\nabla_{{\mathbf{x}}}\cdot\bigl({\mathbf{f}}({\mathbf{x}},t)p_t({\mathbf{x}})\bigr)
+
\frac{1}{2}g^2(t)\Delta p_t({\mathbf{x}}).
$$

## C.1.5 Itô’s Formula Application: Closed-Form Solution of a Linear SDE {#subsec-rigorous-proof-linear-sde}

This subsection demonstrates how to obtain a closed-form solution for a linear SDE by using an integration factor (similar to the ODE case) and Itô’s formula. The approach mirrors classical techniques for solving linear ODEs, but adapted to the stochastic setting.

We consider a linear SDE of the form

<span id="eq-app-linear-sde"></span>

$$
\begin{aligned}
\mathrm{d}{\mathbf{x}}_t = f(t){\mathbf{x}}_t \mathrm{d}t + g(t) \mathrm{d}{\mathbf{w}}_t,
\end{aligned}
\tag{C.1.3}
$$

where $f(t)$ and $g(t)$ are deterministic functions and ${\mathbf{w}}_t$ is a standard Wiener process.

**Closed-Form Solution of a Linear SDE.** We derive the explicit solution to the linear SDE in [Equation (C.1.3)](01-ito-formula.md#eq-app-linear-sde) using the method of integrating factors. This type of forward SDE commonly arises in diffusion models (see [Section 4.1](../../chapters/04-score-sde-framework/01-score-sde-principles.md#sec-dm-today)).

**Step 1: Define an Integration Factor.** Let

$$
\Psi(t) := \exp\left( -\int_0^t f(s) \mathrm{d}s \right),
\quad \text{and define} \quad
{\mathbf{y}}_t := \Psi(t){\mathbf{x}}_t.
$$

**Step 2: Apply Itô’s Formula.** We apply Itô’s formula to the function ${\mathbf{h}}({\mathbf{x}}, t) := \Psi(t){\mathbf{x}}$. This is actually a special case of the Itô product rule in [Equation (C.1.2)](01-ito-formula.md#eq-ito-product). Since $\Psi(t)$ is deterministic, there is no cross-variation term, and the formula simplifies to:

$$
\begin{aligned}
\mathrm{d}{\mathbf{y}}_t
    &= \mathrm{d}[\Psi(t){\mathbf{x}}_t] \\
    &= \Psi'(t){\mathbf{x}}_t \mathrm{d}t + \Psi(t) \mathrm{d}{\mathbf{x}}_t \\
    &= -f(t)\Psi(t){\mathbf{x}}_t \mathrm{d}t + \Psi(t)\left[f(t){\mathbf{x}}_t \mathrm{d}t + g(t) \mathrm{d}{\mathbf{w}}_t\right] \\
    &= \Psi(t)g(t) \mathrm{d}{\mathbf{w}}_t.
\end{aligned}
$$

Hence,

$$
{\mathbf{y}}_t = {\mathbf{y}}_0 + \int_0^t \Psi(s)g(s) \mathrm{d}{\mathbf{w}}(s) = {\mathbf{x}}_0 + \int_0^t \Psi(s)g(s) \mathrm{d}{\mathbf{w}}(s),
$$

since $\Psi(0) = 1$.

**Step 3: Solve for ${\mathbf{x}}_t$.** Using ${\mathbf{x}}_t = \Psi(t)^{-1}{\mathbf{y}}_t$, we obtain

<span id="eq-lienar-sde-solution"></span>

$$
\begin{aligned}
{\mathbf{x}}_t &=  e^{ \int_0^t f(s)\mathrm{d}s}
    \left[ {\mathbf{x}}_0 + \int_0^t e^{ -\int_0^s f(r) \mathrm{d}r } g(s) \mathrm{d}{\mathbf{w}}(s) \right].
\end{aligned}
\tag{C.1.4}
$$

This provides an explicit solution to the vector-valued SDE.

Below, we demonstrate two alternative approaches to compute the analytical form of $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$.

**Analysis of the Closed-Form Solution.** [Equation (C.1.4)](01-ito-formula.md#eq-lienar-sde-solution) reconfirms that $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ is Gaussian. To see this, define

$$
\phi(s) := e^{-\int_0^s f(u) \mathop{}\!\mathrm{d} u} g(s),
$$

which is a deterministic matrix-valued function of time (assuming $f(u)$ and $g(s)$ are deterministic). The Itô integral $\int_0^t \phi(s) \mathop{}\!\mathrm{d} {\mathbf{w}}_s$ is then a zero-mean Gaussian random variable, as it is the stochastic integral of a deterministic function with respect to Brownian motion. Therefore, ${\mathbf{x}}_t|{\mathbf{x}}_0$ is an affine transformation of a Gaussian random variable and hence itself Gaussian. Its distribution is fully characterized by its conditional mean and covariance.

We define the conditional mean and covariance (given initial condition ${\mathbf{x}}_0$) as

$$
{\mathbf{m}}(t) := \mathbb{E}[{\mathbf{x}}_t | {\mathbf{x}}_0], \quad {\mathbf{P}}(t) := \mathbb{E}[({\mathbf{x}}_t - {\mathbf{m}}(t))({\mathbf{x}}_t - {\mathbf{m}}(t))^\top|{\mathbf{x}}_0].
$$

**Mean.** Using linearity of expectation and the fact that the Itô integral of a deterministic function has zero mean:

$$
\begin{aligned}
{\mathbf{m}}(t) = \mathbb{E}\left[ e^{\int_0^t f(s) \mathop{}\!\mathrm{d} s} \left( {\mathbf{x}}_0 + \int_0^t \phi(s) \mathop{}\!\mathrm{d} {\mathbf{w}}_s \right) \bigg| {\mathbf{x}}_0 \right] = e^{\int_0^t f(s) \mathop{}\!\mathrm{d} s} {\mathbf{x}}_0.
\end{aligned}
$$

**Covariance.** Let ${\mathbf{z}}_t := \int_0^t \phi(s) \mathop{}\!\mathrm{d} {\mathbf{w}}_s$. Then ${\mathbf{x}}_t - {\mathbf{m}}(t) = A(t) {\mathbf{z}}_t$, so

$$
{\mathbf{P}}(t) = e^{2\int_0^t f(s) \mathop{}\!\mathrm{d} s} \mathbb{E}[{\mathbf{z}}_t {\mathbf{z}}_t^\top].
$$

By Itô isometry[^1],

$$
\mathbb{E}[{\mathbf{z}}_t {\mathbf{z}}_t^\top] = \left( \int_0^t \phi^2(s) \mathop{}\!\mathrm{d} s \right) {\mathbf{I}}_D,
$$

hence,

$$
{\mathbf{P}}(t) = e^{2\int_0^t f(s) \mathop{}\!\mathrm{d} s} \left( \int_0^t \left( e^{-\int_0^s f(u) \mathop{}\!\mathrm{d} u} g(s) \right)^2 \mathop{}\!\mathrm{d} s \right) {\mathbf{I}}_D.
$$

This shows the conditional covariance is isotropic.

**Derivation of Mean and Variance ODEs in [Equation (4.4.3)](../../chapters/04-score-sde-framework/04-sde-instantiations.md#eq-mean-var-evolution).** Alternatively, we can derive the moment evolution equations directly from the linear SDE [Equation (C.1.3)](01-ito-formula.md#eq-app-linear-sde).

**Mean Evolution.** Taking the conditional expectation of both sides of the SDE and using linearity:

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{m}}(t)}{\mathop{}\!\mathrm{d} t} = \mathbb{E}[f(t) {\mathbf{x}}_t|{\mathbf{x}}_0] = f(t) \mathbb{E}[{\mathbf{x}}_t|{\mathbf{x}}_0] = f(t) {\mathbf{m}}(t).
$$

**Covariance Evolution.** Define the centered process $\tilde{{\mathbf{x}}}_t := {\mathbf{x}}_t - {\mathbf{m}}(t)$. Applying Itô’s product rule (see [Equation (C.1.2)](01-ito-formula.md#eq-ito-product)):

$$
\mathop{}\!\mathrm{d} \left( \tilde{{\mathbf{x}}}_t \tilde{{\mathbf{x}}}_t^\top \right)
= \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t \cdot \tilde{{\mathbf{x}}}_t^\top
+ \tilde{{\mathbf{x}}}_t \cdot \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t^\top
+ \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t \cdot \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t^\top.
$$

From the SDE, we compute:

$$
\mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t = \mathop{}\!\mathrm{d} {\mathbf{x}}_t - \mathop{}\!\mathrm{d} {\mathbf{m}}(t) = f(t) \tilde{{\mathbf{x}}}_t  \mathop{}\!\mathrm{d} t + g(t)  \mathop{}\!\mathrm{d} {\mathbf{w}}_t.
$$

Substituting into the product rule and taking expectation:

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{P}}(t)}{\mathop{}\!\mathrm{d} t}
    &= \mathbb{E}[f(t) \tilde{{\mathbf{x}}}_t \tilde{{\mathbf{x}}}_t^\top + \tilde{{\mathbf{x}}}_t \tilde{{\mathbf{x}}}_t^\top f(t) + g^2(t) {\mathbf{I}}_D] \\
    &= 2 f(t) {\mathbf{P}}(t) + g^2(t) {\mathbf{I}}_D.
\end{aligned}
$$

Thus, we recover the moment evolution equations in [Equation (4.4.3)](../../chapters/04-score-sde-framework/04-sde-instantiations.md#eq-mean-var-evolution).

[^1]: Itô’s isometry links stochastic integrals to standard integrals in expectation; we omit the proof as it requires the full machinery of Itô calculus. For a process $\boldsymbol{\psi} : [0,T] \to \mathbb{R}^{D \times D}$, Itô isometry states

    $$
    \mathbb{E}\left[\left\| \int_0^T \boldsymbol{\psi}(t)   \mathrm{d}{\mathbf{w}}_t \right\|^2 \right]
    = \mathbb{E}\left[\int_0^T \|\boldsymbol{\psi}(t)\|_F^2   \mathrm{d}t \right],
    $$

    where $\|\boldsymbol{\psi}(t)\|_F^2 = \sum_{i,j=1}^D |\psi_{ij}(t)|^2$ is the Frobenius norm. For $\boldsymbol{\psi}(t) \in \mathbb{R}^D$ (a vector), the integral is scalar and the isometry simplifies to

    $$
    \mathbb{E}\left[\left(\int_0^T \boldsymbol{\psi}(t)   \mathrm{d}{\mathbf{w}}_t\right)^2\right] = \mathbb{E}\left[\int_0^T \|\boldsymbol{\psi}(t)\|^2   \mathrm{d}t \right].
    $$
