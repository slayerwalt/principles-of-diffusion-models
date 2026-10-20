# C.1 Itô 公式：随机过程的链式法则 {#sec-ito-formula}

标准微积分不能直接应用于随机过程，因为 Wiener 过程在经典意义下不可微。我们使用 Itô 微积分，它提供了处理随机积分的运算规则。

## C.1.1 动机：为什么需要特殊的链式法则？ {#motivation-special-chain-rule}

考虑一个确定性的时变函数 ${\mathbf{y}}_t$，它随时间 $t$ 光滑演化（例如 ODE 的解）。对于函数 ${\mathbf{h}}({\mathbf{y}}_t, t)$，普通链式法则给出：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{h}}}{\mathop{}\!\mathrm{d} t} = \frac{\partial {\mathbf{h}}}{\partial t}  + \nabla_{{\mathbf{y}}}{\mathbf{h}}  \frac{\mathop{}\!\mathrm{d} {\mathbf{y}}_t}{\mathop{}\!\mathrm{d} t} .
$$

这里，$\nabla_{{\mathbf{y}}}{\mathbf{h}}$ 是 ${\mathbf{h}}$ 的 Jacobian。对于确定性路径 ${\mathbf{y}}_t$，这一法则完全适用。

<span id="question-c-1-1"></span>

**问题 C.1.1**

如果 ${\mathbf{x}}_t$ 是随机过程，例如由 SDE

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = {\mathbf{f}}({\mathbf{x}}_t, t)\mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t
$$

驱动，如[式（A.2.3）](../a-differential-equations/02-stochastic-differential-equations.md#eq-sde)所示，又会怎样？过程 ${\mathbf{h}}({\mathbf{x}}_t, t)$ 满足什么 SDE？

**为什么普通链式法则不再适用？** 直接套用经典链式法则，会得到

$$
\mathop{}\!\mathrm{d} {\mathbf{h}} = \frac{\partial {\mathbf{h}}}{\partial t} \mathop{}\!\mathrm{d} t + \nabla_{{\mathbf{y}}}{\mathbf{h}}  \cdot \mathop{}\!\mathrm{d} {\mathbf{x}}_t.
$$

然而，Brownian 增量的尺度为 $\mathcal{O}(\sqrt{\mathop{}\!\mathrm{d} t})$，其二次变差在 Itô 微积分中由下式表达：

$$
(\mathop{}\!\mathrm{d} {\mathbf{w}}_t)^2 = \mathop{}\!\mathrm{d} t.
$$

因此，随机微积分中 $\mathop{}\!\mathrm{d} {\mathbf{w}}_t$ 的二阶项*不会*消失；这与经典微积分中 $(\mathop{}\!\mathrm{d} t)^2$ 项可以忽略的情形不同。

**示例：一个简单的例子——$h(x_t) = x_t^2$**

为形成直观认识，考虑简单的实值函数 $h(x_t) = x_t^2 \in \mathbb{R}$，其中随机变量 $x_t \in \mathbb{R}$ 满足

$$
\mathop{}\!\mathrm{d} x_t = \sigma \mathop{}\!\mathrm{d} w_t, \quad
x_0=0,
$$

且 $\sigma>0$ 为常数。若尝试使用经典链式法则，

$$
\mathop{}\!\mathrm{d} h = 2 x_t \mathop{}\!\mathrm{d} x_t = 2 x_t \sigma \mathop{}\!\mathrm{d} w_t.
$$

如果该式成立，$h(x_t)$ 的期望就应当不随时间变化，因为

$$
\mathbb{E}[\mathop{}\!\mathrm{d} h ] =  2 \sigma \mathbb{E}[x_t  \mathop{}\!\mathrm{d} w_t]= 2 \sigma \mathbb{E}[x_t]  \underbrace{\mathbb{E}[\mathop{}\!\mathrm{d} w_t]}_{=0}=0.
$$

但由 Brownian 运动的经典性质（见[式（A.2.4）](../a-differential-equations/02-stochastic-differential-equations.md#eq-wiener-gaussian)）可知，

$$
\mathbb{E}[x_t^2] = \sigma^2 t,
$$

它随时间线性增长。因此，普通链式法则遗漏了一个重要项。

## C.1.2 从 Taylor 展开推导一维 Itô 公式 {#subsec-1d-ito}

**通过 Taylor 展开得到确定性链式法则。** 为理解经典链式法则为何不适用于由 SDE 定义的随机过程，先通过 Taylor 展开回顾确定性情形。考虑标量情形：$y_t \in \mathbb{R}$ 和 $h(\cdot,\cdot)\in \mathbb{R}$。形式上取 $\mathop{}\!\mathrm{d} y_t = y_{t+\mathop{}\!\mathrm{d} t} - y_t$，其中 $\mathop{}\!\mathrm{d} t \approx 0$，展开得到：

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

这里，$\mathop{}\!\mathrm{d} t \mathop{}\!\mathrm{d} y_t = \left(\frac{\mathop{}\!\mathrm{d} y_t}{\mathop{}\!\mathrm{d} t}\right)(\mathop{}\!\mathrm{d} t)^2 = \mathcal{O}(\mathop{}\!\mathrm{d} t^2)$，类似地有 $(\mathop{}\!\mathrm{d} t)^2 = \mathcal{O}(\mathop{}\!\mathrm{d} t^2)$。因此，所有灰色部分均可忽略，全微分为：

$$
\mathop{}\!\mathrm{d} h
= \frac{\partial h}{\partial t} \mathop{}\!\mathrm{d} t
+ \frac{\partial h}{\partial y} \mathop{}\!\mathrm{d} y_t
+ \mathcal{O}(\mathop{}\!\mathrm{d} t^2).
$$

**通过随机 Taylor 展开得到 Itô 公式。** 现在考虑随机过程 $x_t \in \mathbb{R}$，它由以下 SDE 支配：

$$
\mathop{}\!\mathrm{d} x_t = f(x_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} w_t,
$$

其中 $w_t$ 为标准 Brownian 运动。我们的目标是计算标量值函数 $h(x_t, t)$ 的微分。

使用随机 Taylor 展开（[Kloeden et al., 1992](../../references.md#ref-kloeden1992stochastic)），保留 $\mathop{}\!\mathrm{d} x_t$ 的二阶项，得到：

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

**可忽略的交叉项。** 根据 Brownian 运动的尺度性质（[式（A.2.4）](../a-differential-equations/02-stochastic-differential-equations.md#eq-wiener-gaussian)），

$$
\mathop{}\!\mathrm{d} w_t = \mathcal{O}(\sqrt{\mathop{}\!\mathrm{d} t}) \quad \Rightarrow \quad \mathop{}\!\mathrm{d} t \cdot \mathop{}\!\mathrm{d} w_t = \mathcal{O}((\mathop{}\!\mathrm{d} t)^{3/2}).
$$

因此，

$$
\begin{aligned}
\mathop{}\!\mathrm{d} t \cdot \mathop{}\!\mathrm{d} x_t
&= \mathop{}\!\mathrm{d} t \left(f \mathop{}\!\mathrm{d} t + g \mathop{}\!\mathrm{d} w_t\right)
= f (\mathop{}\!\mathrm{d} t)^2 + g \cdot \mathop{}\!\mathrm{d} t \cdot \mathop{}\!\mathrm{d} w_t
= \mathcal{O}((\mathop{}\!\mathrm{d} t)^{3/2}).
\end{aligned}
$$

所以灰色项均可忽略，它们为 $\mathcal{O}((\mathop{}\!\mathrm{d} t)^{3/2})$ 或更小量级。

**二阶项 $(\mathop{}\!\mathrm{d} x_t)^2$。** 利用 SDE 展开：

$$
\begin{aligned}
(\mathop{}\!\mathrm{d} x_t)^2
&= \left( f \mathop{}\!\mathrm{d} t + g \mathop{}\!\mathrm{d} w_t \right)^2 \\
&= f^2 (\mathop{}\!\mathrm{d} t)^2 + 2fg \mathop{}\!\mathrm{d} t \mathop{}\!\mathrm{d} w_t + g^2 (\mathop{}\!\mathrm{d} w_t)^2 \\
&= g^2(t) \mathop{}\!\mathrm{d} t,
\end{aligned}
$$

最后一个等式使用了 Itô 微分规则 $(\mathop{}\!\mathrm{d} t)^2=0$、$\mathop{}\!\mathrm{d} t\,\mathop{}\!\mathrm{d} w_t=0$ 和 $(\mathop{}\!\mathrm{d} w_t)^2=\mathop{}\!\mathrm{d} t$，其中最后一个恒等式应在二次变差意义下理解。

合并各项，得到微分：

$$
\mathop{}\!\mathrm{d} h(x_t, t)
= \frac{\partial h}{\partial t} \mathop{}\!\mathrm{d} t
+ \frac{\partial h}{\partial x} \mathop{}\!\mathrm{d} x_t
+ \frac{1}{2} \frac{\partial^2 h}{\partial x^2} g^2(t) \mathop{}\!\mathrm{d} t.
$$

代入 $\mathop{}\!\mathrm{d} x_t = f(x_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} w_t$，得到：

$$
\mathop{}\!\mathrm{d} h(x_t, t)
= \left( \frac{\partial h}{\partial t} + f \frac{\partial h}{\partial x}
+ \frac{1}{2} g^2 \frac{\partial^2 h}{\partial x^2} \right) \mathop{}\!\mathrm{d} t
+ g \frac{\partial h}{\partial x} \mathop{}\!\mathrm{d} w_t.
$$

这就是一维 *Itô 公式*。

**示例：一个简单的例子——$h(x_t) = x_t^2$**

重新考察前面的简单例子：$h(x_t) = x_t^2$，其中随机过程 $x_t \in \mathbb{R}$ 满足

$$
\mathop{}\!\mathrm{d} x_t = \sigma \mathop{}\!\mathrm{d} w_t,
$$

且 $\sigma > 0$ 为常数。对 $h(x_t) = x_t^2$ 正确应用 Itô 公式，得到：

$$
\mathop{}\!\mathrm{d} h(x_t) = \mathop{}\!\mathrm{d} (x_t^2) = 2 x_t \mathop{}\!\mathrm{d} x_t + \sigma^2 \mathop{}\!\mathrm{d} t.
$$

代入 $\mathop{}\!\mathrm{d} x_t = \sigma \mathop{}\!\mathrm{d} w_t$，可得：

$$
\mathop{}\!\mathrm{d}(x_t^2) = 2 x_t \sigma \mathop{}\!\mathrm{d} w_t + \sigma^2 \mathop{}\!\mathrm{d} t.
$$

## C.1.3 Itô 公式：SDE 的链式法则 {#subsec-ito-summary}

下面归纳前面推导的一维 Itô 公式。采用类似论证，可以自然地推广到多维情形。这里省略详细推导，但为完整起见，仍给出一般公式。

最后，通过推导 *Itô 乘积法则*展示 Itô 公式的一个应用；该法则可用于计算 $\mathop{}\!\mathrm{d}({\mathbf{x}}_t^\top {\mathbf{y}}_t)$，其中 ${\mathbf{x}}_t$ 和 ${\mathbf{y}}_t$ 为随机过程。

**$1$ 维 Itô 公式。** 设 $x_t \in \mathbb{R}$ 为满足以下 SDE 的随机过程：

$$
\mathop{}\!\mathrm{d} x_t = f(x_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} w_t.
$$

对于标量函数 $h \colon \mathbb{R} \times [0, T] \to \mathbb{R}$，过程 $h(x_t, t)$ 满足：

$$
\mathop{}\!\mathrm{d} h(x_t, t)
= \left( \frac{\partial h}{\partial t} + f \frac{\partial h}{\partial x}
+ \frac{1}{2} g^2 \frac{\partial^2 h}{\partial x^2} \right) \mathop{}\!\mathrm{d} t
+ g \frac{\partial h}{\partial x} \mathop{}\!\mathrm{d} w_t.
$$

**标量输出的多维 Itô 公式。** 设 ${\mathbf{x}}_t \in \mathbb{R}^D$ 满足 SDE：

$$
\mathop{}\!\mathrm{d} {\mathbf{x}}_t = {\mathbf{f}}({\mathbf{x}}_t, t) \mathop{}\!\mathrm{d} t + g(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t,
$$

其中 ${\mathbf{f}} \colon \mathbb{R}^D \times [0, T] \to \mathbb{R}^D$、$g \colon [0, T] \to \mathbb{R}$，${\mathbf{w}}_t \in \mathbb{R}^D$ 是 $D$ 维 Brownian 运动。设 $h \colon \mathbb{R}^D \times [0, T] \to \mathbb{R}$ 为标量值函数，则 $h({\mathbf{x}}_t, t)$ 满足：

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

其中 $\nabla_{{\mathbf{x}}} h \in \mathbb{R}^D$ 和 $\nabla^2_{{\mathbf{x}}} h \in \mathbb{R}^{D \times D}$ 分别是 $h$ 关于 ${\mathbf{x}}$ 的梯度和 Hessian 矩阵。

**示例：Itô 乘积法则**

设 ${\mathbf{x}}_t, {\mathbf{y}}_t \in \mathbb{R}^D$ 为由以下 SDE 支配的向量值随机过程：

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}}_t &= {\mathbf{a}}({\mathbf{x}}_t, t) \mathop{}\!\mathrm{d} t + b(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t, \\
\mathop{}\!\mathrm{d} {\mathbf{y}}_t &= {\mathbf{c}}({\mathbf{y}}_t, t) \mathop{}\!\mathrm{d} t + d(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t,
\end{aligned}
$$

其中 ${\mathbf{a}}, {\mathbf{c}}: \mathbb{R}^D \times [0, T] \to \mathbb{R}^D$ 为向量场，$b(t), d(t) \in \mathbb{R}$ 为标量值函数。这里，${\mathbf{w}}_t \in \mathbb{R}^D$ 表示标准的 $D$ 维 Brownian 运动。

我们希望推导以下标量值过程的 SDE：

$$
z(t) := {\mathbf{x}}_t^\top {\mathbf{y}}_t.
$$

对双线性函数 $h({\mathbf{x}}, {\mathbf{y}}) := {\mathbf{x}}^\top {\mathbf{y}}$ 应用多元 Itô 公式，得到：

$$
\mathop{}\!\mathrm{d}({\mathbf{x}}^\top {\mathbf{y}}) = (\mathop{}\!\mathrm{d} {\mathbf{x}})^\top {\mathbf{y}} + {\mathbf{x}}^\top \mathop{}\!\mathrm{d} {\mathbf{y}} + \operatorname{Tr}\left[ \mathop{}\!\mathrm{d} {\mathbf{x}} \cdot (\mathop{}\!\mathrm{d} {\mathbf{y}})^\top \right].
$$

Itô 修正项计算如下：

$$
\begin{aligned}
\mathop{}\!\mathrm{d} {\mathbf{x}} \cdot (\mathop{}\!\mathrm{d} {\mathbf{y}})^\top &= b(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t \cdot \left[ d(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t \right]^\top \\&= b(t) d(t) \mathop{}\!\mathrm{d} {\mathbf{w}}_t \cdot \mathop{}\!\mathrm{d} {\mathbf{w}}_t^\top \\&= b(t)d(t) \mathop{}\!\mathrm{d} t \cdot {\mathbf{I}}_D.
\end{aligned}
$$

因此，

$$
\operatorname{Tr}\left[ \mathop{}\!\mathrm{d} {\mathbf{x}} \cdot (\mathop{}\!\mathrm{d} {\mathbf{y}})^\top \right] = b(t)d(t) \operatorname{Tr}({\mathbf{I}}_D) \mathop{}\!\mathrm{d} t = D  b(t)d(t) \mathop{}\!\mathrm{d} t.
$$

合并所有项，得到的 SDE 为：

<span id="eq-ito-product"></span>

$$
\begin{aligned}
        \mathop{}\!\mathrm{d}({\mathbf{x}}^\top {\mathbf{y}}) &= (\mathop{}\!\mathrm{d} {\mathbf{x}})^\top {\mathbf{y}} + {\mathbf{x}}^\top \mathop{}\!\mathrm{d} {\mathbf{y}} + D  b(t)d(t) \mathop{}\!\mathrm{d} t
\end{aligned}
\tag{C.1.2}
$$

## C.1.4 Itô 公式的应用：推导 Fokker–Planck 方程 {#subsec-rigorous-proof-fpe}

本节应用[式（C.1.1）](01-ito-formula.md#eq-ito-scalar)中的 Itô 公式，推导 Fokker–Planck 方程。这个 PDE 描述概率密度 $p_t({\mathbf{x}})$ 随时间的演化，该密度对应于 $D$ 维扩散过程，由[式（A.2.3）](../a-differential-equations/02-stochastic-differential-equations.md#eq-sde)中的 SDE 定义。

**步骤 1：应用 Itô 公式。** 设 $\phi({\mathbf{x}}, t)$ 为光滑测试函数 $\phi \colon \mathbb{R}^D \times [0, T] \to \mathbb{R}$。由 Itô 公式，

$$
\mathop{}\!\mathrm{d} \phi({\mathbf{x}}_t, t) = \left( \frac{\partial \phi}{\partial t} + \nabla_{{\mathbf{x}}} \phi^\top {\mathbf{f}}({\mathbf{x}}_t, t) + \frac{1}{2} g^2(t) \operatorname{Tr}[\nabla_{{\mathbf{x}}}^2 \phi] \right) \mathop{}\!\mathrm{d} t + g(t) \nabla_{{\mathbf{x}}} \phi^\top \mathop{}\!\mathrm{d} {\mathbf{w}}_t.
$$

**步骤 2：取期望。** 两边取期望，并利用在标准可积性条件下 Itô 积分的期望为零这一事实：

$$
\mathbb{E}[\mathop{}\!\mathrm{d} \phi({\mathbf{x}}_t, t)] = \mathbb{E} \left[ \left( \frac{\partial \phi}{\partial t} + \nabla_{{\mathbf{x}}} \phi^\top {\mathbf{f}}({\mathbf{x}}_t, t) + \frac{1}{2} g^2(t) \operatorname{Tr}[\nabla_{{\mathbf{x}}}^2 \phi] \right) \mathop{}\!\mathrm{d} t \right].
$$

**步骤 3：通过密度表示期望。** 该期望可写成：

$$
\mathbb{E}[\mathop{}\!\mathrm{d} \phi({\mathbf{x}}_t, t)] = \int \left( \frac{\partial \phi}{\partial t} + \nabla_{{\mathbf{x}}} \phi^\top {\mathbf{f}}({\mathbf{x}}, t) + \frac{1}{2} g^2(t) \operatorname{Tr}[\nabla_{{\mathbf{x}}}^2 \phi] \right) p_t({\mathbf{x}}) \mathop{}\!\mathrm{d} {\mathbf{x}} \mathop{}\!\mathrm{d} t.
$$

**步骤 4：分部积分。** 在 $\mathbb{R}^D$ 上使用分部积分（散度定理）：

$$
\begin{aligned}
\int \nabla_{{\mathbf{x}}} \phi^\top {\mathbf{f}} p_t \mathop{}\!\mathrm{d} {\mathbf{x}} &= -\int \phi \nabla_{{\mathbf{x}}} \cdot ({\mathbf{f}} p_t) \mathop{}\!\mathrm{d} {\mathbf{x}}, \\
    \int \operatorname{Tr}[\nabla_{{\mathbf{x}}}^2 \phi] p_t  \mathop{}\!\mathrm{d} {\mathbf{x}} &= \int \phi \Delta p_t \mathop{}\!\mathrm{d} {\mathbf{x}}.
\end{aligned}
$$

**步骤 5：代入并整理。** 注意到

$$
\mathbb{E}[\phi({\mathbf{x}}_t,t)] = \int \phi({\mathbf{x}},t) p_t({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

对时间求导，有

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\mathbb{E}[\phi({\mathbf{x}}_t,t)]
= \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\int \phi({\mathbf{x}},t) p_t({\mathbf{x}}) \mathop{}\!\mathrm{d}{\mathbf{x}}
= \int\Bigl(\partial_t\phi({\mathbf{x}},t) p_t({\mathbf{x}}) + \phi({\mathbf{x}},t) \partial_t p_t({\mathbf{x}})\Bigr)\mathop{}\!\mathrm{d}{\mathbf{x}}.
$$

因此，

$$
\mathbb{E}[\mathop{}\!\mathrm{d} \phi({\mathbf{x}}_t,t)]
= \frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}\mathbb{E}[\phi({\mathbf{x}}_t,t)] \mathop{}\!\mathrm{d} t
= \int\Bigl(\partial_t\phi({\mathbf{x}},t) p_t({\mathbf{x}}) + \phi({\mathbf{x}},t) \partial_t p_t({\mathbf{x}})\Bigr)\mathop{}\!\mathrm{d}{\mathbf{x}} \mathop{}\!\mathrm{d} t.
$$

将该表达式与步骤 3 中的结果相等，$\int \partial_t\phi\,p_t\,\mathop{}\!\mathrm{d}{\mathbf{x}}\,\mathop{}\!\mathrm{d} t$ 项抵消。对剩余空间项应用步骤 4，得到

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

由于这对每个光滑测试函数 $\phi$ 都成立，得到 Fokker–Planck 方程

$$
\frac{\partial p_t({\mathbf{x}})}{\partial t}
=
-\nabla_{{\mathbf{x}}}\cdot\bigl({\mathbf{f}}({\mathbf{x}},t)p_t({\mathbf{x}})\bigr)
+
\frac{1}{2}g^2(t)\Delta p_t({\mathbf{x}}).
$$

## C.1.5 Itô 公式的应用：线性 SDE 的闭式解 {#subsec-rigorous-proof-linear-sde}

本小节说明如何使用积分因子（类似于 ODE 的情形）和 Itô 公式求得线性 SDE 的闭式解。这种方法沿用求解线性 ODE 的经典技巧，并将其调整到随机情形。

考虑如下形式的线性 SDE：

<span id="eq-app-linear-sde"></span>

$$
\begin{aligned}
\mathrm{d}{\mathbf{x}}_t = f(t){\mathbf{x}}_t \mathrm{d}t + g(t) \mathrm{d}{\mathbf{w}}_t,
\end{aligned}
\tag{C.1.3}
$$

其中 $f(t)$ 和 $g(t)$ 为确定性函数，${\mathbf{w}}_t$ 为标准 Wiener 过程。

**线性 SDE 的闭式解。** 使用积分因子法，推导[式（C.1.3）](01-ito-formula.md#eq-app-linear-sde)中线性 SDE 的显式解。这类前向 SDE 常见于扩散模型（见[第 4.1 节](../../chapters/04-score-sde-framework/01-score-sde-principles.md#sec-dm-today)）。

**步骤 1：定义积分因子。** 令

$$
\Psi(t) := \exp\left( -\int_0^t f(s) \mathrm{d}s \right),
\quad \text{and define} \quad
{\mathbf{y}}_t := \Psi(t){\mathbf{x}}_t.
$$

**步骤 2：应用 Itô 公式。** 对函数 ${\mathbf{h}}({\mathbf{x}}, t) := \Psi(t){\mathbf{x}}$ 应用 Itô 公式。这实际上是[式（C.1.2）](01-ito-formula.md#eq-ito-product)中 Itô 乘积法则的一个特例。由于 $\Psi(t)$ 为确定性函数，不存在交叉变差项，因此公式简化为：

$$
\begin{aligned}
\mathrm{d}{\mathbf{y}}_t
    &= \mathrm{d}[\Psi(t){\mathbf{x}}_t] \\
    &= \Psi'(t){\mathbf{x}}_t \mathrm{d}t + \Psi(t) \mathrm{d}{\mathbf{x}}_t \\
    &= -f(t)\Psi(t){\mathbf{x}}_t \mathrm{d}t + \Psi(t)\left[f(t){\mathbf{x}}_t \mathrm{d}t + g(t) \mathrm{d}{\mathbf{w}}_t\right] \\
    &= \Psi(t)g(t) \mathrm{d}{\mathbf{w}}_t.
\end{aligned}
$$

于是，

$$
{\mathbf{y}}_t = {\mathbf{y}}_0 + \int_0^t \Psi(s)g(s) \mathrm{d}{\mathbf{w}}(s) = {\mathbf{x}}_0 + \int_0^t \Psi(s)g(s) \mathrm{d}{\mathbf{w}}(s),
$$

因为 $\Psi(0) = 1$。

**步骤 3：求解 ${\mathbf{x}}_t$。** 利用 ${\mathbf{x}}_t = \Psi(t)^{-1}{\mathbf{y}}_t$，得到

<span id="eq-lienar-sde-solution"></span>

$$
\begin{aligned}
{\mathbf{x}}_t &=  e^{ \int_0^t f(s)\mathrm{d}s}
    \left[ {\mathbf{x}}_0 + \int_0^t e^{ -\int_0^s f(r) \mathrm{d}r } g(s) \mathrm{d}{\mathbf{w}}(s) \right].
\end{aligned}
\tag{C.1.4}
$$

由此得到向量值 SDE 的显式解。

下面通过两种方法计算 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 的解析形式。

**闭式解的分析。** [式（C.1.4）](01-ito-formula.md#eq-lienar-sde-solution)再次说明 $p_t({\mathbf{x}}_t|{\mathbf{x}}_0)$ 是 Gaussian 的。为看清这一点，定义

$$
\phi(s) := e^{-\int_0^s f(u) \mathop{}\!\mathrm{d} u} g(s),
$$

它是关于时间的确定性矩阵值函数（假设 $f(u)$ 和 $g(s)$ 为确定性函数）。Itô 积分 $\int_0^t \phi(s) \mathop{}\!\mathrm{d} {\mathbf{w}}_s$ 是零均值 Gaussian 随机变量，因为它是确定性函数关于 Brownian 运动的随机积分。因此，${\mathbf{x}}_t|{\mathbf{x}}_0$ 是 Gaussian 随机变量的仿射变换，其自身也服从 Gaussian 分布。它的分布由条件均值与协方差完全刻画。

定义条件均值和协方差（给定初始条件 ${\mathbf{x}}_0$）为

$$
{\mathbf{m}}(t) := \mathbb{E}[{\mathbf{x}}_t | {\mathbf{x}}_0], \quad {\mathbf{P}}(t) := \mathbb{E}[({\mathbf{x}}_t - {\mathbf{m}}(t))({\mathbf{x}}_t - {\mathbf{m}}(t))^\top|{\mathbf{x}}_0].
$$

**均值。** 利用期望的线性性，以及确定性函数的 Itô 积分均值为零这一事实：

$$
\begin{aligned}
{\mathbf{m}}(t) = \mathbb{E}\left[ e^{\int_0^t f(s) \mathop{}\!\mathrm{d} s} \left( {\mathbf{x}}_0 + \int_0^t \phi(s) \mathop{}\!\mathrm{d} {\mathbf{w}}_s \right) \bigg| {\mathbf{x}}_0 \right] = e^{\int_0^t f(s) \mathop{}\!\mathrm{d} s} {\mathbf{x}}_0.
\end{aligned}
$$

**协方差。** 令 ${\mathbf{z}}_t := \int_0^t \phi(s) \mathop{}\!\mathrm{d} {\mathbf{w}}_s$，则 ${\mathbf{x}}_t - {\mathbf{m}}(t) = A(t) {\mathbf{z}}_t$，因此

$$
{\mathbf{P}}(t) = e^{2\int_0^t f(s) \mathop{}\!\mathrm{d} s} \mathbb{E}[{\mathbf{z}}_t {\mathbf{z}}_t^\top].
$$

由 Itô 等距公式[^1]，

$$
\mathbb{E}[{\mathbf{z}}_t {\mathbf{z}}_t^\top] = \left( \int_0^t \phi^2(s) \mathop{}\!\mathrm{d} s \right) {\mathbf{I}}_D,
$$

因此，

$$
{\mathbf{P}}(t) = e^{2\int_0^t f(s) \mathop{}\!\mathrm{d} s} \left( \int_0^t \left( e^{-\int_0^s f(u) \mathop{}\!\mathrm{d} u} g(s) \right)^2 \mathop{}\!\mathrm{d} s \right) {\mathbf{I}}_D.
$$

这表明条件协方差是各向同性的。

**[式（4.4.3）](../../chapters/04-score-sde-framework/04-sde-instantiations.md#eq-mean-var-evolution)中的均值与方差 ODE 的推导。** 另一种方法是直接从线性 SDE [式（C.1.3）](01-ito-formula.md#eq-app-linear-sde)推导矩的演化方程。

**均值的演化。** 对 SDE 两边取条件期望，并利用线性性：

$$
\frac{\mathop{}\!\mathrm{d} {\mathbf{m}}(t)}{\mathop{}\!\mathrm{d} t} = \mathbb{E}[f(t) {\mathbf{x}}_t|{\mathbf{x}}_0] = f(t) \mathbb{E}[{\mathbf{x}}_t|{\mathbf{x}}_0] = f(t) {\mathbf{m}}(t).
$$

**协方差的演化。** 定义中心化过程 $\tilde{{\mathbf{x}}}_t := {\mathbf{x}}_t - {\mathbf{m}}(t)$，应用 Itô 乘积法则（见[式（C.1.2）](01-ito-formula.md#eq-ito-product)）：

$$
\mathop{}\!\mathrm{d} \left( \tilde{{\mathbf{x}}}_t \tilde{{\mathbf{x}}}_t^\top \right)
= \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t \cdot \tilde{{\mathbf{x}}}_t^\top
+ \tilde{{\mathbf{x}}}_t \cdot \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t^\top
+ \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t \cdot \mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t^\top.
$$

由 SDE 计算得到：

$$
\mathop{}\!\mathrm{d} \tilde{{\mathbf{x}}}_t = \mathop{}\!\mathrm{d} {\mathbf{x}}_t - \mathop{}\!\mathrm{d} {\mathbf{m}}(t) = f(t) \tilde{{\mathbf{x}}}_t  \mathop{}\!\mathrm{d} t + g(t)  \mathop{}\!\mathrm{d} {\mathbf{w}}_t.
$$

代入乘积法则并取期望：

$$
\begin{aligned}
\frac{\mathop{}\!\mathrm{d} {\mathbf{P}}(t)}{\mathop{}\!\mathrm{d} t}
    &= \mathbb{E}[f(t) \tilde{{\mathbf{x}}}_t \tilde{{\mathbf{x}}}_t^\top + \tilde{{\mathbf{x}}}_t \tilde{{\mathbf{x}}}_t^\top f(t) + g^2(t) {\mathbf{I}}_D] \\
    &= 2 f(t) {\mathbf{P}}(t) + g^2(t) {\mathbf{I}}_D.
\end{aligned}
$$

这样就重新得到了[式（4.4.3）](../../chapters/04-score-sde-framework/04-sde-instantiations.md#eq-mean-var-evolution)中的矩演化方程。

[^1]: Itô 等距公式在期望意义下将随机积分与普通积分联系起来；其证明需要 Itô 微积分的完整理论，这里省略。对于过程 $\boldsymbol{\psi} : [0,T] \to \mathbb{R}^{D \times D}$，Itô 等距公式为

    $$
    \mathbb{E}\left[\left\| \int_0^T \boldsymbol{\psi}(t)   \mathrm{d}{\mathbf{w}}_t \right\|^2 \right]
    = \mathbb{E}\left[\int_0^T \|\boldsymbol{\psi}(t)\|_F^2   \mathrm{d}t \right],
    $$

    其中 $\|\boldsymbol{\psi}(t)\|_F^2 = \sum_{i,j=1}^D |\psi_{ij}(t)|^2$ 为 Frobenius 范数。对于 $\boldsymbol{\psi}(t) \in \mathbb{R}^D$（向量），积分为标量，等距公式简化为

    $$
    \mathbb{E}\left[\left(\int_0^T \boldsymbol{\psi}(t)   \mathrm{d}{\mathbf{w}}_t\right)^2\right] = \mathbb{E}\left[\int_0^T \|\boldsymbol{\psi}(t)\|^2   \mathrm{d}t \right].
    $$
