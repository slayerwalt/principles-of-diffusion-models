# D.5 理论补充：学习基于扩散的快速生成器 {#app-sec-flow-map}

## D.5.1 知识蒸馏损失是一般框架[式（10.1.4）](../../chapters/10-distillation/01-prologue.md#eq-general-cm-loss)的一个实例 {#knowledge-distillation-framework}

从 oracle KD 损失出发，

$$
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}({\boldsymbol{\theta}})
={\mathbb{E}}_{{\mathbf{x}}_T\sim p_T}\,
\big\|{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{x}}_T,T,0)-{\boldsymbol{\Psi}}_{T\to 0}({\mathbf{x}}_T)\big\|_2^2,
$$

其中 $p_T = p_{\mathrm{prior}}$。对于确定性 ODE 流映射 ${\boldsymbol{\Psi}}$（满足半群性质，且沿轨迹为双射），其边缘分布满足推前恒等式

$$
p_t
= {\boldsymbol{\Psi}}_{0\to t}\,\sharp\,p_{\mathrm{data}}
= {\boldsymbol{\Psi}}_{T\to t}\,\sharp\,p_{\mathrm{prior}}.
$$

因此，

$$
{\boldsymbol{\Psi}}_{s\to T}\,\sharp\,p_s=p_T
\quad\text{and}\quad
{\boldsymbol{\Psi}}_{T\to 0}\circ{\boldsymbol{\Psi}}_{s\to T}={\boldsymbol{\Psi}}_{s\to 0}.
$$

进行变量替换 ${\mathbf{x}}_T={\boldsymbol{\Psi}}_{s\to T}({\mathbf{x}}_s)$，其中 ${\mathbf{x}}_s\sim p_s$，得到

$$
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}({\boldsymbol{\theta}})
= {\mathbb{E}}_{{\mathbf{x}}_s\sim p_s}
\Big\|{\mathbf{G}}_{\boldsymbol{\theta}} \big({\boldsymbol{\Psi}}_{s\to T}({\mathbf{x}}_s),T,0\big)
- {\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)\Big\|_2^2.
$$

定义拉回后的学生模型 $\widetilde{{\mathbf{G}}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,0):= {\mathbf{G}}_{\boldsymbol{\theta}}({\boldsymbol{\Psi}}_{s\to T}({\mathbf{x}}_s),T,0)$，即可将同一损失写成统一的流映射形式（取 $t=0$）：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{KD}}^{\text{oracle}}({\boldsymbol{\theta}})
= {\mathbb{E}}_{{\mathbf{x}}_s\sim p_s}
\big\|\widetilde{{\mathbf{G}}}_{\boldsymbol{\theta}}({\mathbf{x}}_s,s,0)-{\boldsymbol{\Psi}}_{s\to 0}({\mathbf{x}}_s)\big\|_2^2.
\end{aligned}
$$

这一推导依赖于通过 oracle 流进行的变量替换和半群性质。

$\blacksquare$

**传输解释。** [命题 10.2.1](../../chapters/10-distillation/02-distribution-based-distillation.md#vsd-grad)也可以从分布的角度理解。对于

$$
\hat{\mathbf{x}}_t
=
\alpha_t{\mathbf{G}}_{\boldsymbol{\theta}}({\mathbf{z}})+\sigma_t{\boldsymbol{\epsilon}},
$$

参数的无穷小扰动会引起数据空间中的位移 $(\partial_{\boldsymbol{\theta}}\hat{\mathbf{x}}_t)\,\mathop{}\!\mathrm{d}{\boldsymbol{\theta}}$。相应地，VSD 梯度可写为

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

因此，score 的差异提供了数据空间中的梯度信号，而生成器的 Jacobian 将该信号映射回参数空间；梯度下降随后沿相反方向更新参数。这是[第 B.3 节](../b-density-evolution/03-wasserstein-gradient-flows.md#sec-wgf-distribution-level-training)中分布层面的传输视角在参数化情形下的对应表述。

## D.5.2 [定理 11.2.1](../../chapters/11-fast-generators/02-discrete-consistency-models.md#thm-ct-cm)：CM 与 CT 相等，误差为 $\mathcal{O}(\Delta s^2)$ {#cm-ct-equivalence}

***证明：***

**步骤 1：使用 oracle score 的 DDIM 更新是条件均值。**

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

这里在第三、第四个等式中使用了 Tweedie 公式

$$
{\mathbb{E}}[{\mathbf{x}}_0 |{\mathbf{x}}_s]=\frac{{\mathbf{x}}_s+\sigma_s^2\nabla_{{\mathbf{x}}_s}\log p_s({\mathbf{x}}_s)}{\alpha_s},
$$

以及共享噪声耦合

$$
{\mathbf{x}}_s=\alpha_s{\mathbf{x}}_0+\sigma_s{\boldsymbol{\epsilon}},
\qquad
{\mathbf{x}}_{s'}=\alpha_{s'}{\mathbf{x}}_0+\sigma_{s'}{\boldsymbol{\epsilon}}
\quad (s'=s-\Delta s),
$$

还应注意，更准确地说，$\hat{\mathbf{x}}_{s'}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s]$ 应理解为 $\hat{\mathbf{x}}_{s'}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s, s]$，因为 $s'=s-\Delta s$ 在 $s$ 被抽样后就已经确定。

**步骤 2：围绕条件均值 $\hat{\mathbf{x}}_{s'}$ 展开 CT。**

回顾其构造：先抽取 $s$，再以确定的方式设定 $s':=s-\Delta s$。

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

这里，(1) 对

$$
h({\mathbf{x}}') := d\big({\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s), {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}',s')\big)
$$

的第二个自变量在 $\hat{\mathbf{x}}_{s'}$ 附近作二阶 Taylor 展开。(2) 使用*塔式性质*（也称全期望公式）：

$$
{\mathbb{E}}_{s,{\mathbf{x}}_s}{\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s}[\cdot]
={\mathbb{E}}_{s,{\mathbf{x}}_s}[\cdot],
$$

并注意到，在 ${\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s}$ 内部，各项

$$
w(s),\ {\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s),\ \hat{\mathbf{x}}_{s'},\ s'
$$

不依赖于随机变量 ${\mathbf{x}}_{s'}$（因为 $s' = s-\Delta s$ 和 $\hat{\mathbf{x}}_{s'}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s]$ 在 $(s,{\mathbf{x}}_s)$ 固定后就已经确定）。因此，对 ${\mathbf{x}}_{s'}$ 的依赖仅通过 $({\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'})$ 产生，内层期望化为 ${\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s}({\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'})$ 和 ${\mathbb{E}}_{{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s}\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2$。(3) 使用了

$$
{\mathbb{E}}[{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s]=\boldsymbol 0
\quad\text{because}\quad
\hat{\mathbf{x}}_{s'}={\mathbb{E}}[{\mathbf{x}}_{s'}|{\mathbf{x}}_s,s].
$$

对于 (4)，注意到

$$
{\mathbb{E}}\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2
={\mathbb{E}}\,\operatorname{Var}({\mathbf{x}}_{s'}|{\mathbf{x}}_s,s)
\le {\mathbb{E}}\|{\mathbf{x}}_{s'}-{\mathbf{x}}_s\|^2.
$$

此外，

$$
{\mathbf{x}}_{s'}-{\mathbf{x}}_s=(\alpha_{s'}-\alpha_s){\mathbf{x}}_0+(\sigma_{s'}-\sigma_s){\boldsymbol{\epsilon}},
$$

对于光滑的调度函数，有 $\alpha_{s'}-\alpha_s=\mathcal{O}(\Delta s)$ 和 $\sigma_{s'}-\sigma_s=\mathcal{O}(\Delta s)$，因此

$$
{\mathbb{E}}\|{\mathbf{x}}_{s'}-{\mathbf{x}}_s\|^2=\mathcal{O}(\Delta s^2),
$$

从而得到

$$
{\mathbb{E}}\|{\mathbf{x}}_{s'}-\hat{\mathbf{x}}_{s'}\|^2=\mathcal{O}(\Delta s^2).
$$

$\blacksquare$

## D.5.3 [命题 11.3.1](../../chapters/11-fast-generators/03-continuous-consistency-models.md#continuous-time-ct)：CT 梯度的连续时间极限 {#continuous-time-ct-gradient}

***证明：***

为简化记号，将[式（11.3.2）](../../chapters/11-fast-generators/03-continuous-consistency-models.md#eq-cm-discrete)写为

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

为方便起见，记

$$
\tilde{\mathbf{x}}_{s-\Delta s}:={\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s).
$$

定义

$$
\delta {\mathbf{f}}
:= {\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s)-{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s).
$$

则

$$
\begin{aligned}
\big\|{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s, s)- {\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\tilde{\mathbf{x}}_{s-\Delta s}, s-\Delta s)\big\|_2^2
=
\Big\|\delta {\mathbf{f}}
+
\Big({\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s, s)-{\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\tilde{\mathbf{x}}_{s-\Delta s}, s-\Delta s)\Big)\Big\|_2^2.
\end{aligned}
$$

在 $({\mathbf{x}}_s,s)$ 处作 Taylor 展开，得到

$$
\begin{aligned}
{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s) -
{\mathbf{f}}_{{\boldsymbol{\theta}}^-}(\tilde{\mathbf{x}}_{s-\Delta s}, s-\Delta s)
=
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\,\Delta s
+ \mathcal{O}(|\Delta s|^2),
\end{aligned}
$$

其中使用了 oracle 流映射的一阶展开，

$$
{\mathbf{x}}_s - {\boldsymbol{\Psi}}_{s\to s-\Delta s}({\mathbf{x}}_s)
= {\mathbf{v}}^*({\mathbf{x}}_s,s)\,\Delta s + \mathcal{O}(|\Delta s|^2),
$$

以及全微分恒等式

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} s}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
=
\partial_s {\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
+
\big(\partial_{{\mathbf{x}}}{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)\big)\,{\mathbf{v}}^*({\mathbf{x}}_s,s).
$$

因此，

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

现在对 ${\boldsymbol{\theta}}$ 求导。由于 ${\boldsymbol{\theta}}^-$ 相对于 $\nabla_{{\boldsymbol{\theta}}}$ 被视为固定量（即梯度不经过停止梯度的目标），因此没有梯度经过 ${\mathbf{f}}_{{\boldsymbol{\theta}}^-}$。于是

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

在目标网络为在线网络的停止梯度副本这一比较点上，即 ${\boldsymbol{\theta}}={\boldsymbol{\theta}}^-$，有

$$
\delta {\mathbf{f}}
=
{\mathbf{f}}_{{\boldsymbol{\theta}}}({\mathbf{x}}_s,s)-{\mathbf{f}}_{{\boldsymbol{\theta}}^-}({\mathbf{x}}_s,s)
=
0,
$$

因此

$$
\nabla_{{\boldsymbol{\theta}}}{\mathbb{E}}\!\left[\omega(s)\|\delta {\mathbf{f}}\|_2^2\right]=0.
$$

于是，

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

除以 $\Delta s$ 并取极限，得到

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

这证明了该恒等式在 ${\boldsymbol{\theta}}={\boldsymbol{\theta}}^-$ 处成立。

$\blacksquare$
