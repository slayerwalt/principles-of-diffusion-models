# B.2 连续性方程的直观理解 {#sec-intuition-continuity}

本节给出连续性方程的物理解释，强调它作为动力系统中概率密度守恒律的作用。

## B.2.1 连续性方程的物理解释 {#physical-interpretation-of-the-continuity-equation}

考虑三维空间中一个固定的小控制体（长方体），其一个顶点为 $\mathbf{x} = (x, y, z)$，边长分别为 $\Delta x$、$\Delta y$ 和 $\Delta z$。令 $p(\mathbf{x}, t)$ 表示守恒量（例如质量或概率）在位置 $\mathbf{x}$ 和时刻 $t$ 的密度。对于足够小的长方体，内部该物理量的总量近似为：

$$
\text{Total quantity in box} \approx p(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

**总量如何变化？** 总量的变化只能来自穿过长方体边界的通量。令 $\mathbf{j}(\mathbf{x}, t)$ 表示通量向量，即每单位面积、每单位时间流过的量。

**$x$ 方向的通量。** 通过左侧面（位于 $x$）的流入量近似为：

$$
j_x(x, y, z, t) \Delta y \Delta z,
$$

通过右侧面（位于 $x + \Delta x$）的流出量为：

$$
j_x(x + \Delta x, y, z, t) \Delta y \Delta z.
$$

因此，$x$ 方向的净*流入*为：

$$
\left[ j_x(x, y, z, t) - j_x(x + \Delta x, y, z, t) \right] \Delta y \Delta z.
$$

**所有方向的净通量。** 在 $y$ 和 $z$ 方向也有类似项：

$$
\begin{aligned}
\left[ j_y(x, y, z, t) - j_y(x, y + \Delta y, z, t) \right] \Delta x \Delta z, \\
\left[ j_z(x, y, z, t) - j_z(x, y, z + \Delta z, t) \right] \Delta x \Delta y.
\end{aligned}
$$

汇总各项，进入长方体的总净*流入*为：

$$
- \nabla \cdot \mathbf{j}(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

等价地，离开长方体的总净*流出*为：

$$
\nabla \cdot \mathbf{j}(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

**长方体内的变化率。** 长方体内部总量的变化率为：

$$
\frac{\partial p}{\partial t}(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

**守恒原则。** 假设该量守恒（例如总质量或总概率不随时间改变），则变化率等于净流出的负值：

$$
\frac{\partial p}{\partial t}(\mathbf{x}, t) \Delta x \Delta y \Delta z = - \nabla \cdot \mathbf{j}(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

消去共同的体积因子，并取小控制体极限，得到连续性方程的局部形式：

$$
\frac{\partial p}{\partial t} + \nabla \cdot \mathbf{j} = 0.
$$

这一方程有简单的解释：若 $\nabla\cdot\mathbf{j}(\mathbf{x},t)>0$，流出小长方体的量多于流入量，内部密度下降；若 $\nabla\cdot\mathbf{j}(\mathbf{x},t)<0$，流入量多于流出量，内部密度上升。

## B.2.2 从守恒律推导连续性方程 {#derivation-of-the-continuity-equation-from-conservation-laws}

上述小长方体论证提供了局部直观认识。现在对任意控制体，用不依赖坐标的形式表达同一守恒原则。

连续性方程形式化地描述动力系统中质量、电荷等物理量的守恒。令 $p(\mathbf{x}, t)$ 表示守恒量在位置 $\mathbf{x} \in \mathbb{R}^D$ 和时刻 $t \in [0, T]$ 的密度，$\mathbf{v}(\mathbf{x}, t)$ 表示速度场。当该量由速度为 $\mathbf{v}(\mathbf{x},t)$ 的运动粒子输运时，通量为

$$
\mathbf{j}(\mathbf{x},t)=p(\mathbf{x},t)\mathbf{v}(\mathbf{x},t).
$$

**步骤 1：控制体内的变化率。** 考虑任意控制体 $V \subset \mathbb{R}^D$，边界为 $\partial V$。$V$ 中守恒量的总量为

$$
\int_V p(\mathbf{x}, t) \mathop{}\!\mathrm{d} V,
$$

其时间导数给出累积速率：

$$
\frac{\partial}{\partial t} \int_V p(\mathbf{x}, t) \mathop{}\!\mathrm{d} V.
$$

**步骤 2：穿过边界的净通量。** 该量离开 $V$，穿过边界 $\partial V$，外法向量为 $\mathbf{n}$。净向外通量为

$$
\int_{\partial V} p(\mathbf{x}, t) \mathbf{v}(\mathbf{x}, t) \cdot \mathbf{n} \mathop{}\!\mathrm{d} S.
$$

**步骤 3：守恒原则。** 守恒意味着，$V$ 内的累积速率等于净向外通量的负值：

$$
\frac{\partial}{\partial t} \int_V p \mathop{}\!\mathrm{d} V + \int_{\partial V} p \mathbf{v} \cdot \mathbf{n} \mathop{}\!\mathrm{d} S = 0.
$$

**步骤 4：散度定理。** 应用散度定理，将曲面积分转化为体积分：

$$
\int_{\partial V} p \mathbf{v} \cdot \mathbf{n} \mathop{}\!\mathrm{d} S = \int_V \nabla \cdot (p \mathbf{v}) \mathop{}\!\mathrm{d} V.
$$

因此，

$$
\frac{\partial}{\partial t} \int_V p \mathop{}\!\mathrm{d} V + \int_V \nabla \cdot (p \mathbf{v}) \mathop{}\!\mathrm{d} V = 0.
$$

等价地，

$$
\int_V \left(\frac{\partial p}{\partial t}+\nabla\cdot(p\mathbf{v})\right)\mathop{}\!\mathrm{d} V=0.
$$

由于控制体 $V$ 任意，被积函数必须逐点为零。由此得到连续性方程：

$$
\frac{\partial p}{\partial t}+\nabla\cdot(p\mathbf{v})=0.
$$

利用 $\mathbf{j}=p\mathbf{v}$，这等同于

$$
\frac{\partial p}{\partial t}+\nabla\cdot\mathbf{j}=0.
$$

**粒子层面的直观理解。** 展开散度项，得到

$$
\frac{\partial p}{\partial t}
+
\mathbf{v}\cdot\nabla p
+
p\,\nabla\cdot\mathbf{v}
=
0.
$$

前两项描述沿运动粒子观察到的密度总变化：

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}p(\mathbf{x}_t,t)
=
\frac{\partial p}{\partial t}(\mathbf{x}_t,t)
+
\mathbf{v}(\mathbf{x}_t,t)\cdot\nabla p(\mathbf{x}_t,t).
$$

因此，

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}p(\mathbf{x}_t,t)
=
-
p(\mathbf{x}_t,t)\nabla\cdot\mathbf{v}(\mathbf{x}_t,t).
$$

这提供了另一种有用解释：相邻粒子分散时，密度下降；相邻粒子收缩时，密度上升。因此，速度场描述单个粒子如何运动，连续性方程描述整个密度如何演化。
