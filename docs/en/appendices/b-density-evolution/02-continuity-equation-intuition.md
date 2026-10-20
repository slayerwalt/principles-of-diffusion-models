# B.2 Intuition of the Continuity Equation {#sec-intuition-continuity}

In this section, we give a physical interpretation of the continuity equation, highlighting its role as a conservation law for probability density in a dynamical system.

## B.2.1 Physical Interpretation of the Continuity Equation {#physical-interpretation-of-the-continuity-equation}

Consider a small fixed control volume (a rectangular box) in 3D space with one corner at $\mathbf{x} = (x, y, z)$ and side lengths $\Delta x$, $\Delta y$, and $\Delta z$. Let $p(\mathbf{x}, t)$ denote the density of a conserved quantity (e.g., mass or probability) at position $\mathbf{x}$ and time $t$. For a sufficiently small box, the total amount of the quantity inside the box is approximately:

$$
\text{Total quantity in box} \approx p(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

**How Does the Total Change?** Changes in the total quantity can only arise from flux across the box’s boundary. Let $\mathbf{j}(\mathbf{x}, t)$ denote the flux vector, representing the amount of quantity flowing per unit area per unit time.

**Flux in the $x$-Direction.** The inflow through the left face (at $x$) is approximately:

$$
j_x(x, y, z, t) \Delta y \Delta z,
$$

and the outflow through the right face (at $x + \Delta x$) is:

$$
j_x(x + \Delta x, y, z, t) \Delta y \Delta z.
$$

Thus, the net *influx* in the $x$-direction is:

$$
\left[ j_x(x, y, z, t) - j_x(x + \Delta x, y, z, t) \right] \Delta y \Delta z.
$$

**Net Flux in All Directions.** Analogous terms arise in the $y$- and $z$-directions:

$$
\begin{aligned}
\left[ j_y(x, y, z, t) - j_y(x, y + \Delta y, z, t) \right] \Delta x \Delta z, \\
\left[ j_z(x, y, z, t) - j_z(x, y, z + \Delta z, t) \right] \Delta x \Delta y.
\end{aligned}
$$

Summing all contributions, the total net *influx* into the box is:

$$
- \nabla \cdot \mathbf{j}(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

Equivalently, the total net *outflux* from the box is:

$$
\nabla \cdot \mathbf{j}(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

**Rate of Change Inside the Box.** The rate of change of the total quantity within the box is:

$$
\frac{\partial p}{\partial t}(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

**Conservation Principle.** Assuming the quantity is conserved (e.g., total mass or probability is constant in time), the rate of change equals the negative of the net outflux:

$$
\frac{\partial p}{\partial t}(\mathbf{x}, t) \Delta x \Delta y \Delta z = - \nabla \cdot \mathbf{j}(\mathbf{x}, t) \Delta x \Delta y \Delta z.
$$

Canceling the common volume factor and taking the small-box limit, we obtain the local form of the continuity equation:

$$
\frac{\partial p}{\partial t} + \nabla \cdot \mathbf{j} = 0.
$$

This equation has a simple interpretation: if $\nabla\cdot\mathbf{j}(\mathbf{x},t)>0$, more quantity flows out of the small box than into it, so the density inside decreases. If $\nabla\cdot\mathbf{j}(\mathbf{x},t)<0$, more quantity flows in than out, so the density inside increases.

## B.2.2 Derivation of the Continuity Equation from Conservation Laws {#derivation-of-the-continuity-equation-from-conservation-laws}

The small-box argument above gives the local intuition. We now write the same conservation principle in a coordinate-free form over an arbitrary control volume.

The continuity equation formalizes the conservation of a physical quantity, such as mass or charge, in a dynamical system. Let $p(\mathbf{x}, t)$ denote the density of the conserved quantity at position $\mathbf{x} \in \mathbb{R}^D$ and time $t \in [0, T]$, and let $\mathbf{v}(\mathbf{x}, t)$ denote the velocity field. When the quantity is transported by particles moving with velocity $\mathbf{v}(\mathbf{x},t)$, the flux is

$$
\mathbf{j}(\mathbf{x},t)=p(\mathbf{x},t)\mathbf{v}(\mathbf{x},t).
$$

**Step 1: Rate of Change within a Control Volume.** Consider an arbitrary control volume $V \subset \mathbb{R}^D$ with boundary $\partial V$. The total amount of the conserved quantity in $V$ is

$$
\int_V p(\mathbf{x}, t) \mathop{}\!\mathrm{d} V,
$$

whose time derivative gives the rate of accumulation:

$$
\frac{\partial}{\partial t} \int_V p(\mathbf{x}, t) \mathop{}\!\mathrm{d} V.
$$

**Step 2: Net Flux Through the Boundary.** The quantity exits $V$ through $\partial V$ with outward normal vector $\mathbf{n}$. The net outward flux is

$$
\int_{\partial V} p(\mathbf{x}, t) \mathbf{v}(\mathbf{x}, t) \cdot \mathbf{n} \mathop{}\!\mathrm{d} S.
$$

**Step 3: Conservation Principle.** Conservation implies that the rate of accumulation within $V$ equals the negative of the net outward flux:

$$
\frac{\partial}{\partial t} \int_V p \mathop{}\!\mathrm{d} V + \int_{\partial V} p \mathbf{v} \cdot \mathbf{n} \mathop{}\!\mathrm{d} S = 0.
$$

**Step 4: Divergence Theorem.** Applying the divergence theorem to convert the surface integral to a volume integral:

$$
\int_{\partial V} p \mathbf{v} \cdot \mathbf{n} \mathop{}\!\mathrm{d} S = \int_V \nabla \cdot (p \mathbf{v}) \mathop{}\!\mathrm{d} V.
$$

Hence,

$$
\frac{\partial}{\partial t} \int_V p \mathop{}\!\mathrm{d} V + \int_V \nabla \cdot (p \mathbf{v}) \mathop{}\!\mathrm{d} V = 0.
$$

Equivalently,

$$
\int_V \left(\frac{\partial p}{\partial t}+\nabla\cdot(p\mathbf{v})\right)\mathop{}\!\mathrm{d} V=0.
$$

Since the control volume $V$ is arbitrary, the integrand must vanish pointwise. This yields the continuity equation:

$$
\frac{\partial p}{\partial t}+\nabla\cdot(p\mathbf{v})=0.
$$

Using $\mathbf{j}=p\mathbf{v}$, this is the same as

$$
\frac{\partial p}{\partial t}+\nabla\cdot\mathbf{j}=0.
$$

**Particle-Level Intuition.** Expanding the divergence term gives

$$
\frac{\partial p}{\partial t}
+
\mathbf{v}\cdot\nabla p
+
p\,\nabla\cdot\mathbf{v}
=
0.
$$

The first two terms describe the total change of the density along a moving particle:

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}p(\mathbf{x}_t,t)
=
\frac{\partial p}{\partial t}(\mathbf{x}_t,t)
+
\mathbf{v}(\mathbf{x}_t,t)\cdot\nabla p(\mathbf{x}_t,t).
$$

Therefore,

$$
\frac{\mathop{}\!\mathrm{d}}{\mathop{}\!\mathrm{d} t}p(\mathbf{x}_t,t)
=
-
p(\mathbf{x}_t,t)\nabla\cdot\mathbf{v}(\mathbf{x}_t,t).
$$

This provides another useful interpretation: density decreases when nearby particles spread apart, and density increases when nearby particles contract. Thus, the velocity field tells us how individual particles move, while the continuity equation tells us how the entire density evolves.
