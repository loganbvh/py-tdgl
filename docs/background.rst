.. _background:

**********
Background
**********

Here we sketch out the generalized time-dependent Ginzburg-Landau model implemented in ``pyTDGL``, and the numerical methods used to solve it.
This material is largely based on the following references: [Jonsson-PRA-2022]_, [Jonsson-PHD-2022]_.

``pyTDGL`` can model superconducting thin films of arbitrary geometry, including multiply-connected films (i.e., films with holes).
By "thin" or "two-dimensional" we mean that the film thickness :math:`d` is smaller than the coherence length :math:`xi`
and the London penetration depth :math:`\lambda`. This assumption implies that both the superconducting order parameter :math:`\psi(\mathbf{r})`
and the supercurrent :math:`\mathbf{J}_s(\mathbf{r})` are roughly constant over the thickness of the film.

Time-dependent Ginzburg-Landau
------------------------------

The time-dependent Ginzburg-Landau formalism employed here boils down to a set of coupled partial differential equations for a
complex-valued field :math:`\psi(\mathbf{r}, t)=|\psi|(\mathbf{r}, t)e^{i\theta(\mathbf{r}, t)}` (the superconducting order parameter)
and a real-valued field :math:`\mu(\mathbf{r}, t)` (the electric scalar potential), which evolve deterministically in time for a given
time-independent applied magnetic vector potential :math:`\mathbf{A}(\mathbf{r})`.

The order parameter :math:`\psi` evolves according to:

.. math::
    :label: tdgl

    \frac{u}{\sqrt{1+\gamma^2|\psi|^2}}\left(\frac{\partial}{\partial t}+i\mu+\frac{\gamma^2}{2}\frac{\partial |\psi|^2}{\partial t}\right)\psi
    =(1-|\psi|^2)\psi+(\nabla-i\mathbf{A})^2\psi

The quantity :math:`(\nabla-i\mathbf{A})^2\psi` is the covariant Laplacian of :math:`\psi`,
which is used in place of an ordinary Laplacian in order to maintain gauge-invariance of the order parameter. Similarly,
the quantity :math:`(\frac{\partial}{\partial t}+i\mu)\psi` is the covariant time derivative of :math:`\psi`.
:math:`u=5.79` is the ratio of relaxation times for the amplitude and phase of the order parameter in dirty superconductors and
:math:`\gamma` is a material parameter that characterizes the influence of the inelastic phonon-electron scattering.

.. .. math::
..     :label: helmholtz

..     \kappa^2\nabla\times\nabla\times\mathbf{A} = \mathbf{J}_s-\nabla\mu-\frac{\partial\mathbf{A}}{\partial t}

The electric scalar potential :math:`\mu(\mathbf{r}, t)` evolves according to:

.. math::
    :label: poisson

    \nabla^2\mu = \nabla\cdot\mathrm{Im}[\psi^*(\nabla-i\mathbf{A})\psi],

where :math:`\mathbf{J}_s=\mathrm{Im}[\psi^*(\nabla-i\mathbf{A})\psi]` is the supercurrent density. Again, :math:`(\nabla-i\mathbf{A})\psi`
is the covariant gradient of :math:`\psi`.

Boundary conditions
===================

Isolating boundary conditions are enforced on superconductor-vacuum interfaces:

.. math::
    :label: bc-vacuum

    \begin{split}
        \mathbf{n}\cdot(\nabla-i\mathbf{A})\psi &= 0 \\
        \mathbf{n}\cdot\nabla\mu &= 0
    \end{split}

Superconductor-normal metal interfaces can be used to apply a bias current density :math:`J_\mathrm{ext}`. For such interfaces, the boundary conditions are:

.. math::
    :label: bc-normal

    \begin{split}
        \psi &= 0 \\
        \mathbf{n}\cdot\nabla\mu &= J_\mathrm{ext}
    \end{split}

One may also model normal-metal inclusions by fixing :math:`\psi(\mathbf{r})=0` for some set of points :math:`\mathbf{r}` inside the film. This can be used to simulate pinning centers. 

Units
=====

The TDGL equation [:eq:`tdgl`, :eq:`poisson`] is solved in dimensionless units, where the scale factors are given in terms of fundamental constants and material parameters,
namely the superconducting coherence length :math:`\xi`, London penetration depth :math:`\lambda`, normal state conductivity :math:`\sigma`, and film thickness :math:`d`.
The Ginzburg-Landau parameter is defined as :math:`\kappa=\lambda/\xi`.

Time is measured in units of :math:`\tau_0`:

.. math::
    :label: tau0

    \tau_0 = \mu_0\sigma\lambda^2

Magnetic field is measured in units of the upper critical field :math:`B_0=B_{c2}`:

.. math::
    :label: B0

    B_0 = B_{c2} = \mu_0H_{c2} = \frac{\Phi_0}{2\pi\xi^2}

Magnetic vector potential is measured in units of :math:`A_0=\xi B_0`:

.. math::
    :label: A0

    A_0 = \xi B_0 = \frac{\Phi_0}{2\pi\xi}

Current density is measured in units of :math:`J_0`:

.. math::
    :label: J0

    J_0 = \frac{4\xi B_{c2}}{\mu_0\lambda^2}

Sheet current density is measured in units of :math:`K_0=J_0 d`:

.. math::
    :label: K0

    K_0 = J_0 d = \frac{4\xi B_{c2}}{\mu_0\Lambda},

where :math:`\Lambda=\lambda^2/d` is the effective magnetic penetration depth.

Voltage is measured in terms of :math:`V_0=\xi J_0/\sigma`:

.. math::
    :label: V0

    V_0 = \frac{\xi J_0}{\sigma} = \frac{4\xi^2 B_{c2}}{\mu_0\sigma\lambda^2}

Finite volume method
--------------------

We solve the TDGL [:eq:`tdgl`, :eq:`poisson`] on an unstructured Delaunay mesh in two dimenions.
The mesh is composed of a set of sites :math:`\mathbf{r}_i`
and a set of triangular cells :math:`c_{ijk}`. Each cell :math:`c_{ijk}=(i, j, k)` represents a triangle with three edges
(:math:`(i, j)`, :math:`(j, k)`, and :math:`(k, i)`) that connect sites :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j`, :math:`\mathbf{r}_k` in
a counterclockwise fashion. Each edge has a length :math:`e_{ij}=|\mathbf{r}_j-\mathbf{r}_i|` and a direction :math:`\hat{e}_{ij}=(\mathbf{r}_j-\mathbf{r}_i)/e_{ij}`.
Each site is assigned an effective area :math:`a_i`, which is the area of the `Voronoi region <https://en.wikipedia.org/wiki/Voronoi_diagram>`_
surrounding the site.
The Voronoi region surrounding site :math:`\mathbf{r}_i` consists of all points in space that are closer to site :math:`\mathbf{r}_i`
than to any other site in the mesh. The side of the Voronoi region that intersects edge :math:`(i, j)` is denoted
:math:`\mathbf{s}_{ij}` and has a length :math:`s_{ij}`.

.. image:: images/voronoi.png
  :width: 400
  :alt: Schematic of a mesh.
  :align: center

A scalar function :math:`f(\mathbf{r}, t)` can be discretized at a given time :math:`t_n`
as the value of the function on each site, :math:`f_i^n=f(\mathbf{r}_i, t_n)`. A vector function :math:`\mathbf{F}(\mathbf{r}, t)`
can be discretized at time :math:`t_n` as the flow of the vector field between sites.
In other words, :math:`F_{ij}^n=\mathbf{F}((\mathbf{r}_i+\mathbf{r}_j)/2, t_n)\cdot\hat{e}_{ij}`, where :math:`(\mathbf{r}_i+\mathbf{r}_j)/2=r_{ij}`
is the center of edge :math:`(i, j)`.

To calculate the divergence of a vector field :math:`\mathbf{F}(\mathbf{r})` on the mesh, we assume that
each Voronoi cell is small enough that the value of :math:`\nabla\cdot\mathbf{F}` is constant over the area of the cell and
equal to the value at the cell center, :math:`\mathbf{r}_i`.
Then, using the `divergence theorem <https://en.wikipedia.org/wiki/Divergence_theorem>`_ in two dimensions, we have

.. math::
    :label: divergence

    \begin{split}
        \int(\nabla\cdot\mathbf{F})\,\mathrm{d}^2\mathbf{r} &= \oint(\mathbf{F}\cdot\hat{n})\,\mathrm{d}s\\
        (\nabla\cdot\mathbf{F})_{\mathbf{r}_i}a_i&\approx\sum_{j\in\mathcal{N}(i)}F_{ij}s_{ij}\\
        (\nabla\cdot\mathbf{F})_{\mathbf{r}_i}&\approx\frac{1}{a_i}\sum_{j\in\mathcal{N}(i)}F_{ij}s_{ij},
    \end{split}

where :math:`\mathcal{N}(i)` is the set of sites adjacent to site :math:`\mathbf{r}_i`.
The gradient of a scalar function :math:`g(\mathbf{r})` is approximated on the edges of the mesh. The value of :math:`\nabla g`
at position :math:`\mathbf{r}_{ij}` (i.e., the center of edge :math:`(i, j)`) is:

.. math::
    :label: gradient

    (\nabla g)_{\mathbf{r}_{ij}}\approx\frac{g_j-g_i}{e_{ij}}

The Laplacian of a scalar function :math:`g` is given by :math:`\nabla^2 g=\nabla\cdot\nabla g`, so combining :eq:`divergence` and :eq:`gradient` we have

.. math::
    :label: laplacian

    (\nabla^2g)_{\mathbf{r}_i}\approx\frac{1}{a_i}\sum_{j\in\mathcal{N}(i)}\frac{g_j-g_i}{e_{ij}}s_{ij}

Link variables
==============

.. math::
    :label: link-sym

    U_{ij}(t) = U(\mathbf{r}_i,\mathbf{r}_j, t) = \exp\left(-i\int_{\mathbf{r}_i}^{\mathbf{r}_j}\mathbf{A}(\mathbf{r}, t)\cdot\mathrm{d}\mathbf{r}\right)

Implicit Euler method
=====================

Screening
---------

If :math:`\Lambda=\lambda^2/d\gg L`, then one can neglect screening and assume that the total vector potential in the film is
time-independent and equal to the applied vector potential: :math:`\mathbf{A}(\mathbf{r}, t)=\mathbf{A}_\mathrm{applied}(\mathbf{r})`.
If :math:`\Lambda\approx L`, then one must take screening into account because the total vector potential in the film will be
:math:`\mathbf{A}(\mathbf{r}, t)=\mathbf{A}_\mathrm{applied}(\mathbf{r})+\mathbf{A}_\mathrm{induced}(\mathbf{r}, t)`.
We assume that the magnetic vector potential is either constant as a function of time
or varies slowly enough that its time derivative can be neglected when calculating the electric field:
:math:`\mathbf{E}=-\nabla\mu-\frac{\partial\mathbf{A}}{\partial t}\approx-\nabla\mu`.

If the applied vector potential is due to a local field source, such as a small dipole or small current loop, then one can identify
a length :math:`\rho_0`, which is the radial distance away from the field source at which the sign of the field changes sign.

.. math::
    :label: A_induced

    \mathbf{A}_\mathrm{induced}(\mathbf{r}, t) = \frac{\mu_0}{4\pi}\int_\mathrm{film}\frac{\mathbf{K}(\mathbf{r}', t)}{|\mathbf{r}-\mathbf{r}'|}\,\mathrm{d}^2\mathbf{r}',

where :math:`\mathbf{K}=\mathbf{K}_s+\mathbf{K}_n=d\mathbf{J}=d(\mathbf{J}_s+\mathbf{J}_n)` is the total sheet current density.  

Adaptive time step
------------------

``pyTDGL`` implements an adaptive time step algorithm that optionally adjusts the time step :math:`\Delta t`
based on the speed of the system's dynamics. This functionality is useful if, for example, you are only interested
in the equilibrium behavior of a system. The dynamics may initially be quite fast and then slow down as you approach steady state.
Using an adaptive time step dramatically reduces the wall-clock time needed to model equilibrium behavior in such instances, without
sacrificing solution accuracy. 