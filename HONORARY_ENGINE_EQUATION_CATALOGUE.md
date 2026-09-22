# Honorary engines: governing-equation and coupling catalogue

**Expanded target specification — 22 September 2026**

This document expands the fourteen-engine architecture discussed by Albert: Newton, Timoshenko, Faraday, Navier–Stokes, Bjerknes, Gibbs, Bragg, Hamilton, Curie, Einstein, Lavoisier, Fourier, Boltzmann, and Noether. It is a mathematical reference for future equation bundles, not an inventory of implemented repository functions, a claim of complete experimental validation, or a ready-to-compile program.

A complete *selected model* consists of its equations, constitutive data, domain of validity, initial and boundary conditions, interface bindings, spatial representation, and numerical method. Alternative closures below are alternatives, not terms to enable simultaneously. The catalogue covers the model families required by the proposed chamber/multiphysics package; additional application-specific models must retain the same contracts.

## Contents

0. Conventions and equation-bundle contract
1. Newton — particles, balls, springs, rigid bodies, molecular mechanics
2. Timoshenko — beams, frames, structural reduction
3. Faraday — electromagnetic fields, circuits, waves
4. Navier–Stokes — continuum fluid transport and passages
5. Bjerknes — atmosphere, droplets, aerosols, surface phases
6. Gibbs — thermodynamic potentials, mixtures, phase equilibrium
7. Bragg — lattices, microstructure, constitutive solids, semiconductors
8. Hamilton — quantum states, electronic structure, molecular geometry
9. Curie — nuclides, decay, radiation emission and deposition
10. Einstein — relativistic kinematics and optional spacetime geometry
11. Lavoisier — chemical identity, reactions, electrochemistry
12. Fourier — heat transport, storage, radiation and thermal networks
13. Boltzmann — kinetic distributions and statistical mechanics
14. Noether — symmetry, conservation and cross-engine audits
15. Coupling, population weights and model promotion
16. Numerical realization and verification gates
17. References

---

## 0. Conventions and equation-bundle contract

### 0.1 Units, signs and state

Use SI unless a block explicitly declares atomic units. Temperature is absolute kelvin except in a fitted formula explicitly declaring Celsius. $R_u$ is the molar gas constant; $k_B$ is the per-particle Boltzmann constant; $M_s$ is molar mass in kg/mol. Thus $R_s=R_u/M_s$ and $n_s^{\rm number}=N_A c_s$ for concentration $c_s$ in mol/m³. Species amount $n_s$ in mol is distinct from particle number $N_s$ and number density $n_s^{\rm number}$.

$\rho$ is mass density, $\mathbf u$ continuum velocity, $\mathbf v$ particle velocity, $\mathbf n$ an explicitly oriented unit surface normal. $\boldsymbol\sigma$ is mechanical stress, $\sigma_e$ electrical conductivity, $\sigma_{\rm SB}$ the Stefan–Boltzmann constant, and $\gamma_s$ surface energy. Symbols have namespaces and units; identical spelling is not sufficient to identify physical state.

Continuum material derivative: $D_t=\partial_t+\mathbf u\cdot\nabla$. Specific $e,h,s$ are per mass unless marked molar; extensive $U,H,S,G,A$ are whole-system quantities. $A$ denotes Helmholtz free energy only in thermodynamics; an area or incidence matrix must have a different typed identity.

Electromagnetic phasors below use $\exp(-i\omega t)$ and **peak**, not RMS, amplitudes. Positive imaginary permittivity then represents passive loss. A quantum density matrix is $\hat\varrho$, not fluid density. Relativistic coordinates are $x^0=ct$ and metric signature is $(-,+,+,+)$.

### 0.2 What every law must carry

Recommended record:

```text
law_id, version, source_reference
owned_state, read_state, physical_identity_bindings
continuous_equations or algebraic_residuals
constitutive_model_id, coefficient_data_id, valid_domain
boundary_and_initial_conditions
units, frame, orientation, time_level_and_stage
exchange_ports, energy_reference, population_weight_semantics
invariants, independent_witnesses, numerical_error_contract
```

The general continuous form is

$$\mathcal R(y,\dot y,z,t;\theta)=0,\qquad g(y,z,t;\theta)=0,$$

where $y$ is differential state, $z$ algebraic state, and $\theta$ declared parameters. A common semidiscrete case is $M(y)\dot y=f(y,z,t)$, but some engines instead expose eigenproblems, constrained minima, stochastic generators, or constitutive residuals. Do not misrepresent all of them as a simple explicit ODE.

**Ownership rule:** an engine may supply a force, flux, derivative, residual, or closure without independently advancing a state already advanced by another engine. `dt_system` supplies the admissible numerical realization. The following are equations to implement, not prescriptions to replace that runtime.

---

## 1. Newton — classical motion and force providers

**State:** positions, momenta, orientations, angular momenta, contact/constraint state. Hooke and the spring/ball lineage live here as law providers, not a competing clock. Molecular force fields are selected providers, not universal chemistry. References: [R1–R3].

### N1. Translation and rigid-body orientation

For constant particle masses,

$$\dot{\mathbf x}_i=\frac{\mathbf p_i}{m_i},\qquad \dot{\mathbf p}_i=\mathbf F_i.$$

For rotation matrix $R$ mapping body to world coordinates,

$$\dot R=R[\boldsymbol\omega_B]_\times,\quad R^TR=I,\quad \det R=1,$$

$$I_B\dot{\boldsymbol\omega}_B+\boldsymbol\omega_B\times(I_B\boldsymbol\omega_B)=\boldsymbol\tau_B.$$

$[\omega]_\times a=\omega\times a$. The gyroscopic term must not be replaced by $\tau=I\alpha$ for arbitrary free rotation.

### N2. Energy and generalized mechanics

$$K=\sum_i\frac{\mathbf p_i^2}{2m_i}+\frac12\boldsymbol\omega_B^TI_B\boldsymbol\omega_B,\qquad \mathbf F_i^{\rm cons}=-\nabla_i U.$$

$$\frac{dK}{dt}=\sum_i\mathbf F_i\cdot\mathbf v_i+\boldsymbol\tau_B\cdot\boldsymbol\omega_B.$$

Alternative generalized-coordinate authoring:

$$\frac{d}{dt}\frac{\partial L}{\partial\dot q_a}-\frac{\partial L}{\partial q_a}=Q_a^{\rm nc},\qquad L=K-U.$$

For a regular Hamiltonian description: $\dot q=\partial H/\partial p$, $\dot p=-\partial H/\partial q+Q^{\rm nc}$. These are alternative representations of the same mechanics.

### N3. Springs, damping and graph assembly

Let $\mathbf r=\mathbf x_j-\mathbf x_i$, $r=|\mathbf r|$, $\mathbf n=\mathbf r/r$, and $v_\parallel=(\mathbf v_j-\mathbf v_i)\cdot\mathbf n$. Then

$$U_{ij}=\frac12 k_{ij}(r-\ell_{0,ij})^2,$$

$$\mathbf F_i=[k_{ij}(r-\ell_{0,ij})+c_{ij}v_\parallel]\mathbf n,\qquad \mathbf F_j=-\mathbf F_i,$$

$$P_{\rm dissipated}=c_{ij}v_\parallel^2\ge0.$$

If $B$ is edge-by-node oriented incidence, $\mathbf r=BX$ and nodal force is $-B^T\partial U/\partial\mathbf r$. Since $B\mathbf1=0$, internal edge forces sum to zero. Coincident endpoints require an explicit admissible limit/contact rule; a numerical epsilon is a changed model unless its role is declared.

### N4. Gravity, buoyancy and fluid drag

$$\mathbf F_{i\leftarrow j}=Gm_im_j\frac{\mathbf x_j-\mathbf x_i}{|\mathbf x_j-\mathbf x_i|^3},\qquad \nabla^2\Phi=4\pi G\rho,\qquad \mathbf F=-m\nabla\Phi.$$

Choose either direct pair gravity or the matching field representation, not both. Hydrostatic buoyancy is $\mathbf F_b=-\rho_fV\mathbf g$. A drag closure is

$$\mathbf F_d=-\frac12\rho_f C_D A_p|\mathbf v-\mathbf u|(\mathbf v-\mathbf u).$$

The equal and opposite force goes to the resolved fluid or to a declared external bath. $C_D$ is a regime/shape-dependent constitutive input.

### N5. Unilateral contact and friction

$$g_n\ge0,\quad \lambda_n\ge0,\quad g_n\lambda_n=0,\qquad |\boldsymbol\lambda_t|\le\mu_f\lambda_n.$$

During sliding, $\boldsymbol\lambda_t=-\mu_f\lambda_n\mathbf v_t/|\mathbf v_t|$. For an isolated normal impact with incoming relative normal velocity $v_n^-<0$,

$$v_n^+=-e_rv_n^-,\qquad 0\le e_r\le1,$$

$$J_n=-\frac{(1+e_r)v_n^-}{m_a^{-1}+m_b^{-1}+(\mathbf r_a\times\mathbf n)^TI_a^{-1}(\mathbf r_a\times\mathbf n)+(\mathbf r_b\times\mathbf n)^TI_b^{-1}(\mathbf r_b\times\mathbf n)}.$$

Both inertia tensors, contact offsets and the normal are expressed in the same world frame. This scalar formula is not a simultaneous many-contact solver. Coupled contacts require the corresponding constrained system.

An optional elastic sphere-contact closure is Hertz:

$$F_n=\frac43E^*\sqrt{R^*}\,\delta^{3/2},\quad \frac1{E^*}=\frac{1-\nu_a^2}{E_a}+\frac{1-\nu_b^2}{E_b},\quad \frac1{R^*}=\frac1{R_a}+\frac1{R_b}.$$

Use only under its small-contact, elastic assumptions; do not apply it on top of another contact law for the same normal reaction.

### N6. Molecular mechanics and geometry

A representative fixed-topology force field is

$$\begin{aligned}
U(R)={}&\sum_b\frac{k_b}{2}(r_b-r_b^0)^2+\sum_a\frac{k_a}{2}(\theta_a-\theta_a^0)^2\\
&+\sum_d k_d[1+\cos(n_d\phi_d-\delta_d)]\\
&+\sum_{i<j}^{\rm selected\ nonbonded}\left\{4\epsilon_{ij}\left[\left(\frac{\sigma_{ij}}{r_{ij}}\right)^{12}-\left(\frac{\sigma_{ij}}{r_{ij}}\right)^6\right]
+\frac{q_iq_j}{4\pi\epsilon r_{ij}}\right\}.
\end{aligned}$$

Bond exclusions, 1–4 scaling, combining rules, periodic electrostatics and cutoffs are part of the model. Forces are $-\nabla_RU$. Bond-angle and torsion terms are essential to molecular geometry; distance-only springs do not specify it.

For a breakable *single-bond approximation*, a Morse provider can replace—not supplement the same harmonic bond:

$$U_M(r)=D_e[1-e^{-a(r-r_e)}]^2.$$

A topology-changing reactive force field or quantum energy surface is required before this becomes general bond chemistry. A molecular mechanics Coulomb term must not duplicate an explicitly resolved Faraday interaction.

### N7. Constraints and varying mass

Holonomic constraints $g(q,t)=0$ add $G^T\lambda$ to generalized forces, with $G=\partial g/\partial q$. Their work and driven-boundary energy must be accounted for. Mass exchange requires momentum of entering/leaving material:

$$\frac{d(m\mathbf v)}{dt}=\mathbf F_{\rm ext}+\sum_{\rm in}\dot m\mathbf v_{\rm in}-\sum_{\rm out}\dot m\mathbf v_{\rm out}.$$

This is a lumped open-body balance, not permission to apply $m\dot v=F$ while silently changing $m$.

**Witnesses:** isolated pair action–reaction; free rotation; harmonic oscillator; elastic sphere collision; molecular energy-gradient finite differences; momentum and mechanical-plus-dissipated-energy balance.

---

## 2. Timoshenko — structural mechanics and reduction

**State:** structural generalized coordinates and velocities; section strains/resultants; modal coordinates when reduced. Bragg supplies constitutive state. References: [R4, R5].

### T1. Beam kinematics and resultants

For one bending plane,

$$\gamma=w_x-\theta,\quad \kappa_b=\theta_x,\quad Q=\kappa_sGA\gamma,\quad M=EI\kappa_b.$$

$w$ is transverse displacement, $\theta$ section rotation, and $\kappa_s$ the shear correction factor. Here $I$ is area moment, not a rigid-body inertia tensor.

### T2. Timoshenko dynamics

$$\rho A w_{tt}=\partial_x Q+q,\qquad \rho I\theta_{tt}=\partial_x M+Q+c.$$

$q$ is transverse force per length and $c$ distributed couple per length. Repeat in the second bending plane using the corresponding section properties and orientation.

### T3. Axial and torsional dynamics

$$\rho A u_{tt}=\partial_x(EA u_x)+f_x,$$

$$\rho I_p\varphi_{tt}=\partial_x(GJ\varphi_x)+m_x.$$

The polar area moment $I_p$ and Saint-Venant torsion constant $J$ are generally different for noncircular sections.

### T4. Energy and finite-element assembly

$$U_b=\frac12\int[EAu_x^2+EI\theta_x^2+\kappa_sGA(w_x-\theta)^2+GJ\varphi_x^2]dx.$$

With declared interpolation $N$ and strain operator $B_s$,

$$K_e=\int B_s^TC_sB_s\,dx,\qquad M_e=\int N^T\rho_sN\,dx.$$

The assembly obeys

$$M\ddot q+C\dot q+f_{\rm int}(q,z)=f_{\rm ext},$$

where $z$ contains material history. Interpolation, shear locking treatment, joint releases and boundary conditions are required parts of an element definition; the matrix-integral formula alone does not select an element.

### T5. Linear modes and damping

$$K\phi_k=\omega_k^2M\phi_k,\qquad \phi_i^TM\phi_j=\delta_{ij},$$

$$\ddot a_k+2\zeta_k\omega_k\dot a_k+\omega_k^2a_k=\phi_k^Tf.$$

Optional Rayleigh damping: $C=\alpha_M M+\beta_KK$. These fitted damping coefficients are not universal material constants. Modal truncation requires an explicit omitted-mode/error policy.

### T6. Prestress, buckling and thermal strain

$$K_{\rm tangent}=K_{\rm material}+K_{\rm geometric},\qquad P_{\rm cr}=\frac{\pi^2EI}{(K_LL)^2}.$$

The Euler buckling expression is for an ideal prismatic slender column, with effective-length factor $K_L$ set by its supports. It is not a universal postbuckling law.

$$\epsilon^{\rm th}=\int_{T_0}^{T}\alpha_T(T',z)\,dT',\qquad \sigma=E(z,T)(\epsilon-\epsilon^{\rm th}-\epsilon^p).$$

### T7. Substructure reduction

For static internal coordinates $i$ and exposed coordinates $b$,

$$q_i=K_{ii}^{-1}(f_i-K_{ib}q_b),\qquad K_{\rm red}=K_{bb}-K_{bi}K_{ii}^{-1}K_{ib}.$$

Dynamic reduction retains internal modes or frequency-dependent impedance. A static Schur complement does not preserve arbitrary transient dynamics. A thin-plate lane, if selected, uses

$$\rho h w_{tt}+D_b\nabla^4w=q,\qquad D_b=\frac{Eh^3}{12(1-\nu^2)},$$

with its own thin-plate assumptions; this is not the same model as the Timoshenko beam.

**Witnesses:** rigid-body nullspace, patch test, cantilever deflection, shear-flexible limit, known free modes, reciprocity, energy/work balance, reduced/full interface response over the declared band.

---

## 3. Faraday — electromagnetic fields, networks and waves

**State:** electromagnetic fields or potentials, dispersive material states, circuit storage and selected modal amplitudes. Shared polarization/magnetization states have one owner. References: [R6, R7].

### F1. Maxwell equations and charge continuity

$$\partial_t\mathbf B=-\nabla\times\mathbf E,\qquad \partial_t\mathbf D=\nabla\times\mathbf H-\mathbf J_f,$$

$$\nabla\cdot\mathbf D=\rho_f,\qquad \nabla\cdot\mathbf B=0,\qquad \partial_t\rho_f+\nabla\cdot\mathbf J_f=0.$$

$$\mathbf D=\epsilon_0\mathbf E+\mathbf P,\qquad \mathbf B=\mu_0(\mathbf H+\mathbf M).$$

Simple nondispersive closure: $D=\epsilon E$, $B=\mu H$, $J_c=\sigma_e E$. Tensor coefficients require the appropriate symmetry/passivity conditions.

### F2. Topology-native DEC form

Let $d_0$ map vertices to edges, $d_1$ edges to faces, and $d_2$ faces to cells. Then

$$d_1d_0=0,\qquad d_2d_1=0,$$

$$\dot b=-d_1e,\qquad M_\epsilon\dot e=d_1^TM_{\mu^{-1}}b-M_\sigma e-j.$$

It is **$d_2d_1=0$** that preserves $d_2b$ under the magnetic update. $d_1d_0=0$ instead states curl-of-gradient is zero. Nontrivial topology can also support harmonic fields; do not assume $\ker d_1=\operatorname{im}d_0$ on every domain.

### F3. Energy, power and force

For fixed nondispersive linear material,

$$u_{\rm EM}=\frac12(E\cdot D+B\cdot H),\quad \mathbf S=\mathbf E\times\mathbf H,$$

$$\partial_tu_{\rm EM}+\nabla\cdot\mathbf S=-\mathbf J_f\cdot\mathbf E.$$

Dispersive, nonlinear or evolving material requires its own stored energy and work terms. The DEC quadratic counterpart is $\tfrac12(e^TM_\epsilon e+b^TM_{\mu^{-1}}b)$ only under the same fixed-coefficient assumptions.

$$\mathbf F=q(\mathbf E+\mathbf v\times\mathbf B),\qquad \mathbf f=\rho_q\mathbf E+\mathbf J\times\mathbf B.$$

In vacuum,

$$\mathsf T=\epsilon_0(EE-\tfrac12E^2I)+\mu_0^{-1}(BB-\tfrac12B^2I),$$

$$\mathbf F_{\rm matter}=\oint\mathsf T\mathbf n\,dA-\frac{d}{dt}\int\epsilon_0\mathbf E\times\mathbf B\,dV.$$

A material medium needs a consistent total field-plus-matter stress formulation, not automatic use of the vacuum formula inside it.

### F4. Electrostatics, magnetic potential and interface conditions

$$\nabla\cdot(\epsilon\nabla\phi)=-\rho_f,\qquad \mathbf E=-\nabla\phi.$$

For a vector potential, $\mathbf B=\nabla\times\mathbf A$ and $\mathbf E=-\nabla\phi-\partial_t\mathbf A$, with an explicitly selected gauge.

For a stationary ordinary interface, $[a]=a_2-a_1$ and $\mathbf n$ points from side 1 to side 2:

$$\mathbf n\times[E]=0,\quad \mathbf n\cdot[B]=0,\quad \mathbf n\cdot[D]=\sigma_f^{\rm surface},\quad \mathbf n\times[H]=\mathbf K_f.$$

PEC: tangential $E=0$. Moving interfaces require the appropriate frame-dependent conditions.

### F5. Dispersive material closures

A Lorentz oscillator satisfies

$$\ddot P+\gamma\dot P+\omega_0^2P=\epsilon_0\omega_p^2E.$$

Drude response is the $\omega_0=0$ lane. A Debye relaxation lane is

$$\tau_D\dot P+P=\epsilon_0\Delta\chi E.$$

With peak phasors,

$$\langle q_J\rangle=\tfrac12\sigma_e|E|^2,\qquad \langle q_d\rangle=\tfrac12\omega\epsilon''|E|^2.$$

If $\epsilon''$ already includes conductive loss, adding $q_J$ again double-counts it. Good-conductor skin depth and surface resistance are

$$\delta=\sqrt{\frac{2}{\omega\mu\sigma_e}},\qquad R_s=\frac1{\sigma_e\delta}.$$

### F6. Circuits and transmission lines

For node-by-edge incidence $A_c$, branch currents $i$ and node potentials $\phi$:

$$A_ci=s,\qquad v=A_c^T\phi.$$

For constant linear elements,

$$v_R=Ri_R,\qquad i_C=C\dot v_C,\qquad v_L=L\dot i_L,$$

$$U_C=\tfrac12Cv_C^2,\qquad U_L=\tfrac12Li_L^2.$$

Variable parameters require charge/flux linkage laws $i=\dot Q$, $v=\dot\lambda$, including actuator/material work. Coupled inductors require one symmetric passive inductance matrix, not independently inverted pairwise couplings.

$$\partial_zV=-L'\partial_tI-R'I,\qquad \partial_zI=-C'\partial_tV-G'V.$$

For a load impedance, $\Gamma=(Z_L-Z_0)/(Z_L+Z_0)$. In harmonic linear analysis, port reduction is

$$Y_{\rm port}(\omega)=Y_{pp}-Y_{pi}Y_{ii}^{-1}Y_{ip}.$$

This is exact for the selected linear system where the inverse exists. Its frequency dependence and initial internal energy must be retained or approximated explicitly when realizing it in time.

### F7. Modes, waveguides and quantum/classical radiation separation

$$\nabla\times\mu^{-1}\nabla\times E_m=\omega_m^2\epsilon E_m.$$

A uniform guide has $\beta^2=\omega^2\mu\epsilon-k_c^2$. Below cutoff $\beta$ becomes imaginary: exponential field decay need not be dissipative heating. Rectangular PEC cavity frequencies are

$$f_{mnp}=\frac{c_{\rm medium}}2\sqrt{(m/a)^2+(n/b)^2+(p/d)^2},$$

using only the indices admitted by the selected TE/TM field family. Mode shape, normalization and port overlap are needed in addition to frequency.

Incoherent radiance transport under Fourier and discrete radiative particles under Curie/Boltzmann are alternative/coarse-grained descriptions of radiation. Exchanges between them must debit the represented energy once.

**Witnesses:** charge continuity; both DEC chain identities; plane-wave dispersion and energy; PEC cavity frequencies; lossy ringdown; circuit/field power equality; passive port reduction.

---

## 4. Navier–Stokes — continuum fluid transport and passages

**State:** conserved density, momentum, total energy and species masses in the resolved region. Gibbs supplies thermodynamic closure; transport coefficients need separate data or kinetic/material closures. References: [R5, R8, R9].

### NS1. Conserved balances

$$\partial_t\rho+\nabla\cdot(\rho\mathbf u)=S_m,$$

$$\partial_t(\rho\mathbf u)+\nabla\cdot(\rho\mathbf u\otimes\mathbf u+pI)=\nabla\cdot\tau+\rho\mathbf g+\mathbf f_{\rm other}+\mathbf S_p,$$

$$\partial_t(\rho E)+\nabla\cdot[(\rho E+p)\mathbf u]=\nabla\cdot(\tau\cdot\mathbf u-\mathbf q)+\rho\mathbf g\cdot\mathbf u+\mathbf f_{\rm other}\cdot\mathbf u+S_E,$$

$$E=e+\tfrac12|\mathbf u|^2,$$

$$\partial_t(\rho Y_s)+\nabla\cdot(\rho Y_s\mathbf u+\mathbf j_s)=\dot\omega_s+S_s.$$

$S_m,S_p,S_E$ must describe the *same* material transfers. For chemical reactions without nuclear conversion, $\sum_s\dot\omega_s=0$; for external sources, $\sum_sS_s=S_m$.

### NS2. Stress and heat/species transport closure

$$D=\tfrac12(\nabla u+\nabla u^T),\qquad \tau=2\mu\left(D-\tfrac13(\nabla\cdot u)I\right)+\zeta(\nabla\cdot u)I.$$

$\mu\ge0$ and $\zeta\ge0$ are shear and bulk viscosities in this Newtonian closure. Setting $\zeta=0$ is an assumption, not a universal identity.

$$\mathbf q=-k\nabla T+\sum_sh_s\mathbf j_s,\qquad \sum_s\mathbf j_s=0,\qquad \sum_sY_s=1.$$

A simple mixture-averaged diffusion approximation with mass closure is

$$\mathbf j_s^0=-\rho D_s\nabla Y_s,\qquad \mathbf j_s=\mathbf j_s^0-Y_s\sum_r\mathbf j_r^0.$$

For an isothermal, isobaric ideal-gas mixture, a Maxwell–Stefan alternative is

$$-\nabla x_s=\sum_{r\ne s}\frac{x_r\mathbf J_s-x_s\mathbf J_r}{c_{\rm tot}\,\mathcal D_{sr}},$$

where $\mathbf J_s$ are molar diffusion fluxes relative to the molar-average velocity. Frame conversion is required before inserting them as barycentric mass fluxes. Pressure, thermal and electrochemical diffusion add their corresponding driving forces; do not silently apply the restricted formula outside its domain.

### NS3. Equation of state and energy reference

$$p=p(\rho,e,Y,z),\qquad T=T(\rho,e,Y,z).$$

The simplest ideal-mixture closure is $p=\rho R_uT/\bar M$, $\bar M^{-1}=\sum_sY_s/M_s$. Real-fluid closure comes from Gibbs/data. If $e$ includes species formation energy, do **not** add the same reaction enthalpy again as an independent $S_E$.

### NS4. Incompressible and low-Mach lanes

For constant-density incompressible flow,

$$\nabla\cdot u=0,\qquad \rho(\partial_tu+u\cdot\nabla u)=-\nabla p+\mu\nabla^2u+f.$$

Low Mach number does not imply zero divergence when density changes through heating, composition or phase exchange. Such a lane must preserve its own variable-density continuity and pressure decomposition.

### NS5. Boundaries and moving interfaces

A no-slip wall imposes $u=u_w$. Traction is $(-pI+\tau)n$. Pressure outlets, mass-flow inlets, characteristic compressible boundaries, and thermal/species boundary conditions are distinct choices.

For an interface moving with velocity $v_I$, let $u_n=(u-v_I)\cdot n$. In the absence of surface storage,

$$[\rho u_n]=0,\qquad [\rho u_n u-\boldsymbol\sigma n]=f_{\rm surface},\qquad \boldsymbol\sigma=-pI+\tau.$$

Surface tension supplies normal curvature and tangential Marangoni traction according to the chosen normal/curvature convention. At static equilibrium, the normal pressure jump has magnitude $\gamma_s(1/R_1+1/R_2)$. Species, charge and energy require their own matching interface balances.

### NS6. Reduced passage laws

These are alternative port closures when resolving the passage is unnecessary:

$$\Delta p=\frac{8\mu L}{\pi r^4}Q\quad\text{(steady laminar circular tube)},$$

$$\Delta p=f_D\frac LD\frac{\rho U|U|}{2}+K_{\rm minor}\frac{\rho U|U|}{2},$$

$$Q=C_dA\operatorname{sgn}(\Delta p)\sqrt{2|\Delta p|/\rho}\quad\text{(incompressible orifice)}.$$

For a calorically perfect ideal gas with reservoir $p_0,T_0$, an ideal choked mass-flow closure is

$$\dot m=C_dAp_0\sqrt{\frac{\gamma}{R_sT_0}}\left(\frac2{\gamma+1}\right)^{\frac{\gamma+1}{2(\gamma-1)}},$$

valid on the choked branch $p_b/p_0\le[2/(\gamma+1)]^{\gamma/(\gamma-1)}$. The unchoked branch must be supplied separately. A resolved CFD passage and its reduced pressure-loss element must not both impose the same loss.

### NS7. Turbulence and boundary-layer closures

A Reynolds-averaged model introduces Reynolds stress $-\rho\overline{u'u'}$; a common eddy-viscosity form is

$$-\rho\overline{u'u'}=2\mu_tD-\tfrac23(\rho k_t+\mu_t\nabla\cdot u)I.$$

An example LES subgrid closure is

$$\nu_t=(C_s\Delta)^2\sqrt{2\widetilde D:\widetilde D}.$$

These are fitted closures with filter/wall requirements, not additional fundamental forces. A thin, steady laminar boundary-layer lane can use

$$u u_x+v u_y=-\rho^{-1}p_x+\nu u_{yy},\qquad p_y=0,\qquad u_x+v_y=0.$$

### NS8. Regime and numerical observables

$$Re=\rho UL/\mu,\quad Ma=U/c_s,\quad Pe=UL/\alpha,\quad Kn=\lambda_{\rm mfp}/L.$$

An explicit wave-resolving discretization typically restricts a Courant number based on $|u|+c_s$, while viscous terms add a diffusive restriction. The exact allowed value belongs to the selected spatial operator and integrator, not to the continuous PDE alone.

**Witnesses:** uniform-state preservation; hydrostatic balance; tube-flow solution; contact/shock test for compressible lanes; species closure; conservative port fluxes; mesh/time convergence.

---

## 5. Bjerknes — atmosphere, aerosols, droplets and surface phases

**State:** reduced environmental cells and dispersed-phase distributions or parcels. A resolved Navier–Stokes region replaces the corresponding reduced transport owner; it does not add a second copy of gas mass. References: [R10–R13].

### B1. Bulk atmosphere and rotating-frame reductions

$$\frac{dp}{dz}=-\rho g,\qquad p=\rho R_{\rm mix}T,\qquad \theta=T(p_0/p)^{R_d/c_{pd}}.$$

Potential temperature in this form is for dry ideal air with constant $c_{pd}$. Rotation contributes $-2\Omega\times u$; geostrophic balance is $f\hat z\times u_g=-\rho^{-1}\nabla_hp$. An inviscid material-loop circulation identity, under conservative body forces, is

$$\frac{d\Gamma}{dt}=-\oint\frac1\rho\,dp,\qquad \Gamma=\oint u\cdot d\ell.$$

Reduced turbulent mixing may use $\partial_t\chi=\nabla\cdot(K\nabla\chi)$ with an explicitly modeled diffusivity. These large-scale reductions are optional in a small chamber.

### B2. Saturation, humidity and dew point

Under ideal-vapor/negligible-condensed-volume assumptions,

$$\ln\frac{p_s(T)}{p_s(T_0)}=\int_{T_0}^{T}\frac{L(T')}{R_sT'^2}\,dT'.$$

For constant latent heat,

$$p_s(T)=p_s(T_0)\exp\left[\frac L{R_s}\left(\frac1{T_0}-\frac1T\right)\right].$$

Substituting a varying $L(T)$ into the constant-$L$ closed form does not generally evaluate the integral. An empirical Antoine alternative is $\log_{10}(p_s/p_{\rm unit})=A_A-B_A/(T_{\rm declared}+C_A)$, with units/range bound to the coefficient set.

$$RH=p_v/p_s(T),\quad s=RH-1,\qquad p_s(T_d)=p_v.$$

Magnus inversion is an optional range-limited water approximation, not the general definition of dew point.

### B3. Curvature, solute activity and activation

The general equilibrium ratio for a spherical droplet is

$$S_{\rm eq}(r)=a_w\exp\left(\frac{2\gamma_sM_w}{\rho_lR_uTr}\right).$$

An ideal dilute-solute approximation is

$$S_{\rm eq}=\exp(A_K/r-B_K/r^3),\quad A_K=\frac{2\gamma_sM_w}{\rho_lR_uT},\quad B_K=\frac{3i n_{\rm solute}M_w}{4\pi\rho_l}.$$

$$r_c=\sqrt{3B_K/A_K},\qquad \ln S_c=\sqrt{4A_K^3/(27B_K)}.$$

This is a dilute-solution radius convention. The dissociation factor $i$, solute amount in mol and activity approximation must be explicit. Nonideal activity comes from Gibbs/Lavoisier, not a second incompatible local table.

### B4. Interfacial phase flux: mass versus moles

With positive flux meaning evaporation from condensed phase,

$$j_m=\alpha_e p_s(T_s)\sqrt{\frac{M}{2\pi R_uT_s}}-\alpha_c p_v\sqrt{\frac{M}{2\pi R_uT_v}}.$$

For equal temperatures and accommodation coefficients,

$$j_m=\alpha[p_s(T)-p_v]\sqrt{\frac{M}{2\pi R_uT}}\quad[\mathrm{kg\,m^{-2}\,s^{-1}}],$$

$$j_{\rm mol}=j_m/M=\frac{\alpha[p_s(T)-p_v]}{\sqrt{2\pi M R_uT}}\quad[\mathrm{mol\,m^{-2}\,s^{-1}}].$$

The $1/\sqrt M$ formula is **molar**, not mass flux. Hertz–Knudsen is an interfacial kinetic closure; a continuum diffusion-limited droplet generally requires coupled gas diffusion, interface kinetics and heat transport, not their independently added mass rates.

### B5. Droplet growth and thermal balance

In a quasi-steady continuum diffusion approximation,

$$\dot m_{\rm cond}=4\pi rD_v[\rho_{v,\infty}-\rho_{v,s}(T_p,r,a_w)],\qquad \dot r=\frac{\dot m_{\rm cond}}{4\pi\rho_l r^2}.$$

For a well-mixed droplet with approximately constant properties,

$$mc_p\dot T_p=4\pi r k_g(T_g-T_p)+L_v\dot m_{\rm cond}+P_{\rm rad}.$$

Ventilation, Knudsen and accommodation corrections are selectable closures. More general multicomponent droplets require composition-dependent enthalpy and mass balances, not this one-temperature scalar approximation.

### B6. Phase inventories and latent energy

$$\dot m_v=-\dot m_{\rm cond}+\dot m_{\rm evap}+S_v,\qquad \dot m_l=\dot m_{\rm cond}-\dot m_{\rm evap}-\dot m_{\rm freeze}+\dot m_{\rm melt}+S_l,$$

$$\dot m_i=\dot m_{\rm freeze}-\dot m_{\rm melt}+S_i.$$

With positive $L_v,L_f$,

$$\dot Q_{\rm latent}=L_v(\dot m_{\rm cond}-\dot m_{\rm evap})+L_f(\dot m_{\rm freeze}-\dot m_{\rm melt}).$$

Use this as a sensible-heat source only if latent/formation energy is not already included in the advanced enthalpy. Clipping a negative inventory without compensating all exchanges is not conservation.

### B7. Settling and inertial parcels

$$m_p\dot v_p=m_p g-\rho_gV_pg+F_d+F_{\rm other}.$$

For a small spherical particle with $Re_p\ll1$,

$$v_t=\frac{2r^2(\rho_p-\rho_g)g}{9\mu_g}.$$

At other Reynolds numbers solve the chosen drag-force balance. A smooth blend of Stokes and quadratic asymptotes is an approximation with a validation range, not an exact all-regime law.

### B8. Population balance, collisions and nucleation

For number density distribution $n(m,x,t)$ per unit particle mass,

$$\partial_t n+\nabla_x\cdot(v_p n)+\partial_m(\dot m n)=\mathcal C[n]+\mathcal B[n]+S_n.$$

A mass-conserving binary-coagulation operator is

$$\mathcal C[n](m)=\frac12\int_0^mK(m',m-m')n(m')n(m-m')\,dm'-n(m)\int_0^\infty K(m,m')n(m')\,dm'.$$

For breakup rate $a(m)$ and daughter distribution $b(m|M)$,

$$\mathcal B[n](m)=\int_m^\infty a(M)b(m|M)n(M)\,dM-a(m)n(m),\quad \int_0^M m b(m|M)\,dm=M.$$

Collision kernels include geometry, relative velocity and collection efficiency; these need actual closures. An optional classical nucleation model is

$$r_*=2\gamma_s/\Delta g_v,\quad \Delta G_*=16\pi\gamma_s^3/(3\Delta g_v^2),\quad J=J_0e^{-\Delta G_*/k_BT},$$

where $\Delta g_v>0$ is the bulk free-energy gain per volume. Heterogeneous nucleation, ice pathways and nanoscale limits need their own model/data.

### B9. Surface films, wetting and frost

$$\partial_t\Gamma_s+\nabla_s\cdot(\Gamma_su_f+j_s^{\rm surf})=j_{s,\rm deposit}-j_{s,\rm evaporate}+S_s^{\rm surf}.$$

Film thickness is $h_f=\Gamma/\rho_l$ only for the selected uniform-density film. Young's equilibrium contact relation is $\gamma_{sv}-\gamma_{sl}=\gamma_{lv}\cos\theta_e$; hysteresis requires additional state. Porous frost requires density/porosity and transport closures, not solid-ice conductivity applied blindly.

### B10. Optical coupling from the actual distribution

For number distribution $n(r)$,

$$\beta_{\rm ext}(\lambda)=\int Q_{\rm ext}(r,\lambda)\pi r^2n(r)\,dr,\qquad \mathcal T=\exp[-\int\beta_{\rm ext}\,ds].$$

For a nonmagnetic homogeneous sphere in a nonabsorbing host, $x=k_hr$ and relative refractive index $m_r$ give Rayleigh scattering

$$C_{\rm sca}=\frac{8\pi}{3}k_h^4r^6\left|\frac{m_r^2-1}{m_r^2+2}\right|^2\quad(x\ll1).$$

The full sphere Mie lane instead uses

$$Q_{\rm ext}=\frac2{x^2}\sum_{n=1}^\infty(2n+1)\Re(a_n+b_n),\quad Q_{\rm sca}=\frac2{x^2}\sum_{n=1}^\infty(2n+1)(|a_n|^2+|b_n|^2),$$

with Riccati–Bessel functions $\psi_n(z)=zj_n(z)$ and outgoing $\xi_n(z)=zh_n^{(1)}(z)$:

$$a_n=\frac{m_r\psi_n(m_rx)\psi_n'(x)-\psi_n(x)\psi_n'(m_rx)}{m_r\psi_n(m_rx)\xi_n'(x)-\xi_n(x)\psi_n'(m_rx)},$$

$$b_n=\frac{\psi_n(m_rx)\psi_n'(x)-m_r\psi_n(x)\psi_n'(m_rx)}{\psi_n(m_rx)\xi_n'(x)-m_r\xi_n(x)\psi_n'(m_rx)}.$$

A scalar Rayleigh/Mie blending weight is not a replacement for these cross-sections, angular scattering and their validity assumptions.

**Witnesses:** saturation equilibrium; dimensional check of phase flux; exact species transfer cancellation; droplet growth reference; settling limit; coagulation mass moment; extinction from measured/independent cross-sections.

---

## 6. Gibbs — thermodynamics and phase-state closure

**State/authority:** thermodynamic potentials, consistent property relations and admissible equilibrium constraints. Actual phase inventory/morphology has one shared owner; Gibbs need not independently evolve Bragg's microstructure. References: [R8, R9, R14].

### G1. Fundamental potentials

$$dU=T\,dS-p\,dV+\sum_s\mu_s\,dn_s,$$

$$H=U+pV,\quad A=U-TS,\quad G=U+pV-TS,$$

$$dA=-S\,dT-p\,dV+\sum_s\mu_sdn_s,\qquad dG=-S\,dT+V\,dp+\sum_s\mu_sdn_s.$$

These expressions describe a simple compressible mixture. Strain, polarization, magnetization, interfaces and other generalized work coordinates require their additional conjugate terms.

### G2. Property derivatives and consistency

$$S=-\left(\frac{\partial G}{\partial T}\right)_{p,n},\quad V=\left(\frac{\partial G}{\partial p}\right)_{T,n},\quad \mu_s=\left(\frac{\partial G}{\partial n_s}\right)_{T,p,n_{r\ne s}},$$

$$C_p=\left(\frac{\partial H}{\partial T}\right)_{p,n},\quad C_V=\left(\frac{\partial U}{\partial T}\right)_{V,n},$$

$$\alpha_V=V^{-1}(\partial_TV)_p,\quad \kappa_T=-V^{-1}(\partial_pV)_T,\quad C_p-C_V=\frac{TV\alpha_V^2}{\kappa_T}.$$

A Maxwell identity is $(\partial_pS)_T=-(\partial_TV)_p$. Gibbs–Duhem:

$$S\,dT-V\,dp+\sum_sn_s\,d\mu_s=0.$$

The heat-capacity relation assumes fixed composition and the associated stable single-phase response.

### G3. Thermochemical reference curves

$$h_s^\circ(T)=h_s^\circ(T_0)+\int_{T_0}^Tc_{p,s}^\circ(T')\,dT',$$

$$s_s^\circ(T)=s_s^\circ(T_0)+\int_{T_0}^T\frac{c_{p,s}^\circ(T')}{T'}\,dT',\qquad g_s^\circ=h_s^\circ-Ts_s^\circ.$$

Here quantities are molar. Reference states and temperature ranges are part of the data contract; an arbitrary independent fit for each property may violate these identities.

### G4. Activities, fugacity and electrochemical potential

$$\mu_s=\mu_s^\circ+R_uT\ln a_s,\qquad \widetilde\mu_s=\mu_s+z_sF_c\phi,$$

$$a_s^{\rm solution}=\gamma_s x_s\quad\text{(selected mole-fraction standard state)},$$

$$a_s^{\rm gas}=f_s/p^\circ,\qquad f_s=\phi_s y_sp.$$

Activity and fugacity coefficients need a thermodynamically consistent selected model or data. $a_s$ is dimensionless; do not put dimensional concentration directly inside a logarithm.

### G5. Equations of state

$$pV=nR_uT\quad\text{(ideal gas)},$$

$$(p+a n^2/V^2)(V-nb)=nR_uT\quad\text{(van der Waals closure)}.$$

A thermodynamically integrated specific Helmholtz model $a(T,\rho,Y)$ gives

$$p=\rho^2(\partial_\rho a)_{T,Y},\qquad s=-(\partial_Ta)_{\rho,Y},\qquad e=a+Ts.$$

Validated multiparameter EOS, cubic EOS or tabulated closures may replace the simple examples. Transport coefficients such as viscosity are not determined by an equilibrium EOS alone.

### G6. Phase equilibrium and stability

At fixed $T,p$ and conserved element totals $b$,

$$\min_{n\ge0}G(T,p,n)\quad\text{subject to}\quad A_{\rm elem}n=b\ \text{and the declared charge constraints}.$$

For coexisting phases $\alpha,\beta$ of transferable species,

$$T^\alpha=T^\beta,\quad p^\alpha=p^\beta\quad\text{(flat interface)},\quad \mu_s^\alpha=\mu_s^\beta.$$

Curved/stressed interfaces add their mechanical/generalized-work corrections. Positive constrained second variation identifies local stable equilibrium; a local metastable minimum does not specify a nucleation rate.

$$\frac{dp_{\rm coex}}{dT}=\frac{L}{T\Delta v},\qquad L=T\Delta s.$$

The phase rule $F=C-P+2$ assumes independent thermodynamic components and no extra imposed fields/constraints; reactive species count is not automatically component count.

### G7. Phase kinetics are separate from phase preference

Gibbs supplies free-energy differences or chemical affinities. The kinetics may be a mobility law such as $\dot\xi=-L_\xi\partial_\xi G$, $L_\xi\ge0$, but nucleation, pinning and history require the relevant Bjerknes/Bragg/Lavoisier evolution model. Do not instantly equilibrate a metastable state merely because a lower-energy state exists.

### G8. First and second law for an open system

$$\dot U=\dot Q-\dot W+\sum_{\rm in}\dot m h-\sum_{\rm out}\dot m h\quad\text{(plus kinetic/potential transport when significant)},$$

$$\dot S=\sum_k\dot Q_k/T_k+\sum_{\rm in}\dot m s-\sum_{\rm out}\dot m s+\dot S_{\rm gen},\qquad \dot S_{\rm gen}\ge0.$$

Joule–Thomson closure:

$$\mu_{JT}=\left(\frac{\partial T}{\partial p}\right)_h=\frac{T(\partial_Tv)_p-v}{c_p}.$$

The linear estimate $\Delta T\approx\mu_{JT}\Delta p$ is local; a finite throttle should solve equal enthalpy with the selected EOS.

**Witnesses:** Maxwell/Gibbs–Duhem identities; heat-capacity derivative consistency; phase coexistence references; stable compressibility; nonnegative entropy production under the selected closure.

---

## 7. Bragg — solid-state structure and evolving material response

**State:** lattice/microstructure, defects, plastic history, phase morphology, carriers/traps, polarization/magnetization where enabled. These are selectable submodels; a semiconductor need not instantiate a metallurgical grain-growth model. References: [R15–R19].

### BR1. Crystal geometry and diffraction

$$\mathbf R_{n\alpha}=\sum_{i=1}^3n_i\mathbf a_i+\mathbf r_\alpha,\qquad \mathbf a_i\cdot\mathbf b_j=2\pi\delta_{ij},$$

$$\mathbf G_{hkl}=h\mathbf b_1+k\mathbf b_2+l\mathbf b_3,\qquad 2d_{hkl}\sin\theta=n\lambda,$$

$$F(\mathbf G)=\sum_\alpha f_\alpha(\mathbf G)e^{i\mathbf G\cdot\mathbf r_\alpha}.$$

The structure factor and scattering data decide allowed/intense reflections; Bragg's geometric condition alone is not a diffraction intensity model.

### BR2. Elasticity, thermal strain and finite deformation

$$\epsilon=\tfrac12(\nabla u+\nabla u^T),\qquad \sigma=C:(\epsilon-\epsilon^p-\epsilon^{\rm th}),$$

$$\epsilon^{\rm th}=\int_{T_0}^{T}\alpha(T',z)\,dT'.$$

For an elastic potential, $\sigma=\partial\psi/\partial\epsilon$ and the consistent tangent is $C=\partial\sigma/\partial\epsilon$. In finite strain,

$$F=\frac{\partial x}{\partial X}=F_eF_p,\qquad P=\frac{\partial\Psi}{\partial F},\qquad \sigma=J^{-1}PF^T,\quad J=\det F.$$

$\Psi$ is energy per reference volume with the specified internal variables held fixed. Objectivity and elastic stability are part of the admissible model. Spatial momentum balance is Newton's continuum counterpart, $\rho\ddot u=\nabla\cdot\sigma+f$; a separate solver must not advance the same displacement twice.

### BR3. Macroscopic plasticity and hardening

$$s=\sigma-\tfrac13\operatorname{tr}(\sigma)I,\qquad \sigma_{\rm eq}=\sqrt{\tfrac32s:s},$$

$$f_y=\sigma_{\rm eq}-\sigma_y(\bar\epsilon^p,T,z)\le0,$$

$$\dot\epsilon^p=\dot\lambda\frac{3s}{2\sigma_{\rm eq}},\quad \dot{\bar\epsilon}^p=\dot\lambda,\quad \dot\lambda\ge0,\quad \dot\lambda f_y=0.$$

The displayed associated J2 law is one closure. It does not describe every crystal/polymer/brittle solid. Plastic work $\sigma:\dot\epsilon^p$ partitions into heat and stored microstructural energy; declaring it all heat is an additional assumption.

### BR4. Crystal slip and dislocation kinetics

In a small-strain crystal slip model,

$$\tau_a=s_a\cdot\sigma n_a,\qquad \dot\epsilon^p=\sum_a\dot\gamma_a\operatorname{sym}(s_a\otimes n_a),$$

$$\dot\gamma_a=\dot\gamma_0\left|\frac{\tau_a}{g_a}\right|^{1/m_r}\operatorname{sgn}\tau_a.$$

Slip resistances $g_a$ require a declared hardening/recovery model. Microscopic closures include

$$\dot\gamma=\rho_{\rm mobile}bv_{\rm disl},\qquad \tau_c=\tau_0+\alpha Gb\sqrt{\rho_{\rm disl}}.$$

The finite-strain counterpart requires the appropriate intermediate-configuration stress and slip geometry; do not mix these small-strain expressions into it unchanged.

### BR5. Atomistic metals and defect transport

An embedded-atom potential example is

$$U=\sum_iF_{\alpha_i}\left(\sum_{j\ne i}\rho_{\alpha_j}(r_{ij})\right)+\frac12\sum_{i\ne j}\phi_{\alpha_i\alpha_j}(r_{ij}).$$

Its parameterization must be validated for the intended composition and regime; it is not a universal metal/alloy law. Newton receives its energy gradient.

For a defect concentration,

$$D=D_0e^{-E_a/(R_uT)},\qquad \partial_tc=\nabla\cdot(D\nabla c)+S_c.$$

A more general chemical-potential-driven flux is $j_c=-M_c\nabla\mu_c$. Vacancy production, sinks, trapping and migration energies are declared mechanism/data inputs.

### BR6. Phase morphology and grain evolution

A shared free-energy functional may be

$$\mathcal F[c,\eta]=\int\left[f_{\rm Gibbs}(c,\eta,T)+\frac{\kappa_c}{2}|\nabla c|^2+\frac{\kappa_\eta}{2}|\nabla\eta|^2+f_{\rm elastic}\right]dV.$$

Conserved composition and nonconserved order parameter evolve differently:

$$\partial_tc=\nabla\cdot\left(M_c\nabla\frac{\delta\mathcal F}{\delta c}\right),\qquad \partial_t\eta=-L_\eta\frac{\delta\mathcal F}{\delta\eta}.$$

These are Cahn–Hilliard and Allen–Cahn model families. Gibbs supplies thermodynamic density; Bragg owns the actual microstructure and mobility law. Phase fractions and chemical amounts must map to the same inventory used by the other engines.

### BR7. Fracture and accumulated damage

$$G\ge G_c,\qquad K_I=Y\sigma\sqrt{\pi a},\qquad G=K_I^2/E',$$

where $E'=E$ in plane stress and $E'=E/(1-\nu^2)$ in plane strain. This is linear elastic fracture mechanics. An optional diffuse fracture energy is

$$\mathcal E=\int\left[g(d)\psi_e+\frac{G_c}{2}\left(\frac{d^2}{\ell}+\ell|\nabla d|^2\right)\right]dV,\qquad 0\le d\le1,\quad \dot d\ge0.$$

Its degradation function, irreversibility and tension/compression split must be specified. Optional fatigue law $da/dN=C(\Delta K)^m$ is empirical and range-limited. Creating crack surface consumes energy; not all lost elastic energy is immediately heat.

### BR8. Bands, lattice vibrations and thermal properties

Bloch representation and band kinematics:

$$\psi_{n\mathbf k}(r)=e^{i\mathbf k\cdot r}u_{n\mathbf k}(r),\qquad v_{n\mathbf k}=\hbar^{-1}\nabla_kE_n,\qquad (m^{*-1})_{ij}=\hbar^{-2}\partial_{k_i}\partial_{k_j}E_n.$$

For atomistic force constants $\Phi_{i\alpha,j\beta}=\partial^2U/\partial u_{i\alpha}\partial u_{j\beta}$, the mass-weighted dynamical matrix gives $D(k)e_{k\nu}=\omega_{k\nu}^2e_{k\nu}$.

$$n_B(\omega,T)=\frac1{e^{\hbar\omega/k_BT}-1},\quad C_{k\nu}=k_Bx^2\frac{e^x}{(e^x-1)^2},\quad x=\hbar\omega/k_BT.$$

An optional phonon relaxation-time conductivity is

$$k_{ij}=\frac1V\sum_{k\nu}C_{k\nu}v_{k\nu,i}v_{k\nu,j}\tau_{k\nu}.$$

Scattering lifetimes are model inputs; a lattice geometry alone does not determine them. Band energies may be supplied by Hamilton or validated reduced data.

### BR9. Semiconductor field and carrier transport

With positive elementary charge $q_e$,

$$-\nabla\cdot(\epsilon\nabla\phi)=q_e(p-n+N_D^+-N_A^-)+\rho_{\rm trap},\qquad E=-\nabla\phi,$$

$$J_n=q_e\mu_nnE+q_eD_n\nabla n,\qquad J_p=q_e\mu_ppE-q_eD_p\nabla p,$$

$$\partial_tn=q_e^{-1}\nabla\cdot J_n+G-R,\qquad \partial_tp=-q_e^{-1}\nabla\cdot J_p+G-R.$$

This Poisson solve is the same electrostatic constraint Faraday supplies, not a second independent electric field. In a nondegenerate isothermal mobility model,

$$D_n/\mu_n=D_p/\mu_p=k_BT/q_e.$$

Degenerate, high-field, ballistic and hot-carrier regimes require their corresponding closures rather than this simple drift-diffusion form.

### BR10. Carrier statistics, recombination and traps

$$f_{FD}(E,\mu)=\frac1{e^{(E-\mu)/k_BT}+1},\qquad n=\int_{E_c}^{\infty}D_c(E)f_{FD}(E,\mu_n)\,dE,$$

$$p=\int_{-\infty}^{E_v}D_v(E)[1-f_{FD}(E,\mu_p)]\,dE.$$

Common recombination closures are

$$R_{\rm SRH}=\frac{np-n_i^2}{\tau_p(n+n_1)+\tau_n(p+p_1)},$$

$$R_{\rm rad}=B(np-n_i^2),\qquad R_{\rm Auger}=(C_nn+C_pp)(np-n_i^2).$$

For one trap occupancy $f_t$,

$$\dot f_t=c_nn(1-f_t)-e_nf_t-c_ppf_t+e_p(1-f_t).$$

Carrier capture/emission terms and trap charge must enter the same carrier/charge ledger. Optional reduced device and band-gap models are

$$I=I_s[e^{q_eV/(n_dk_BT)}-1],\qquad E_g(T)=E_g(0)-\alpha_VT^2/(T+\beta_V).$$

The Shockley and Varshni models carry material/device parameters and validity ranges; they do not replace all semiconductor physics.

### BR11. Magnetization and polarization

With $\gamma>0$ in rad/(s·T) and effective field in A/m,

$$\dot M=-\gamma\mu_0M\times H_{\rm eff}+\frac{\alpha_G}{M_s}M\times\dot M,\qquad H_{\rm eff}=-\mu_0^{-1}\frac{\delta\mathcal F}{\delta M}.$$

The effective energy includes the selected exchange, anisotropy, demagnetizing and applied-field terms. A relaxational ferroelectric lane can use $\dot P=-L_P\delta\mathcal F/\delta P$. A Curie–Weiss susceptibility $\chi=C/(T-T_C)$ is a limited equilibrium closure above the transition, not the Curie nuclear engine and not a universal magnetic law.

### BR12. Viscoelastic and hyperelastic material options

A Kelvin–Voigt solid has $\sigma=E\epsilon+\eta\dot\epsilon$; a Maxwell element satisfies $\dot\sigma+\sigma/\tau=E\dot\epsilon$, $\tau=\eta/E$. A generalized linear viscoelastic relaxation model uses

$$G(t)=G_\infty+\sum_aG_ae^{-t/\tau_a},\qquad \sigma(t)=\int_{-\infty}^{t}G(t-s)\dot\epsilon(s)\,ds,$$

with the appropriate tensor and shear/axial conventions. Stored branch strains are genuine history state. Passive branch coefficients and the bath heat ledger are required.

For a finite-strain isotropic compressible neo-Hookean example,

$$\Psi=\frac\mu2(I_1-3)-\mu\ln J+\frac\lambda2(\ln J)^2,\quad I_1=\operatorname{tr}(F^TF),\quad J>0.$$

The selected model determines stress through $P=\partial\Psi/\partial F$. This particular law is not a universal large-strain rubber or near-failure model; parameter range and stability must be checked.

**Witnesses:** energy-gradient forces/tangents; crystal symmetry; elastic patch test; plastic dissipation; defect conservation; free-energy descent under dissipative phase-field dynamics; carrier charge closure; equilibrium diode limit; fixed magnetization norm.

---

## 8. Hamilton — quantum state, electronic structure and molecular geometry

**State:** wavefunction or density matrix in a declared Hilbert space/basis, plus optional environment state. Classical nuclei remain Newton-owned unless a quantum-nuclear lane is explicitly selected. References: [R20–R22].

### H1. Quantum evolution, stationary states and normalization

$$i\hbar\partial_t|\psi\rangle=\hat H|\psi\rangle,\qquad \hat H|n\rangle=E_n|n\rangle,\qquad \hat H=\hat H^\dagger,$$

$$\langle\psi|\psi\rangle=1,\qquad \langle O\rangle=\langle\psi|\hat O|\psi\rangle=\operatorname{Tr}(\hat\varrho\hat O).$$

A fixed nonorthogonal basis yields $i\hbar S\dot c=Hc$, $Hc=ESc$. An evolving basis adds overlap/connection terms; dropping them changes the dynamics.

### H2. Electromagnetic coupling and probability current

$$\hat H=\frac{(-i\hbar\nabla-q\mathbf A)^2}{2m}+q\phi+V,$$

$$\rho_P=|\psi|^2,\qquad \mathbf j_P=\frac\hbar m\Im(\psi^*\nabla\psi)-\frac qm\mathbf A|\psi|^2,\qquad \partial_t\rho_P+\nabla\cdot j_P=0.$$

Spin adds the specified magnetic moment coupling $-\hat\mu\cdot B$, with Pauli spin structure in its applicable regime. The same electromagnetic interaction must not be added again as an unrelated classical Coulomb force on the same electrons.

### H3. Mixed states and open systems

$$\dot{\hat\varrho}=-\frac i\hbar[\hat H,\hat\varrho].$$

A Markovian completely positive open-system lane is

$$\dot{\hat\varrho}=-\frac i\hbar[\hat H,\hat\varrho]+\sum_a\gamma_a\left(L_a\hat\varrho L_a^\dagger-\tfrac12\{L_a^\dagger L_a,\hat\varrho\}\right),\quad \gamma_a\ge0.$$

Require trace one, Hermiticity and nonnegative eigenvalues. Collapse operators/rates require a physical bath model. Any energy transferred to the bath or emitted radiation belongs in an exchange ledger, not an unexplained disappearance.

### H4. Clamped-nuclei electronic Hamiltonian

$$\hat H_e=-\sum_i\frac{\hbar^2}{2m_e}\nabla_i^2-\sum_{iI}\frac{Z_Ie^2}{4\pi\epsilon_0|r_i-R_I|}+\sum_{i<j}\frac{e^2}{4\pi\epsilon_0|r_i-r_j|},$$

$$V_{NN}=\sum_{I<J}\frac{Z_IZ_Je^2}{4\pi\epsilon_0|R_I-R_J|},\qquad E_{BO}(R)=E_e(R)+V_{NN}(R).$$

Electronic antisymmetry and spin are part of the state space. This nonrelativistic Coulomb model is not an all-regime electronic Hamiltonian.

### H5. Molecular geometry and Born–Oppenheimer motion

$$F_I=-\nabla_{R_I}E_{BO},\qquad M_I\ddot R_I=F_I,$$

$$\nabla_RE_{BO}(R_*)=0,\qquad H^{\rm mw}_{I\alpha,J\beta}=\frac{\partial_{R_{I\alpha}}\partial_{R_{J\beta}}E_{BO}}{\sqrt{M_IM_J}},\qquad H^{\rm mw}e_k=\omega_k^2e_k.$$

A stable molecular geometry has nonnegative vibrational curvature after removing the appropriate rigid-motion zero modes. Forces in finite atom-centered bases need basis-response/Pulay contributions; a raw energy eigenvalue alone does not guarantee correct forces.

### H6. Hartree–Fock lane

In **atomic units**, for occupied spin orbitals,

$$\hat F=\hat h+\sum_{j\in\rm occ}(\hat J_j-\hat K_j),\qquad FC=SC\varepsilon,$$

$$E_{HF}=\sum_i h_{ii}+\frac12\sum_{ij}(J_{ij}-K_{ij})+V_{NN}.$$

$\hat J_j\phi_i(r)=\phi_i(r)\int|\phi_j(r')|^2/|r-r'|\,dr'$ and $\hat K_j\phi_i(r)=\phi_j(r)\int\phi_j^*(r')\phi_i(r')/|r-r'|\,dr'$. Occupation, spin restrictions, basis, convergence tolerance and exchange integrals complete this approximation.

### H7. Kohn–Sham density-functional lane

Also in atomic units,

$$\left[-\frac12\nabla^2+v_{\rm ext}+v_H+v_{xc}\right]\phi_i=\varepsilon_i\phi_i,$$

$$n(r)=\sum_i f_i|\phi_i(r)|^2,\quad v_H(r)=\int\frac{n(r')}{|r-r'|}\,dr',\quad v_{xc}=\frac{\delta E_{xc}[n]}{\delta n},$$

$$E[n]=T_s[n]+\int v_{\rm ext}n\,dr+\frac12\iint\frac{n(r)n(r')}{|r-r'|}\,drdr'+E_{xc}[n]+V_{NN}.$$

The exchange-correlation functional, pseudopotential/all-electron choice and basis are explicit model selections. Writing the Kohn–Sham equation does not provide an exact universal functional.

### H8. Reduced Hamiltonians and transitions

A tight-binding model is

$$\hat H=\sum_i\epsilon_i|i\rangle\langle i|+\sum_{i\ne j}t_{ij}|i\rangle\langle j|,\qquad t_{ji}=t_{ij}^*.$$

Transition energy and electric-dipole coupling:

$$\hbar\omega_{fi}=E_f-E_i,\qquad H'(t)=-\hat d\cdot E(t),$$

$$\Gamma_{i\to f}=\frac{2\pi}{\hbar}|\langle f|H'|i\rangle|^2\rho_f(E)\quad\text{(weak-coupling continuum/golden-rule regime)}.$$

A one-dimensional semiclassical tunneling estimate is

$$\mathcal T\approx\exp\left[-\frac2\hbar\int_{x_1}^{x_2}\sqrt{2m(V(x)-E)}\,dx\right].$$

Turning-point and barrier assumptions matter. Quantum harmonic zero-point energy is $E_0=\tfrac12\hbar\omega$, not a classical temperature floor.

### H9. Nonadiabatic and classical-limit checks

For a nuclear trajectory in an adiabatic electronic basis,

$$i\hbar\dot c_a=E_ac_a-i\hbar\sum_b\dot R\cdot d_{ab}\,c_b,\qquad d_{ab}=\langle\phi_a|\nabla_R\phi_b\rangle.$$

This is only the electronic part of a selected mixed quantum/classical scheme; nuclear branching and energy bookkeeping still require a complete method.

$$\frac{d\langle O\rangle}{dt}=\frac i\hbar\langle[H,O]\rangle+\langle\partial_tO\rangle.$$

For a simple scalar potential, $d\langle x\rangle/dt=\langle p\rangle/m$ and $d\langle p\rangle/dt=-\langle\nabla V\rangle$. The variational check is $E_0\le\langle\psi|H|\psi\rangle/\langle\psi|\psi\rangle$ for the applicable ground-state space.

**Witnesses:** norm/trace preservation, Hermiticity/positivity, analytic oscillator/two-level system, gauge consistency, variational bound, basis convergence, molecular force finite differences and vibrational Hessian.

---

## 9. Curie — nuclides, decay and radiological transport interfaces

**State:** nuclide populations, nuclear levels where resolved, emission histories and transport tallies. This roster describes passive decay and general radiation/material interaction accounting; evaluated nuclear data are required. References: [R23–R25].

### C1. Decay, activity and proper-time clock

$$\lambda_i=\ln2/t_{1/2,i},\qquad \frac{dN_i}{d\tau}=-\lambda_iN_i,\qquad N_i(\tau)=N_i(0)e^{-\lambda_i\tau},\qquad A_i=\lambda_iN_i.$$

$\tau$ is physical proper time; for ordinary stationary chamber matter it agrees with the usual simulation time to the model's accuracy. Scheduler time slip is not radioactive time dilation.

### C2. Chains, branches and sources

$$\dot N_i=\sum_jb_{j\to i}\lambda_jN_j-\lambda_iN_i+S_i,\qquad \sum_kb_{i\to k}=1.$$

$$\dot{\mathbf N}=A\mathbf N+S,\qquad \mathbf N(t+\Delta t)=e^{A\Delta t}\mathbf N(t)\quad\text{for constant $A$ and zero $S$}.$$

The matrix form covers branching and equal decay constants without a singular hand-written distinct-rate formula. Explicit daughters, escaping products and external feeds must remain in the material ledger.

### C3. Discrete stochastic realizations

For independent identical nuclei over a constant-rate interval,

$$N_{\rm decayed}\sim\operatorname{Binomial}(N,1-e^{-\lambda\Delta\tau}).$$

Event survival probability is $\exp[-\int\lambda\,d\tau]$. A Poisson event count is only an appropriate limit and must not create more decays than the population. Snapshot/RNG identity belongs to the stochastic state contract.

### C4. Energy release and emission spectra

$$Q=(\sum m_{\rm initial}-\sum m_{\rm final})c^2,$$

using consistent nuclear or atomic mass conventions, including the required electrons. A spectral source can be written

$$S_r(E)=\sum_iA_i\sum_kb_{ik}\,y_{ik,r}(E).$$

$y$ is the evaluated particle/photon yield spectrum per decay in that branch. Momentum, charge and energy include all daughters and emitted particles; element counts need not remain fixed under nuclear transformations.

### C5. Absorption, scattering and deposited heat

For a narrow monoenergetic uncollided photon beam,

$$I(s)=I_0\exp[-\int_0^s\mu(E,x)\,dx].$$

Scattering buildup, secondary radiation and changing energy require a transport equation, not repeated use of one attenuation number. A stopping-power closure is $dE/ds=-S(E,\mathrm{material})$ with range $\int_0^{E_0}dE/S(E)$ under its continuous-slowing-down assumptions.

$$P_{\rm deposited}=\sum_iA_i\,\overline E_{{\rm dep},i},\qquad D_{\rm absorbed}=E_{\rm deposited}/m.$$

Deposited energy excludes escaping photons/particles and neutrino energy. Using total $Q$ as local heat is generally wrong unless the deposition assumptions justify it. Dose here is a physical energy-per-mass tally, not a biological risk model.

### C6. Optional externally driven transformations

For specified incident differential particle flux $\phi(E)$ and evaluated reaction cross-section $\sigma_i(E)$,

$$r_i=N_i\int\sigma_i(E)\phi(E)\,dE.$$

The resulting products enter the same chain matrix/inventory. No reaction yield, half-life or cross-section should be invented from a qualitative stability estimate.

An optional semi-empirical **binding-energy estimate**, not a precision decay model, is

$$B(A,Z)=a_vA-a_sA^{2/3}-a_c\frac{Z(Z-1)}{A^{1/3}}-a_a\frac{(A-2Z)^2}{A}+\delta(A,Z).$$

It does not determine detailed branching or half-lives; evaluated masses and decay data are the default authority.

**Witnesses:** one-nuclide exponential; two-generation chain including equal-rate limit; branch/population closure; stochastic mean and variance; complete energy deposition-plus-escape ledger.

---

## 10. Einstein — relativity and spacetime

**State:** selected physical frame/metric, relativistic momentum and proper time. Relativity changes the constitutive relation between momentum and velocity; it does not invalidate $F=dp/dt$. References: [R26, R27].

### E1. Massive and massless kinematics

$$\gamma=(1-v^2/c^2)^{-1/2},\quad \mathbf p=\gamma m\mathbf v,\quad E=\gamma mc^2,$$

$$E^2=c^2p^2+m^2c^4,\qquad \mathbf v=c^2\mathbf p/E,\qquad d\tau=dt/\gamma.$$

For photons or other massless particles, $E=c|p|$ and propagation is null; a massive-particle proper-time integrator cannot be reused unchanged.

### E2. Relativistic dynamics

$$\dot{\mathbf p}=\mathbf F,\qquad \dot E=\mathbf F\cdot\mathbf v,$$

$$\mathbf a=\frac{\mathbf F-(\mathbf F\cdot\mathbf v)\mathbf v/c^2}{\gamma m}\quad\text{(constant rest mass)}.$$

The $F/(\gamma^3m)$ formula is only the parallel-acceleration case. Multiplying every 3-D acceleration by $\gamma^{-3}$ is not general relativity or general special-relativistic mechanics.

### E3. Lorentz transformations and Doppler shift

For a boost speed $V$ along $x$,

$$t'=\gamma_V(t-Vx/c^2),\quad x'=\gamma_V(x-Vt),\quad y'=y,\quad z'=z.$$

$$p'^\mu=\Lambda^\mu{}_{\nu}p^\nu,\qquad \Lambda^T\eta\Lambda=\eta,\qquad \eta=\operatorname{diag}(-1,1,1,1).$$

For longitudinal recession, $f_{\rm obs}/f_{\rm src}=\sqrt{(1-\beta)/(1+\beta)}$. The general observed frequency follows the contraction of photon four-momentum with observer four-velocity.

### E4. Relativistic field coupling

With convention-matched electromagnetic tensor $F^{\mu\nu}$,

$$\frac{dp^\mu}{d\tau}=qF^\mu{}_{\nu}u^\nu,\qquad u^\mu u_\mu=-c^2.$$

With the stated signature, take $F^{0i}=E_i/c$ and $F^{ij}=\epsilon^{ijk}B_k$; the spatial equation then reduces to $d\mathbf p/dt=q(E+v\times B)$. Faraday's field and Einstein's particle cannot silently use different frames or electromagnetic units.

### E5. Curved-spacetime geometry — optional lane

$$ds^2=g_{\mu\nu}dx^\mu dx^\nu=-c^2d\tau^2,$$

$$\Gamma^\alpha_{\mu\nu}=\frac12g^{\alpha\beta}(\partial_\mu g_{\beta\nu}+\partial_\nu g_{\beta\mu}-\partial_\beta g_{\mu\nu}),$$

$$\frac{d^2x^\alpha}{d\tau^2}+\Gamma^\alpha_{\mu\nu}\frac{dx^\mu}{d\tau}\frac{dx^\nu}{d\tau}=0.$$

An affine parameter replaces proper time for null geodesics. With curvature convention

$$R^\rho{}_{\sigma\mu\nu}=\partial_\mu\Gamma^\rho_{\nu\sigma}-\partial_\nu\Gamma^\rho_{\mu\sigma}+\Gamma^\rho_{\mu\lambda}\Gamma^\lambda_{\nu\sigma}-\Gamma^\rho_{\nu\lambda}\Gamma^\lambda_{\mu\sigma},$$

$$R_{\mu\nu}=R^\alpha{}_{\mu\alpha\nu},\quad R=g^{\mu\nu}R_{\mu\nu},\quad G_{\mu\nu}=R_{\mu\nu}-\tfrac12Rg_{\mu\nu},$$

$$G_{\mu\nu}+\Lambda g_{\mu\nu}=\frac{8\pi G}{c^4}T_{\mu\nu},\qquad \nabla_\mu T^{\mu\nu}=0.$$

A numerical GR lane additionally needs a formulation, gauge and constraint-compatible initial/boundary data. These tensor equations alone do not constitute such a solver.

### E6. Weak-field limit and conserved observables

$$g_{00}\simeq-(1+2\Phi/c^2),\qquad \nabla^2\Phi=4\pi G\rho.$$

A spacetime Killing vector satisfies $\nabla_{(\mu}\xi_{\nu)}=0$ and yields a geodesic conserved quantity $p_\mu\xi^\mu$. An arbitrary evolving spacetime does not supply a universal global scalar energy conserved by summing Newtonian energy ledgers.

**Witnesses:** Lorentz invariant interval/mass shell; boost round trip; uniform force in parallel and perpendicular configurations; low-speed limit; geodesic normalization and the available symmetry-derived constants. Computational latency is not a spacetime metric.

---

## 11. Lavoisier — chemical identity, reactions and electrochemistry

**State:** species amounts, chemical states, reaction coordinates or stochastic populations. Gibbs supplies thermodynamic potentials; transport belongs to its actual transport engine. References: [R8, R9, R28].

### L1. Composition, charge and stoichiometry

Let $A_{es}$ count element $e$ in species $s$, $z_s$ its integer charge and $\nu_{sr}=\nu^+_{sr}-\nu^-_{sr}$ its signed reaction stoichiometry. Require

$$A\nu=0,\qquad z^T\nu=0.$$

$$M_s=\sum_eA_{es}M_e,\qquad b=An,\qquad Q=F_c\sum_sz_sn_s.$$

The species axis distinguishes chemical identity, phase and relevant internal state. Formula-equivalent labels need explicit mappings; string similarity does not establish equivalence.

### L2. Compartment and continuum reaction balances

For well-mixed amount state,

$$\dot n_s=V\sum_r\nu_{sr}r_r+\dot n_s^{\rm boundary}.$$

For a varying volume, $c_s=n_s/V$ implies

$$\dot c_s=\sum_r\nu_{sr}r_r+\dot n_s^{\rm boundary}/V-c_s\dot V/V.$$

The continuum mass source for Navier–Stokes is $\dot\omega_s=M_s\sum_r\nu_{sr}r_r$. Nuclear changes require Curie's accounting rather than enforcing chemical element conservation on them.

### L3. Mass action and rate constants

For a reversible elementary reaction,

$$r_r=k_r^+\prod_sc_s^{\nu^-_{sr}}-k_r^-\prod_sc_s^{\nu^+_{sr}},$$

$$k(T)=A_kT^{b_k}e^{-E_a/(R_uT)}.$$

The rate coefficient's dimensions depend on reaction order. Non-elementary fitted orders are not automatically stoichiometric coefficients. Pressure-dependent falloff, surface coverage and catalytic kinetics are additional declared closures.

A simple third-body/falloff family is

$$[M]_{\rm eff}=\sum_s\alpha_sc_s,\quad P_r=k_0[M]_{\rm eff}/k_\infty,\quad k_{\rm eff}=k_\infty\frac{P_r}{1+P_r}F_{\rm falloff},$$

where $F_{\rm falloff}=1$ is the Lindemann form and other choices need their own parameterization.

### L4. Thermodynamic equilibrium and reaction energy

$$\Delta_rG=\sum_s\nu_{sr}\mu_s,\qquad K_a=e^{-\Delta_rG^\circ/(R_uT)},\qquad Q_a=\prod_sa_s^{\nu_{sr}}.$$

At equilibrium $Q_a=K_a$. For ideal concentration standard state $a_s=c_s/c^\circ$,

$$k^+/k^-=(c^\circ)^{\Delta\nu}K_a,\qquad \Delta\nu=\sum_s\nu_s,$$

provided the matching mass-action convention is used. A dimensional $K_c$ and dimensionless $K_a$ must not be conflated.

$$\dot q_{\rm chem}=-\sum_r\Delta_rH\,r_r,\qquad \dot s_{\rm gen,rxn}=-\frac1T\sum_rr_r\Delta_rG\ge0.$$

The heat source is for an energy convention that excludes the same formation energy. The entropy inequality constrains thermodynamically consistent kinetics; arbitrary independently fitted forward/reverse rates need not satisfy it.

### L5. Electrochemical transport and electrode kinetics

For dilute species in an isothermal solution,

$$J_s=-D_s\nabla c_s-\frac{z_sF_cD_s}{R_uT}c_s\nabla\phi+c_su.$$

This is the molar Nernst–Planck flux under its dilute-solution assumptions. Electrochemical potential gradients give the general nonideal formulation.

$$E_{\rm eq}=E^\circ-\frac{R_uT}{n_eF_c}\ln Q_a,$$

$$j=j_0\left[e^{\alpha_an_eF_c\eta/(R_uT)}-e^{-\alpha_cn_eF_c\eta/(R_uT)}\right],\qquad \eta=E-E_{\rm eq},$$

$$m_{\rm deposited}=\frac{Q_{\rm passed}M}{n_eF_c}.$$

Nernst, Butler–Volmer and Faraday electrolysis relations require explicit electrode stoichiometry, reference potential and current sign. Charge drawn from the circuit must exactly equal the chemical electron transfer.

### L6. Solution, dissolution and surface chemistry

$$pH=-\log_{10}a_{H^+},\qquad K_a^{\rm acid}=a_{H^+}a_{A^-}/a_{HA}.$$

Henry's example convention is $c_s=k_{H,s}p_s$; a different table may use its inverse and different units. An adsorption model is

$$\dot\theta=k_ac(1-\theta)-k_d\theta,\qquad \Gamma_{\rm ads}=\Gamma_{\max}\theta.$$

The corresponding material flux must debit the fluid species and credit the surface species. For reactive surfaces, surface-reaction rates have area rather than volume units and enter volume equations through actual interface geometry.

### L7. Stochastic reaction lane

With integer population $N$, reaction count increment $\nu_r$ and propensity $a_r(N)$,

$$\partial_tP(N,t)=\sum_r[a_r(N-\nu_r)P(N-\nu_r,t)-a_r(N)P(N,t)].$$

Propensities need combinatorial count factors and volume/unit conversion; a concentration ODE rate is not automatically an event probability. Stochastic draws and accepted events are replayable state.

**Witnesses:** exact integer composition/charge balance; detailed equilibrium consistency; concentration/amount conversion including changing volume; reversible two-species relaxation; electrode current/species closure; boundary round trips.

---

## 12. Fourier — heat transport, storage and radiation

**State:** the selected thermal energy field or lumped stores. Gibbs supplies energy–temperature/phase closure; Bragg supplies microstructure-dependent transport. Temperature should be derived from the same owned energy state, not independently stepped by every engine. References: [R14, R29, R30].

### FO1. Conduction and general thermal balance

$$q_c=-\mathsf k\nabla T.$$

For a fixed-composition, single-phase, constant-density medium with the stated heat-capacity convention,

$$\rho c_pD_tT=\nabla\cdot(\mathsf k\nabla T)+\dot q.$$

The more general energy/enthalpy balance must be used for compression, composition change and latent heat. Symmetric positive semidefinite $\mathsf k$ yields conductive entropy production $(\nabla T)^T\mathsf k\nabla T/T^2\ge0$.

### FO2. Sensible/latent storage and lumped networks

$$h(T,z)-h(T_0,z_0)=\int c_p\,dT+\sum_\alpha L_\alpha\Delta f_\alpha$$

is a path/model-specific sensible-plus-latent representation, not a substitute for a consistent multicomponent EOS.

$$\dot U_i=\sum_jH_{ij}(T_j-T_i)+P_i^{\rm external},\qquad T_i=T(U_i,\mathrm{composition},z_i).$$

For a simple slab $H=kA/L$. With equal pair conductance $H_{ij}=H_{ji}$, internal heat exchange cancels globally. For temperature-dependent/phase-changing storage, advancing $U$ and inverting its closure avoids assuming a fixed $mc_p$.

### FO3. Boundary heat transfer and thermal contact

$$q''=h_c(T_s-T_f),\qquad q''=h_{tc}(T_1-T_2),$$

for convection and imperfect solid–solid contact respectively. $h_c$ depends on flow/geometry; $h_{tc}$ on contact conditions. Both sides receive equal and opposite power $Aq''$.

At an ideal interface without surface storage, normal energy flux is continuous. Across a phase interface with common mass flux $j_m$ in the interface frame,

$$[q\cdot n+j_m h]=0$$

when kinetic/work terms are negligible. This is the latent-heat jump in a convention that avoids inserting the same latent source twice.

### FO4. Thermal emission and absorption

$$B_\nu(T)=\frac{2h\nu^3}{c^2}\frac1{e^{h\nu/(k_BT)}-1},\qquad E_b=\sigma_{\rm SB}T^4.$$

For a gray surface seeing a large isothermal environment,

$$P_{\rm net}=\epsilon\sigma_{\rm SB}A(T_s^4-T_{\rm env}^4).$$

This is not the general many-body exchange formula. Under thermal equilibrium assumptions, spectral emissivity and absorptivity are related by Kirchhoff's law with matching direction, polarization and wavelength conventions.

### FO5. Diffuse-gray enclosure radiation

$$G_i=\sum_jF_{ij}J_j,\qquad J_i=\epsilon_i\sigma_{\rm SB}T_i^4+(1-\epsilon_i)G_i,$$

$$P_i=A_i(J_i-G_i),\qquad \sum_jF_{ij}=1,\qquad A_iF_{ij}=A_jF_{ji}.$$

An open environment is an additional boundary in this accounting. A bare two-surface blackbody exchange formula does not model arbitrary gray-surface reflections.

### FO6. Participating-medium radiative transfer

For spectral radiance $I_\nu(x,\Omega,t)$,

$$\frac1c\partial_tI_\nu+\Omega\cdot\nabla I_\nu=-(\kappa_{a,\nu}+\kappa_{s,\nu})I_\nu+\kappa_{a,\nu}B_\nu(T)+\kappa_{s,\nu}\int_{4\pi}p_\nu(\Omega',\Omega)I_\nu(\Omega')\,d\Omega'.$$

The phase function is normalized over outgoing directions. The displayed thermal source assumes local thermodynamic equilibrium; fluorescence/nonthermal emission requires different source laws. Absorption/emission power is exchanged with the material energy ledger. This incoherent description does not replace a phase-resolving Maxwell solve for interference.

### FO7. Heat exchangers and optional thermoelectric coupling

$$C_{\min}=\min(\dot m_hc_{p,h},\dot m_cc_{p,c}),\quad C_r=C_{\min}/C_{\max},\quad NTU=UA/C_{\min},$$

$$\varepsilon=\frac{1-e^{-NTU(1-C_r)}}{1-C_re^{-NTU(1-C_r)}}\quad\text{(ideal counterflow, $C_r\ne1$)},$$

$$\varepsilon=\frac{NTU}{1+NTU}\quad(C_r=1),\qquad \dot Q=\varepsilon C_{\min}(T_{h,in}-T_{c,in}).$$

Phase change or strongly varying properties may require a segmented enthalpy model. A thermoelectric material can use

$$J=\sigma_e(E-S_T\nabla T),\qquad q=\Pi J-k\nabla T,\qquad \Pi=S_TT,$$

with thermodynamically consistent Joule, Peltier and Thomson energy accounting and measured coefficients.

### FO8. Regime diagnostics

$$Bi=h_cL_c/k,\qquad Fo=\alpha\Delta t/L^2,\qquad \alpha=k/(\rho c_p).$$

Small Biot number supports a lumped approximation only with the corresponding geometry/boundary assumptions. Explicit diffusion timestep limits depend on the mesh and discretization; implicit stability does not eliminate accuracy requirements.

**Witnesses:** slab heat equation, two-store equilibration, latent plateau, radiative view-factor reciprocity, emission integral, exchange cancellation and nonnegative entropy production.

---

## 13. Boltzmann — kinetic distributions and statistical mechanics

**State:** phase-space distributions, stochastic populations or selected moments. The mesoscopic lane is not simply a multiplier on one deterministic trajectory. References: [R31–R33].

### BO1. Kinetic transport

For dilute classical particles with velocity-space number distribution $f_s(x,v,t)$,

$$\partial_tf_s+v\cdot\nabla_xf_s+\frac{F_s}{m_s}\cdot\nabla_vf_s=C_s[f]+S_s.$$

This form assumes the usual divergence-free phase-space acceleration, such as Newtonian external or Lorentz acceleration. General phase-space drift requires conservative divergence form.

For elastic single-species binary collisions,

$$C[f](v)=\int_{\mathbb R^3}\int_{S^2}B(|v-v_*|,\Omega)[f'f_*'-ff_*]d\Omega\,dv_*,$$

where $B=g\,d\sigma/d\Omega$ in the scattering-cross-section convention and primed/unprimed pairs are related by the reversible collision map.

### BO2. Moments and continuum coupling

$$n=\int f\,dv,\qquad \rho=mn,\qquad \rho u=m\int vf\,dv,$$

$$\mathsf P=m\int(v-u)\otimes(v-u)f\,dv,\qquad \rho e_{\rm transl}=\frac m2\int|v-u|^2f\,dv,$$

$$q=\frac m2\int|v-u|^2(v-u)f\,dv.$$

Polyatomic internal energy adds internal-state coordinates/moments. These moments define what a Boltzmann region exchanges with a Navier–Stokes region; a matched density alone is insufficient.

### BO3. Equilibrium and BGK relaxation

$$f_M=n\left(\frac{m}{2\pi k_BT}\right)^{3/2}\exp[-m|v-u|^2/(2k_BT)],$$

$$C_{BGK}=-(f-f_M)/\tau.$$

$f_M$ must match the local conserved moments. For a monatomic ideal BGK gas, the continuum-limit transport coefficients satisfy $\mu=p\tau$ and $Pr=1$; other Prandtl numbers require a modified collision closure. The Chapman–Enskog expansion $f=f^{(0)}+Kn f^{(1)}+\cdots$ derives continuum constitutive response in its small-Knudsen regime, not in arbitrary rarefied flow.

### BO4. Collision invariants and entropy

$$\int C[f]\{1,mv,\tfrac12mv^2\}\,dv=0,$$

$$H=\int f\ln(f/f_*)\,dx\,dv,\qquad \left.\frac{dH}{dt}\right|_{\rm collisions}\le0.$$

The arbitrary fixed reference $f_*$ makes the logarithm dimensionless. The H-theorem assumes the selected reversible molecular-collision model; boundary entropy fluxes and external driving must be included separately.

### BO5. Regime selection and free-molecular surfaces

$$\lambda_{\rm mfp}=\frac1{\sqrt2n\sigma_{\rm coll}},\qquad Kn=\lambda_{\rm mfp}/L$$

for the hard-sphere equilibrium estimate. The actual collision cross-section depends on the interaction model. A Maxwell wall model mixes specular reflection with an accommodation-weighted wall Maxwellian; its outgoing density is chosen to satisfy the prescribed wall mass flux. Wall temperature/momentum accommodation must be declared.

### BO6. Statistical ensembles and quantum occupation

$$Z=\operatorname{Tr}(e^{-\beta H}),\quad \beta=(k_BT)^{-1},\quad A=-k_BT\ln Z,$$

$$U=-\partial_\beta\ln Z,\qquad S=k_B(\ln Z+\beta U).$$

For classical phase-space ensembles the integral measure includes the appropriate $h^{3N}$ and indistinguishability factors. Mean quantum occupation is

$$\bar n_{FD}(\epsilon)=\frac1{e^{\beta(\epsilon-\mu)}+1},\qquad \bar n_{BE}(\epsilon)=\frac1{e^{\beta(\epsilon-\mu)}-1}.$$

Photons/phonons in the corresponding equilibrium model have chemical potential zero. Equilibrium occupations do not by themselves provide nonequilibrium quantum collision dynamics.

### BO7. Brownian/Langevin reduction

For an isothermal Markovian bath with translational drag coefficient $\zeta$,

$$dx=v\,dt,\qquad m\,dv=F_{\rm cons}\,dt-\zeta v\,dt+\sqrt{2\zeta k_BT}\,dW,$$

$$D=k_BT/\zeta.$$

Friction and noise are paired by fluctuation–dissipation. A thermostat is an energy-exchanging bath, not free damping; its heat/work belongs in Noether's ledger. Multiplicative noise or constrained coordinates require the correct stochastic-calculus convention and numerical scheme.

### BO8. General linear particle/radiation transport

For angular flux $\psi(x,\Omega,E,t)$,

$$\frac1{v(E)}\partial_t\psi+\Omega\cdot\nabla\psi+\Sigma_t\psi=\int\Sigma_s(E',\Omega'\to E,\Omega)\psi(E',\Omega')\,dE'd\Omega'+S.$$

Curie may supply sources, materials supply interaction kernels, and Fourier receives deposited energy. The chosen angular-flux/radiance/number-density normalization is part of every port.

**Witnesses:** collision moment invariants; Maxwellian stationary solution; H-theorem test; Brownian mean-square displacement; correct hydrodynamic limit; matching mass, momentum and energy across a kinetic/continuum seam.

---

## 14. Noether — invariants and physical-ledger verification

**Role:** a cross-cutting auditor and source of mathematical constraints, not a replacement integrator and not a mechanism for silently repairing bad states. References: [R34, R26, R27]. The implementation contract below is proposed architecture.

### NO1. Symmetry-derived conserved quantities

For a variational symmetry $\delta q_a=\eta_a\varepsilon$, $\delta t=\tau\varepsilon$, whose Lagrangian action changes by the boundary term $dB/dt$, the on-shell charge is

$$Q_N=\sum_a\frac{\partial L}{\partial\dot q_a}\eta_a-H\tau-B,\qquad \frac{dQ_N}{dt}=0.$$

For an internal field symmetry at fixed coordinates,

$$j^\mu=\sum_a\frac{\partial\mathcal L}{\partial(\partial_\mu\phi_a)}\delta\phi_a-K^\mu,\qquad \partial_\mu j^\mu=0.$$

Time, spatial and rotational symmetries yield their respective energy/momentum/angular-momentum charges where those symmetries exist. Global phase symmetry gives the associated charge; local gauge symmetry also entails constraint identities. Not every bookkeeping invariant is literally an application of Noether's first theorem.

### NO2. Open-domain conservation residual

For any selected conserved extensive quantity $Q$,

$$R_Q=Q(t_1)-Q(t_0)+\int_{t_0}^{t_1}\oint_{\partial\Omega}\mathcal F_Q\cdot n\,dA\,dt-\int_{t_0}^{t_1}\int_\Omega S_Q\,dV\,dt.$$

The assertion is $R_Q=0$ in the continuous model and within the declared numerical error in its discretization. Energy, mass, charge and each momentum component have different units and flux definitions; one undimensioned residual threshold is insufficient.

### NO3. Internal exchanges and single energy ownership

For an instantaneous exchange between two stores,

$$\Delta Q_{a\to b}^{(a)}+\Delta Q_{a\to b}^{(b)}=0.$$

For a finite-propagation link with storage,

$$\Delta Q_a+\Delta Q_{\rm link}+\Delta Q_b=0.$$

Energy must be summed over **disjoint physical stores**, not over every engine's reported energy indiscriminately. An electronic energy already represented in a force-field potential, phonon energy already represented as thermal energy, or phase energy already included in enthalpy must not be counted again under another engine name.

### NO4. Chemical, nuclear and field checks

Chemistry without nuclear conversion:

$$A\Delta n=A\Delta n_{\rm boundary},\qquad F_cz^T\Delta n=\Delta Q_{\rm boundary}.$$

For nuclear transformations use the appropriate total charge, energy/momentum and declared particle/nuclear inventories. Rest mass alone and each chemical element are not individually conserved through transmutation.

Faraday checks include

$$d_2b=0,\qquad \text{Gauss residual}=\nabla\cdot D-\rho_f,\qquad \partial_t\rho_f+\nabla\cdot J_f=0.$$

These are propagated from constraint-compatible initial state and boundary currents; a compatible incidence matrix alone cannot fix inconsistent initialization or charge deposition.

### NO5. Inequalities and admissibility

$$\rho>0,\quad T>0,\quad n_s\ge0,\quad \sum_sY_s=1,\quad \dot S_{\rm gen}\ge0,$$

$$\hat\varrho=\hat\varrho^\dagger,\quad \operatorname{Tr}\hat\varrho=1,\quad \hat\varrho\succeq0,$$

$$P_{\rm diss}\ge0,\qquad R^TR=I,\quad \det R=1.$$

Admissibility does not mean all models share the same positivity/stability criteria: a physical instability or an unstable free-energy state is not automatically a numerical error.

### NO6. Numerical and model-change evidence

Recommended normalized residual:

$$r_Q=\frac{|R_Q|}{a_Q+r_Q^{\rm tol}Q_{\rm scale}}.$$

Use separately named absolute/relative tolerances with physical units, and retain the raw residual. For an expected order $p$, the refinement relation $\|y_h-y_{h/2}\|\approx2^p\|y_{h/2}-y_{h/4}\|$ is a useful diagnostic in its asymptotic regime, not a proof for every trajectory.

A proposed optimization must distinguish exact real-arithmetic identities, floating-point behavior, numerical convergence and physical validation. Passing finite test vectors does not prove equivalence for every state. A changed closure may preserve conservation while predicting the wrong physics; independent measurements/solutions remain necessary.

**Witnesses:** intentionally mis-signed heat transfer is detected; double-counted latent energy is detected; weighted populations close; failed speculative steps leave no ledger entries; replacement kernels preserve the declared observables and error bounds.

---

## 15. Cross-engine coupling and model-promotion equations

The following are **proposed simulator contracts**, not claims that the repository already implements them.

### X1. Mechanical, thermal, electrical and chemical power ports

$$P_{\rm mech}=F\cdot v+\tau\cdot\omega,\quad P_{\rm electric}=VI,\quad P_{\rm heat}=\dot Q,\quad P_{\rm chemical}=\sum_s\mu_s\dot n_s.$$

These conjugate pairs identify consistent work/exchange terms. They do not authorize summing chemical-potential work and transported enthalpy as independent contributions when a chosen thermodynamic balance already includes the same energy.

### X2. Transport through a geometric port

For boundary normal $n$ and boundary velocity $v_b$,

$$\dot m=\int_A\rho(u-v_b)\cdot n\,dA,$$

$$\dot m_s=\int_A[\rho Y_s(u-v_b)+j_s]\cdot n\,dA.$$

Momentum and energy use the matching conservative flux, including stress/work and heat transport. The same oriented aperture is shared by fluid, thermal, chemical and field models; each applies its own boundary law to the same geometry.

### X3. Population scaling

For weights $w_i$ representing counts of equivalent instances,

$$Q_{\rm total}=\sum_iw_iQ_i,\qquad \dot Q_{\rm total}=\sum_iw_i\dot Q_i.$$

Temperatures, pressures, phase fractions and probabilities are not multiplied by population count. A thermal aggregate obtains temperature from total energy and its EOS, not an unweighted mean unless that mean is physically justified.

For nonlinear observables,

$$\sum_iw_if(x_i)\ne\left(\sum_iw_i\right)f\!\left(\frac{\sum_iw_ix_i}{\sum_iw_i}\right)\quad\text{in general}.$$

A representative-particle edge must state whether it preserves a trajectory, an expectation, moments, or a distribution; fluctuations/correlations cannot be silently replaced by deterministic multiplicity.

### X4. Fine-to-coarse restriction

A basic particle-to-cell restriction is

$$M=\sum_iw_im_i,\quad P=\sum_iw_ip_i,\quad E=\sum_iw_iE_i,\quad N_s=\sum_iw_iN_{s,i}.$$

Angular momentum and material history must also be preserved where observable. Bulk kinetic energy is $|P|^2/(2M)$; kinetic energy relative to the mean must be retained as thermal/internal or explicitly unresolved energy, not deleted.

### X5. Coarse-to-fine lifting

A coarse state does not uniquely determine microscopic positions, velocities, defects or a quantum wavefunction. A lifting map $L$ and restriction map $R$ should satisfy

$$R(L(y_{\rm coarse};\xi))=y_{\rm coarse}$$

for the preserved observables, with $\xi$ naming the declared ensemble/initialization information. This is conservation-compatible initialization, not recovery of the actual lost microstate. An invalid coarse closure may require refusing promotion until necessary boundary/data information exists.

### X6. Geometry is an address, not automatically a volume model

A triangle can bind a surface law and expose a deeper region. A volumetric thermal, CFD, elastic, atomistic or quantum solve still needs a volume or dual-cell/thickness interpretation, neighbors, boundary conditions and the appropriate basis/mesh. Never infer an entire physical volume merely from three vertices.

### X7. State and source identity before CSE

The composition key includes physical identity, quantity, units, frame, material version, time level and integrator stage. Only expressions reading the same bound state at the same relevant stage are candidates for exact sharing. Wall temperature, gas temperature and droplet temperature are not one symbol because they are all spelled `T`.

---

## 16. Numerical realization and verification gates

### 16.1 Separate physics from discretization

A useful implementation chain is

```text
continuous law + closure + boundary conditions
→ spatial weak/finite-volume/DEC/basis discretization
→ semidiscrete residual and Jacobians
→ selected dt_system integrator / algebraic or eigen solve
→ symbolic discrete equations / state-machine stages
→ ProcessGraph / SSA / compiled artifact
→ Nodus composition and execution
```

A PDE cannot be compiled into a faithful simulator merely by emitting its symbols. Discrete operators, quadrature, state layout and boundary rules are mathematical inputs too. Stabilization, limiting, filtering and regularization belong in the auditable model/numerical specification.

### 16.2 Compile-time fusion does not require temporal fusion

One compiled entry may execute several integration rates, event boundaries or algebraic stages. A single-step fused update with one shared `dt` is one *particular schedule*, not an unavoidable property of native compilation. Common-subexpression elimination does not select time semantics. Preserve existing `dt_system` ownership instead of creating an engine-local clock.

### 16.3 Required evidence for an implemented bundle

Each instantiated model should expose the following checks:

| Evidence class | Required question |
|---|---|
| Dimensional | Do all sides and all port quantities have compatible units? |
| Algebraic | Do identities, stoichiometry, nullspaces and symmetry conditions hold? |
| Domain | Are the data, closure and approximation valid for this state? |
| Constitutive | Are coefficients and material functions consistent and independently grounded? |
| Numerical | Does the selected discretization converge and preserve the claimed invariants? |
| Cross-engine | Do mass, charge, momentum, energy and material identities close at interfaces? |
| Backend | Does a compiled realization match the specified reference observables/error contract? |
| Promotion | Does restriction/lifting preserve promised observables and unresolved energy? |
| Experimental | Does the model match independent measurements or established benchmark solutions? |

The first seven mathematical checks are not substitutes for the last empirical question. Conversely, a model can fit selected measurements while violating conservation elsewhere.

### 16.4 A practical chamber convergence witness

Begin from the existing law families, not new parallel implementations. One closed chamber witness can couple a gas species inventory, a condensing surface or droplet population, a thermal wall and a driven passive EM/material response. Check that changes in phase or temperature alter the selected constitutive properties, that every exchange is counted once, and that the symbolic/reference and compiled paths agree over the same accepted time intervals.

This witness need not instantiate all fourteen domains. Quantum, nuclear and relativistic lanes are activated when the physical question needs them, with analytic small-system witnesses first. “General simulator” means composable validated descriptions, not making every wall solve every scale continuously.

---

## 17. Reference register

The user-supplied naming/proposition transcript defines the engine names and architectural intent. The following primary papers, original research/teaching notes, and official implementation documentation support the mathematical model families. These references do not certify this document or the user's code; each implemented closure still requires its own parameter provenance and validation.

- **R1. OpenMM, Standard Forces.** Harmonic bonds/angles, torsions, nonbonded potentials and conventions. https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html
- **R2. LAMMPS, Lennard-Jones pair style.** Potential and parameter/cutoff conventions. https://docs.lammps.org/pair_lj.html
- **R3. LAMMPS, Langevin thermostat.** Force/noise/bath coupling and energy accounting. https://docs.lammps.org/fix_langevin.html
- **R4. MOOSE, C0 Timoshenko Beam.** Beam kinematics, interpolation and structural formulation. https://mooseframework.inl.gov/modules/solid_mechanics/C0TimoshenkoBeam.html
- **R5. SU2, Governing Equations.** Compressible/incompressible fluid, energy and structural equations. https://su2code.github.io/docs_v7/Theory/
- **R6. Meep, Introduction.** Maxwell/FDTD mathematical formulation. https://meep.readthedocs.io/en/latest/Introduction/
- **R7. Meep, Materials.** Dispersive constitutive models, loss and field/material coupling. https://meep.readthedocs.io/en/latest/Materials/
- **R8. Cantera, Phase Thermodynamic Properties.** Mixture/phase thermodynamic conventions. https://cantera.org/stable/reference/thermo/phase-thermo.html
- **R9. Cantera, Species Thermodynamic Properties.** Consistent heat-capacity, enthalpy and entropy descriptions. https://cantera.org/stable/reference/thermo/species-thermo.html
- **R10. PySDM, Physics formula modules.** Atmospheric microphysics, droplets, phase properties, optics and dimensional-analysis organization. https://open-atmos.github.io/PySDM/PySDM/physics.html
- **R11. Petters and Kreidenweis (2007), A single parameter representation of hygroscopic growth and cloud condensation nucleus activity.** Primary hygroscopicity/activation paper. https://acp.copernicus.org/articles/7/1961/2007/
- **R12. Scott Prahl, miepython: Mie Scattering Algorithms.** Sphere scattering formulas and numerical conventions. https://miepython.readthedocs.io/en/latest/07_algorithm.html
- **R13. PySDM, Diffusion kinetics.** Transition-regime corrections for condensational growth/evaporation. https://open-atmos.github.io/PySDM/PySDM/physics/diffusion_kinetics.html
- **R14. NIST Chemistry WebBook, Thermophysical Properties of Fluid Systems.** Evaluated properties and units; appropriate data input rather than invented constants. https://webbook.nist.gov/chemistry/fluid/
- **R15. LAMMPS, Embedded-atom method.** Many-body metal potential structure. https://docs.lammps.org/pair_eam.html
- **R16. NIST Interatomic Potentials Repository.** Parameterized potential provenance and scope. https://www.ctcms.nist.gov/potentials/
- **R17. MOOSE, Phase Field Equations.** Conserved and nonconserved order-parameter evolution. https://mooseframework.inl.gov/modules/phase_field/Phase_Field_Equations.html
- **R18. MOOSE, Finite Strain Crystal Plasticity.** Reference implementation of crystal slip/hardening; the page marks its implementation deprecated, so use it as formulation documentation, not a recommendation to adopt that class. https://mooseframework.inl.gov/source/materials/crystal_plasticity/FiniteStrainCrystalPlasticity.html
- **R19. COMSOL, The Semiconductor Equations.** Carrier statistics, charge and drift-diffusion equations. https://doc.comsol.com/6.2/doc/com.comsol.help.semicond/semicond_ug_semiconductor.6.55.html
- **R20. QuTiP, Lindblad Master Equation Solver.** Closed/open quantum evolution and its assumptions. https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-master.html
- **R21. Psi4, Hartree–Fock Theory.** SCF mathematical formulation. https://psicode.org/psi4manual/master/scf.html
- **R22. GPAW, Basic formalism.** Electronic-structure and density-functional formulation. https://gpaw.readthedocs.io/documentation/basic.html
- **R23. OpenMC, Depletion.** General decay/transformation population equations and matrix evolution. https://docs.openmc.org/en/stable/methods/depletion.html
- **R24. Brookhaven National Nuclear Data Center, ENSDF.** Evaluated nuclear structure and decay data. https://www.nndc.bnl.gov/ensdf/
- **R25. NIST, X-Ray Mass Attenuation Coefficients, Section 2.** Narrow-beam attenuation and interaction-data assumptions. https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html
- **R26. Sean Carroll, Lecture Notes on General Relativity: Special Relativity and Flat Spacetime.** https://ned.ipac.caltech.edu/level5/March01/Carroll3/Carroll1.html
- **R27. Sean Carroll, Lecture Notes on General Relativity: Gravitation.** Curvature, Einstein equations and conservation structure. https://ned.ipac.caltech.edu/level5/March01/Carroll3/Carroll4.html
- **R28. Cantera, Reaction Rates.** Rate-law types, temperature dependence and pressure-dependent closures. https://cantera.org/stable/reference/kinetics/reaction-rates.html
- **R29. COMSOL, Theory for the Heat Transfer Module.** Heat-transfer mathematical theory and linked submodels. https://doc.comsol.com/6.2/doc/com.comsol.help.heat/heat_ug_theory.07.001.html
- **R30. COMSOL, Conductive Thermal Resistor.** Thermal network conductance conventions. https://doc.comsol.com/6.2/doc/com.comsol.help.heat/heat_ug_theory.07.043.html
- **R31. Richard Fitzpatrick, Boltzmann Collision Operator.** Kinetic binary-collision formulation. https://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node33.html
- **R32. Richard Fitzpatrick, Collisional Conservation Laws.** Number, momentum and energy moments. https://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node34.html
- **R33. LAMMPS, Langevin thermostat.** Fluctuation–dissipation and bath reduction; same primary reference as R3. https://docs.lammps.org/fix_langevin.html
- **R34. Emmy Noether, Invariant Variation Problems, translated by M. A. Tavel.** Primary symmetry/variational-theorem source. https://arxiv.org/html/physics/0503066v3

## Closing contract

**A hosted engine is not just a formula. It is a declared physical model with owned state, explicit assumptions, conservative interfaces, an auditable numerical realization, and independent witnesses.**

The equations are the model's mathematical authority. Data and experiments establish its physical range. Turing/Nodus may change representation and execution placement without silently changing either.
