# Annex: the equation roster for every honorary engine

Companion to
[HIERARCHICAL_PHYSICS_SIMULATOR_PROPOSITION.md](HIERARCHICAL_PHYSICS_SIMULATOR_PROPOSITION.md).
That document names the fourteen honorary engines and what each one
owns. This annex goes one level deeper: for each engine, it separates

1. **already authored** — equations that exist today as SymPy law
   sets or as explicit, named plain-Python formulas somewhere in
   `turing/` or `engine_toy/` (both sibling repositories, not tracked
   in this one — cited with plain code spans, not links), so an
   equation-system bundle for that engine could be assembled by
   collecting and re-authoring these rather than inventing them, from
2. **the canonical roster** — the textbook equations a complete,
   authoritative treatment of that field requires, whether or not the
   repository has written them yet, so a bundle author knows what is
   still missing rather than mistaking "what happens to exist" for
   "the complete physics."

This is not a build plan. No SymPy source was written or changed to
produce this annex. It is the survey §14 of the main proposition
("SymPy laws as scientific authority") calls for before any bundle is
authored: reading what is already true before adding to it.

## Method and honesty notes

- "Already authored" entries were read directly from source; equations
  are transcribed in plain-text math notation, not copied verbatim as
  code, and each cites its file and, where the survey found one, its
  law/function name.
- A law authored as **plain Python** (an explicit named formula, not a
  `sympy.Eq`) is marked as such. It is real, working physics — most of
  it is fully compilable if re-authored as SymPy — but it has not yet
  gone through `compile_sympy_equations`/`compile_symbolic_program`,
  so it does not yet carry a `dt_limit`, an energy/power channel, or a
  `LAWS`-table identity the way the files in `turing/examples/` do.
- Several existing laws sit at a domain boundary and are noted as
  such rather than forced into one engine: `schrodinger_step` is
  authored today inside `turing/examples/symbolic_em_solvers.py`
  (a Faraday-named file) but is squarely a **Hamilton** law; several
  entries under Fourier and Gibbs are the same physical fact (thermal
  storage's `E/kg = c_p·ΔT + L`) already combining a Fourier term
  (sensible heat) and a Gibbs term (latent heat) in one formula, which
  is exactly the "thermal is a shared consumer" pattern §12 of the
  main proposition describes, not a filing error.
- Two canonical constants carry the same name as an engine without
  being that engine's physics: the **Faraday constant** `F` (charge
  per mole of electrons, used in electrolysis stoichiometry) is
  Lavoisier-domain chemistry, not the Faraday field engine; the
  **Curie–Weiss law** (`chi = C/(T - T_c)`, magnetic susceptibility
  near a phase transition) is Bragg-domain solid-state physics, not
  the Curie nuclear engine. Both are flagged inline where they appear
  so the naming coincidence does not get mistaken for a domain match.

---

## 1. Newton — classical force / particle / rigid-body engine

### Already authored

| Law | Equation | Source |
|---|---|---|
| Interior-ballistics motion | `travel_next`, `velocity_next` from breech/base pressure, `recoil_velocity_next` from momentum balance, `shot_energy_j = 1/2 m v^2`, `recoil_energy_j = 1/2 m_recoil v_recoil^2` | `engine_toy/interior_ballistics.py::symbolic_ballistics_equations` |
| Reduced tire contact | `contact_patch_area` from a Gauss–Legendre integral of the ring profile; `contact_force_n = p_gas · area` | `turing/src/compiler/vehicle_tire_reduced_contact_law.py::symbolic_reduced_contact_law_equations` |
| Member plasticity (J2-like return map) | `sigma_eq = sqrt(sigma_axial^2 + sigma_bending^2 + 3 tau^2)`; yield `sigma_y = sigma_y0 + H·eps_p`; plastic multiplier and updated axial/bending/shear plastic strain; `elastic_energy_j`, `plastic_work_j` | `turing/src/compiler/vehicle_mechanical_material.py::symbolic_vehicle_member_material_equations` |
| Membrane/gas/contact laws for a pressurized shell | strain energy, dissipation power, per-vertex elastic/damping/pressure/construction forces; ideal-gas polytropic pressure `p = p_ref (V_ref/V)^n`, `T = T_ref (V_ref/V)^(gamma-1)`; bead spring-damper reaction; vertex–triangle and vertex–cylinder continuous-collision-detection (CCD) geometry; frictional unilateral contact impulse `j_n = -(1+e)·v_closing / m_eff^-1`, `j_t = min(|v_t|/m^-1, mu·j_n)` | `turing/src/compiler/vehicle_balloon_tire.py` (`symbolic_balloon_membrane_face_equations`, `symbolic_balloon_gas_equations`, `symbolic_balloon_bead_constraint_equations`, `symbolic_balloon_vertex_triangle_contact_equations`, `symbolic_balloon_cylinder_contact_geometry_equations`, `symbolic_balloon_contact_impulse_equations`) |
| General world-body integration | position/velocity update, AABB and obstacle contact penetration, specific kinetic energy, gravity, drag, portal traversal | `turing/src/compiler/abstract_ui_physics.py::symbolic_world_physics_equations` |
| Full vehicle rigid-body dynamics | chassis translation/rotation update, tire compression/slip/friction utilization, spring/damper forces, traction control, braking, driveline and engine reaction torque, wheel angular update | `turing/src/compiler/abstract_ui_vehicles.py::symbolic_vehicle_equations` |
| Rigid-body force library (numeric, not yet SymPy) | rope, steel-beam, spring, gas-damper, lever-arm force laws | `turing/src/common/dt_system/classic_mechanics/rigid_body_engine.py` |

### Canonical roster

- **Newton's second law**, generalized: `F = dp/dt` (translation), `tau = dL/dt` (rotation), reducing to `F = m·a` and `tau = I·alpha` for constant mass/inertia — the assertion the main proposition already states Newton owns; every force law above is a *provider* into this, never a replacement of it.
- **Newton's third law**: `F_ab = -F_ba` for any interacting pair — already implicit in the balloon contact/rim-reaction pair (`rim_force = -skin_force`) but worth stating as the general identity a bundle should assert once, not per-force.
- **Newton's law of gravitation**: `F = G m1 m2 / r^2`.
- **Coulomb friction**: `|F_t| <= mu |F_n|`, with the balloon contact impulse law above as the discrete-time instance.
- **Newton's law of restitution**: `v'_separating = -e · v_closing`, `0 <= e <= 1` — present implicitly in the impulse law's `(1+e)` term but not stated as its own general identity.
- **Work–energy theorem**: `dT/dt = F·v` (power delivered equals rate of kinetic-energy change).
- **Conservation of linear and angular momentum** for an isolated system: `sum(p_i) = const`, `sum(L_i) = const` — the discrete ledger a Noether-style audit (§14 below) would check against every Newton step.
- **Euler's equations for rigid-body rotation** (body frame, principal axes): `I1 dw1/dt - (I2-I3) w2 w3 = tau1` (and cyclic) — needed once free rotation of a non-spherical rigid body (not just a wheel/chassis) is asked for.
- **Lagrangian/Hamiltonian form** as an alternate authoring path: `d/dt(dL/dq_dot) - dL/dq = Q_nonconservative`, useful when a constrained multibody system is easier to state via generalized coordinates than via per-force accumulation.
- **Drag/aerodynamic force**: `F_drag = -1/2 rho C_d A |v| v` — present as a term inside the vehicle/world-physics laws above but not as its own standalone reusable provider.

---

## 2. Timoshenko — beam / frame / reduced structural engine

### Already authored

| Law | Equation | Source |
|---|---|---|
| Beam kinematics | `axial_strain = delta_axial/L`; curvature `kappa_y = (theta_y_b - theta_y_a)/L` (and z); `bending_strain = sqrt(kappa_y^2+kappa_z^2)·r_outer`; transverse shear `gamma_y = delta_lateral_y/L - (theta_z_a+theta_z_b)/2` (and z); `twist = (theta_x_b - theta_x_a)/L`; `shear_strain = sqrt(gamma_y^2+gamma_z^2+gamma_torsion^2)` | `engine_toy/beam_theory.py::symbolic_beam_kinematics_equations` |
| Geometric/elastic stiffness and buckling | `k_elastic = 12EI/L^3`; `k_geometric = 6P/(5L)`; `k_effective = k_elastic + k_geometric`; Euler buckling `P_cr = pi^2 EI / L^2`; `buckling_demand = -P/P_cr` | `engine_toy/beam_theory.py::symbolic_geometric_stiffness_equations` |
| First-mode cantilever oscillator | `q_ddot + 2 zeta omega q_dot + omega^2 q = F/m_modal`; `f_n = omega/(2 pi)` | `engine_toy/beam_theory.py::symbolic_beam_dynamics_equations` |
| Member elastoplastic stress/energy (shared with Newton above) | `sigma_axial`, `sigma_bending`, `tau_shear`, elastic/plastic energy split | `turing/src/compiler/vehicle_mechanical_material.py` |
| Numeric frame/Timoshenko element assembly (not yet SymPy) | element stiffness matrices, generalized-eigenvalue modal solve | `engine_toy/frame_solver.py`, `engine_toy/block_dynamics.py` |

### Canonical roster

- **Timoshenko beam pair** (the two coupled PDEs the engine is named for, not yet authored as one symbolic system — the repository has the *kinematics* the pair implies, and the *statically reduced* stiffness form, but not the dynamic PDE itself):
  `rho A d^2w/dt^2 = d/dx[kappa G A (dw/dx - phi)] + q(x,t)`
  `rho I d^2phi/dt^2 = d/dx[EI dphi/dx] + kappa G A (dw/dx - phi)`
  where `kappa` is the shear correction factor, distinguishing this from Euler–Bernoulli (which drops the shear term entirely) — the distinction the repository's own comments already insist on.
- **Constitutive law**: `sigma = E·epsilon` (axial), `tau = G·gamma` (shear) — Hooke's law feeding the strains already computed above.
- **Torsion**: `tau_torsion = G·J·(dtheta_x/dx)`, `T = GJ·twist` for a shaft/beam segment under pure torque.
- **6-DOF frame element stiffness matrix** in closed symbolic form (axial `EA/L`, bending `EI` terms already partially present, shear `kappa GA/L`, torsion `GJ/L`) as one authored symbolic block, rather than only the numeric assembly in `frame_solver.py`.
- **Natural frequency of a general mode**: `omega_n = sqrt(k_n/m_n)` for the n-th mode, generalizing the first-mode oscillator already authored.
- **Rayleigh damping**: `C = alpha·M + beta·K`, the standard way to add modal damping to a multi-mode structural assembly without inventing a new damping law per mode.
- **Yield/failure criteria**: von Mises `sigma_vm = sqrt(sigma_axial^2 + 3 tau^2 - sigma_axial·sigma_bending + ...)` (the repository's `sigma_eq` in `vehicle_mechanical_material.py` is close to this already) and Tresca `max(|sigma_1-sigma_2|, ...) <= sigma_y` as the alternative criterion.
- **Consuming deeper material state** (the pattern §2 of the main proposition names explicitly): `E, G, nu, alpha_thermal, rho` supplied by Gibbs/Bragg rather than declared as beam-local constants — not an equation of Timoshenko's own, but the interface contract the roster above should be written against.

---

## 3. Faraday — electromagnetic field / circuit engine

### Already authored

This is the most complete roster of any engine in the repository —
`turing/examples/symbolic_em_solvers.py` already authors a full
Yee-lattice FDTD pair plus supporting field laws:

| Law | Equation | Source |
|---|---|---|
| Faraday's law (H update) | `H_next = H - dt/mu · curl(E)`; `dt_limit = dx/(c·sqrt(3))`, `c = 1/sqrt(eps·mu)` | `symbolic_em_solvers.py::maxwell_faraday_step` |
| Ampère–Maxwell law (E update, with conduction) | `E_next = (E(1-loss) + dt/eps·(curl(H) - J)) / (1+loss)`, `loss = sigma·dt/(2 eps)`; field energy `u = 1/2(eps E^2 + mu H^2)`; Poynting `S = E x H`; Joule heat `Q = sigma E^2 V dt`; impedance `sqrt(mu/eps)`; dielectric relaxation `tau = eps/sigma` | `symbolic_em_solvers.py::maxwell_ampere_step` |
| Scalar wave equation | `psi_next = 2psi - psi_prev + c^2 dt^2 lap(psi) - gamma dt(psi-psi_prev) + s dt^2` | `symbolic_em_solvers.py::wave_step` |
| Charged species transport (Nernst–Planck + ionization) | drift-diffusion `n_next`; `rho_q = e z n`; `J = e z n·(z mu E)`; conductivity `sigma_q = e^2 z^2 n mu`; plasma frequency `omega_p = sqrt(n e^2 z^2/(eps m))`; Debye length; Townsend ionization coefficient `alpha_T = A p exp(-B p/E)` | `symbolic_em_solvers.py::charged_species_step` |
| Electrostatic relaxation | Jacobi sweep of `lap(phi) = -rho/eps`, `E = -grad(phi)` | `symbolic_em_solvers.py::poisson_relax_step` |
| Material EM constitutive properties (plain Python) | skin depth `delta = 1/sqrt(pi f mu sigma)`; surface resistance `R_s = 1/(sigma delta)`; dielectric loss density `P_v = omega eps0 eps_r tan(delta) |E|^2` | `engine_toy/em_materials.py` |

### Canonical roster

- **Gauss's law**: `div(D) = rho_free` — the Poisson relaxation above is exactly this in electrostatic form; the general time-dependent divergence identity (used as a consistency check on the FDTD stepper, not a separate stepped law) is not yet stated on its own.
- **Gauss's law for magnetism**: `div(B) = 0` — the constraint the Yee-lattice staggering enforces structurally; worth stating explicitly as a bundle-level invariant a Noether-style auditor (§14) could check.
- **Constitutive relations**: `D = eps E`, `B = mu H`, `J = sigma E` (Ohm's law, already used inside `maxwell_ampere_step`'s loss term).
- **Continuity of charge**: `d(rho)/dt + div(J) = 0` — implied by the charged-species step's mass balance but not stated as the general cross-species identity.
- **Lorentz force**: `F = q(E + v x B)` — the force this engine hands to Newton per §3 of the main proposition; not yet authored as its own reusable law (today it lives only inside the full vehicle/world dynamics as an implicit term where charged particles are moved at all).
- **Circuit equations (Kirchhoff)**: current law `sum(I_in) = sum(I_out)` at a node, voltage law `sum(V) = 0` around a loop — the discrete-topology counterpart the main proposition's §3 already names (`d0^T Y d0`, the same DEC Laplacian family as the field solver) but that has not yet been authored as an explicit `LAWS` entry the way the field pair has.
- **Boundary conditions at a material interface**: tangential `E` continuous, normal `D` discontinuous by free surface charge; tangential `H` discontinuous by free surface current, normal `B` continuous — needed once Faraday must model a real dielectric/conductor interface rather than a uniform cell.

---

## 4. Navier–Stokes — continuum fluid transport engine

### Already authored

This is the largest gap in the repository relative to how central the
engine is meant to be. No file surveyed authors the compressible or
incompressible Navier–Stokes system itself as a SymPy law; the fluid
work that exists is either numeric plumbing or belongs to Bjerknes
(§5) or circuit-style hydraulics:

| What exists | Character | Source |
|---|---|---|
| Hydraulic loss / valve-flow / routing models | plain-Python, empirical loss coefficients, not conserved-quantity PDEs | `engine_toy/fluid_circuit_laws.py`, `engine_toy/hydraulic_losses.py`, `engine_toy/port_flow.py`, `engine_toy/fluid_routing.py` |
| Discrete/voxel/columnar fluid engines | numeric solvers/adapters, no `LAWS` table | `turing/src/common/dt_system/fluid_mechanics/{discrete_fluid_engine,voxel_fluid_engine,columnar_multifluid_kernels}.py` |
| Voxel gas-cell mass/energy update (closest existing analogue) | `m_a_next` from advective/conductive flux; `T_next` from an energy balance; ideal-gas `P`, `rho_a`; `dt_limit = min(dx^2/(6 alpha), tau_p)` | `turing/examples/symbolic_chamber_solvers.py::voxel_air_step` — this is Bjerknes-domain (a bulk gas cell), not resolved Navier–Stokes, but it is the nearest existing law and the natural promotion target once a region needs full momentum resolution |

### Canonical roster

The full conserved system, none of which is yet authored anywhere in
the repository as SymPy:

- **Continuity (mass conservation)**: `d(rho)/dt + div(rho u) = 0`.
- **Momentum (Cauchy/Navier–Stokes proper)**: `d(rho u)/dt + div(rho u (x) u) = -grad(p) + div(tau) + rho g`, with the Newtonian stress tensor `tau = mu(grad(u) + grad(u)^T) - (2/3) mu (div u) I`.
- **Energy**: `d(rho E)/dt + div((rho E + p) u) = div(k grad(T)) + div(tau·u) + source`, `E = e + |u|^2/2`.
- **Species transport** (the multi-species extension the proposition's §4 names): `d(rho Y_i)/dt + div(rho u Y_i) = div(rho D_i grad(Y_i)) + omega_dot_i`, coupling directly to Lavoisier's reaction-rate term `omega_dot_i` (§11).
- **Equation of state closure** (asked from Gibbs, per §4's explicit rule that Navier–Stokes owns transport and Gibbs owns thermodynamic truth): ideal gas `p = rho R T` as the simplest closure, with a real equation of state (§6) substituted where compressibility departs from ideal.
- **Incompressible limit**: `div(u) = 0` replacing the continuity equation when Mach number is low, the reduction that makes most engine-bay and duct flows tractable without full compressible resolution.
- **Reynolds number** `Re = rho U L / mu` and **CFL condition** `dt <= dx/(|u|+c)` — the second is already the `cfl` field on `Targets` in `dt_system.dt_controller`, so this engine's own stability contribution already has a home; only the momentum/energy law itself is missing.
- **Boundary-layer reduction (Prandtl)**: `u du/dx + v du/dy = -1/rho dp/dx + nu d^2u/dy^2`, `dp/dy = 0` — the natural reduced model for a duct or aperture wall rather than full 3D resolution, matching §4's "detailed passage through apertures" use case.

---

## 5. Bjerknes — atmosphere / aerosol / dispersed-phase engine

### Already authored

The second-most complete roster after Faraday, split across two
files:

| Law | Equation | Source |
|---|---|---|
| Saturation vapor pressure (Clausius–Clapeyron, integrated form) | `e_s = P_tp exp((L_v/R_v)(T-T_tp)/(T T_tp))` | `turing/examples/symbolic_atmosphere_model.py` |
| Antoine equation | `log10(P) = A - B/(C+T)` | same |
| Relative humidity | `RH = e/e_s` | same |
| Dew point (Magnus–Tetens inversion) | `T_d = c·gamma/(b-gamma)`, `gamma = ln(RH) + bT/(c+T)` | same |
| Köhler curve | `S = exp(A_k/D - B_k/D^3)`, Kelvin term `A_k = 4 M_w sigma_w/(R T rho_w)`, Raoult term `B_k = 6 n_s M_w/(pi rho_w)` | same |
| Hertz–Knudsen surface flux | `J = alpha (P_vapor - P_sat)/sqrt(2 pi M R T)` | same |
| Beer–Lambert extinction | `beta_ext = k·LWC`, `T_trans = exp(-beta_ext d)` | same |
| Mie/Rayleigh size parameter | `x = pi D/lambda` | same |
| Bulk gas-cell mass/energy | `m_a_next`, `T_next`, `P`, `rho_a`, `dt_limit = min(dx^2/(6 alpha), tau_p)` | `turing/examples/symbolic_chamber_solvers.py::voxel_air_step` |
| Vapor/cloud/ice/rain microphysics | `m_v_next, m_l_next, m_i_next, m_r_next`; `S = e/e_s`; `Q_lat = L_v(dm_cond-dm_revap) + L_f dm_freeze`; `LWC, IWC, RWC` | `symbolic_chamber_solvers.py::voxel_species_step` |
| Aerosol evolution/activation/optics | `N_next, M_next, D_mean, v_settle, N_drop, f_act, S_crit, beta_ext, mie_weight, transmittance` | `symbolic_chamber_solvers.py::aerosol_step` |
| Droplet growth/fallout | `r_next, T_p_next, w_next, S_eq, a_w, v_t` (terminal velocity), `dry_fraction` | `symbolic_chamber_solvers.py::droplet_step` |
| Solution thermodynamics | `T_sol_next, Q_reaction, m_w_next, S_c, a_w, T_freeze, crystal_fraction` | `symbolic_chamber_solvers.py::salt_solution_step` |
| Surface film/frost/crust/sediment | `m_film_next, h_frost, m_crust_next, T_s_next, J, runoff, U_eff` | `symbolic_chamber_solvers.py::surface_step` |
| Sessile pool | `m_p_next, n_s_next, T_l_next, A_wet, h_pool, overflow_rate` | `symbolic_chamber_solvers.py::pool_step` |

### Canonical roster

- **The Bjerknes circulation theorem itself** — the namesake law, not yet authored anywhere: `dGamma/dt = -∮(1/rho) dp` (the baroclinic circulation term, vanishing when density depends on pressure alone). This is the theorem that makes weather baroclinic rather than barotropic, and it is the one piece of "why this name" the repository has not yet written down even though it has written nearly everything downstream of it.
- **Hydrostatic balance**: `dp/dz = -rho g` — the vertical structure a bulk atmosphere column needs before any of the microphysics above can be placed at the right altitude/pressure.
- **Ideal/moist gas law for air**: `p = rho R_specific T`, with `R_specific` a mixture of dry-air and water-vapor gas constants — the closure the bulk cell (`voxel_air_step`) already assumes implicitly via `P`/`rho_a` but has not stated as its own reusable identity.
- **Potential temperature**: `theta = T (p0/p)^(R/cp)`, the quantity that is conserved for dry adiabatic vertical motion and is the natural state variable once vertical convection (not just one fixed-altitude cell) is modeled.
- **Geostrophic balance**: `f × u = -(1/rho) grad(p)`, Coriolis parameter `f = 2 Omega sin(latitude)` — needed once large-scale horizontal wind (not just a bounded chamber) enters the picture; this is the equation the Bjerknes circulation theorem is usually taught alongside.
- **Turbulent/eddy diffusion (K-theory) for boundary-layer mixing**: `d(phi)/dt = d/dz(K_z dphi/dz)` — the reduced closure for vertical mixing of heat/moisture/aerosol without resolving turbulence explicitly (i.e. the Bjerknes-level alternative to promoting the region to Navier–Stokes).
- **Terminal velocity (Stokes' law)** for small droplets/aerosol: `v_t = 2 r^2 (rho_p - rho_a) g / (9 mu)` — the repository's `droplet_step`/`aerosol_step` already compute a `v_settle`/`v_t`; stating the underlying Stokes closure explicitly documents which flow regime it assumes (valid only at low particle Reynolds number).

---

## 6. Gibbs — thermodynamics / phase-state engine

### Already authored

Entirely plain-Python today; real thermodynamic content, not yet
compiled through the SymPy pipeline:

| Law | Equation | Source |
|---|---|---|
| Phase transition table (reversible pairing) | `latent_rev = -latent_fwd`, `dV_rev = -dV_fwd`; water: `T_freeze=273.15K, L_freeze=-333550 J/kg`, `T_boil=373.15K, L_vap=2256000 J/kg` | `engine_toy/phase_table.py` |
| Cryogen boil-off cold recovery | `E_sensible = c_p (277 - T_boil)`; vent-worth fraction `f = E_sensible/(L+E_sensible)`; expansion ratio `rho_liquid/rho_gas`; JT throttling-cools test `T_in < T_inversion` | `engine_toy/cryogenics.py` |
| Joule–Thomson throttling temperature | `T_out = T_in - mu_JT · dP_bar` | `engine_toy/thermal_emission.py` |
| Choked-orifice static temperature | `T_throat = T_in · 2/(gamma+1)` | `engine_toy/thermal_emission.py` |
| Isentropic blowdown with efficiency | `T_ideal = T0 · r^((gamma-1)/gamma)`; `T_real = T_ideal + (T0-T_ideal)(1-eta)` | `engine_toy/thermal_emission.py` |
| Ideal cutoff-expansion indicator law | `MEP = p_s c (1 + (1-c^(n-1))/(n-1)) - p_b` (and the `n->1` logarithmic limit) | `engine_toy/expander.py` |
| Refrigerant saturation curve | `ln(p) = A - B/T`, inverted `T_sat(p) = B/(A - ln p)` | `engine_toy/refrigeration.py` |
| Water activity / freezing-point depression / crystallization | `a_w`, `T_freeze`, `crystal_fraction` (a Raoult/colligative-law instance) | `turing/examples/symbolic_chamber_solvers.py::salt_solution_step` |

### Canonical roster

- **Gibbs free energy**: `G = H - T S`; **Helmholtz free energy**: `A = U - T S` — the two potentials everything else in this engine (equilibrium, phase selection, chemical potential) should ultimately be derived from, rather than each phenomenon carrying its own private fitted curve as most of the entries above currently do.
- **Chemical potential and equilibrium condition**: `mu_i = (dG/dn_i)_{T,p,n_j}`; equilibrium: `sum_i mu_i dn_i = 0` for any admissible reaction/phase-transfer direction.
- **Gibbs–Duhem equation**: `S dT - V dp + sum_i n_i dmu_i = 0` — the identity that keeps a multi-component phase's independently-fitted properties (already the practice in `phase_table.py`/`refrigeration.py`) thermodynamically consistent with each other rather than independently curve-fit.
- **Gibbs phase rule**: `F = C - P + 2` (degrees of freedom = components - phases + 2) — the bookkeeping identity that tells a bundle author how many state variables a given multi-phase mixture actually needs.
- **Clausius–Clapeyron, differential form**: `dp/dT = L/(T dv)` — the repository already uses the *integrated* form (§5's `e_s`), but the differential form is the one that generalizes cleanly to any two-phase boundary, not just vapor pressure.
- **Equation of state beyond ideal gas**: van der Waals `(p + a n^2/V^2)(V-nb) = nRT` or Peng–Robinson, for the cryogenic/high-pressure regimes `cryogenics.py` and `refrigeration.py` already operate in but currently handle with fitted curves rather than a real equation of state.
- **Raoult's law / activity coefficients**: `p_i = x_i gamma_i p_i^sat` — the general form of what `salt_solution_step`'s water activity `a_w` already computes for one specific case.
- **Entropy production (second law)**: `dS_universe/dt >= 0`, `dS_universe = dS_system + dS_surroundings` — the constraint a Noether-style auditor (§14) should check against every Gibbs-owned transition, since none of the fitted curves above currently assert it.

---

## 7. Bragg — solid-state / lattice / semiconductor engine

### Already authored

No file surveyed authors solid-state, crystallographic, or
semiconductor physics as such. The two adjacent things that exist are
electromagnetic material constants (arguably Bragg-owned constitutive
data, currently filed as plain EM material properties) and mechanical
damage/wear bookkeeping that Bragg would eventually own the
microstructural side of:

| What exists | Character | Source |
|---|---|---|
| Skin depth, surface resistance, dielectric loss density | plain-Python constitutive formulas for a conductor/dielectric | `engine_toy/em_materials.py` |
| Wear-debris mass ledger | plain-Python conservation bookkeeping (`entered = captured + suspended + deposited + drained`), not a microstructural damage law | `engine_toy/wear_debris.py` |
| Member plastic/damage state (shared with Newton/Timoshenko) | accumulated plastic strain, `failed` flag, dissipated energy | `turing/src/compiler/vehicle_mechanical_material.py` |

### Canonical roster

Since nothing exists yet, this is the field's textbook roster in full,
organized by the state Bragg is meant to own (§7 of the main
proposition):

- **Bragg's law itself** (diffraction condition, the namesake identity): `n lambda = 2 d sin(theta)` — crystal-lattice spacing `d` from a diffraction angle; the literal reason the engine carries this name.
- **Generalized Hooke's law (anisotropic elasticity)**: `sigma_ij = C_ijkl epsilon_kl`, with `C_ijkl` the elastic tensor set by crystal symmetry — the tensor Timoshenko's `E, G, nu` (§2) are an isotropic reduction of.
- **Dislocation/plasticity kinetics**: Taylor hardening `tau = tau_0 + alpha G b sqrt(rho_dislocation)`; Orowan's equation for plastic shear rate `gamma_dot = rho_mobile b v_dislocation` — the microscopic origin of the phenomenological hardening `vehicle_mechanical_material.py` already fits macroscopically.
- **Griffith fracture criterion**: `sigma_f = sqrt(2 E gamma_s / (pi a))` — crack-length-dependent failure stress, the natural refinement of the existing binary `failed` flag.
- **Point-defect diffusion (Fick/Arrhenius)**: `D = D_0 exp(-Q/(R T))`, `dC/dt = D lap(C)` — vacancy/interstitial/dopant migration.
- **Semiconductor drift-diffusion**: carrier continuity `dn/dt = (1/q) div(J_n) + G - R`, `dp/dt = -(1/q) div(J_p) + G - R`; drift-diffusion current `J_n = q n mu_n E + q D_n grad(n)` (and the hole analogue); Poisson's equation coupling carrier density to potential (shared structurally with Faraday's `poisson_relax_step`, but sourced by carrier charge rather than free charge alone).
- **Shockley diode equation**: `I = I_0 (exp(qV/(nkT)) - 1)` — the reduced device-level law once full drift-diffusion is not needed.
- **Band-gap temperature dependence (Varshni equation)**: `E_g(T) = E_g(0) - alpha T^2/(T+beta)`.
- **Phonon thermal conductivity (kinetic theory)**: `k = (1/3) C_v v_sound l_mean_free_path` — the lattice-vibration route to a thermal conductivity Fourier (§12) would otherwise treat as a bare tabulated constant.
- **Curie–Weiss law** (magnetic susceptibility near a ferromagnetic transition): `chi = C/(T - T_c)` — flagged per the naming note above: this is Bragg-domain solid-state physics (a lattice/magnetic phase property) despite sharing a name with the Curie nuclear engine (§9).
- **Landau–Lifshitz–Gilbert equation** (magnetization dynamics): `dM/dt = -gamma M x H_eff + (alpha/M_s) M x dM/dt` — for any ferromagnetic/magnetic-domain material state.

---

## 8. Hamilton — quantum-state / electronic-structure engine

### Already authored

| Law | Equation | Source |
|---|---|---|
| Staggered-leapfrog Schrödinger evolution | `R_next = R + dt/hbar · H(I)`; `I_next = I - dt/hbar · H(R_next)`; probability density `R^2+I^2`; probability current; `dt_limit = hbar/(E_max+eps)` | `turing/examples/symbolic_em_solvers.py::schrodinger_step` — authored today inside the Faraday-named field-solvers file, but the law itself is Hamilton-domain and should be recognized as such when a bundle is assembled, per the domain-boundary note above. |

### Canonical roster

- **Time-dependent Schrödinger equation**, the general form the staggered scheme above discretizes: `i hbar dpsi/dt = H psi`, `H = -hbar^2/(2m) lap + V(x)`.
- **Time-independent Schrödinger equation / eigenvalue problem**: `H psi_n = E_n psi_n` — needed for equilibrium electronic-structure questions rather than time evolution.
- **Tight-binding/Hückel Hamiltonian**: `H_ij = alpha` (on-site) or `beta` (hopping between bonded sites), `0` otherwise — the cheapest lane named in §8 of the main proposition, well short of full electronic structure.
- **Born–Oppenheimer force on nuclei**: `F_i = -d(E_e(R))/dR_i` — the literal Hamilton→Newton coupling §8 describes for molecular geometry; not yet authored anywhere, since no electronic energy surface `E_e(R)` exists in the repository yet for any species.
- **Ehrenfest theorem** (classical-limit consistency check): `d<x>/dt = <p>/m`, `d<p>/dt = -<dV/dx>` — a useful validation identity per §14's "independent reference calculation" requirement, checking a quantum simulation against its classical limit.
- **Variational principle**: `E_0 <= <psi|H|psi>/<psi|psi>` for any trial wavefunction — the basis of Hartree–Fock/DFT approximations named in §8 as deeper fidelity lanes.
- **Hartree–Fock self-consistent field equations** and **Kohn–Sham equations** (density-functional theory) as the two named "increasingly sophisticated electronic structure" lanes in §8, both still unauthored.
- **Perturbation theory, first order**: `E_n^(1) = <psi_n^(0)|H'|psi_n^(0)>` — for small deviations from a solved reference Hamiltonian, the cheapest way to approximate a nearby quantum state without a full re-solve.

---

## 9. Curie — nuclear / radiological engine

### Already authored

Nothing. No isotope, decay, or nuclear-reaction law was found anywhere
in either repository.

### Canonical roster

- **Radioactive decay law**: `N(t) = N_0 exp(-lambda t)`, activity `A = lambda N`, half-life `lambda = ln(2)/t_half` — the minimal law this engine cannot exist without.
- **Bateman equations** for a decay chain: `dN_i/dt = lambda_{i-1} N_{i-1} - lambda_i N_i` for each generation `i`, with the closed-form solution for a chain of arbitrary length — needed the moment a decay product is itself radioactive rather than stable.
- **Branching decay**: `lambda_total = sum_k lambda_k`, branching ratio `b_k = lambda_k/lambda_total` for competing decay modes.
- **Decay heat power**: `P = sum_i lambda_i N_i E_i` — the source term this engine hands to Fourier (§12) as deposited heat, per the main proposition's "Curie produces particles/radiation/energy which other engines consume."
- **Reaction rate (activation/transmutation)**: `R = N sigma Phi` (number density × cross-section × neutron/particle flux) — needed only where the main proposition's "eventually justified" activation case actually arises.
- **Semi-empirical mass formula (binding energy)**: `B(A,Z) = a_V A - a_S A^{2/3} - a_C Z(Z-1)/A^{1/3} - a_A (A-2Z)^2/A + delta(A,Z)` — the identity that predicts which nuclides are stable and what a reaction's Q-value is.
- **Q-value of a nuclear reaction/decay**: `Q = (m_initial - m_final) c^2` — directly consumes Einstein's mass–energy equivalence (§10), the one place these two engines are expected to meet.
- **Radiation dose/attenuation**: `I(x) = I_0 exp(-mu x)` (linear attenuation) and inverse-square falloff `I ~ 1/r^2` from a point source — for any downstream dose or shielding question.

---

## 10. Einstein — relativistic spacetime engine

### Already authored

Nothing found in either repository is relativistic; `turing/src/
common/tensors/riemann/geodesic.py` exists but implements a
"geodesic convolution" neural-network layer, an unrelated use of the
word "geodesic" from differential geometry applied to graph/manifold
learning, not a physical spacetime geodesic — flagged here so it is
not mistaken for existing relativity content.

### Canonical roster

- **Lorentz factor**: `gamma = 1/sqrt(1 - v^2/c^2)`.
- **Relativistic momentum and energy**: `p = gamma m v`; `E = gamma m c^2`; the invariant relation `E^2 = (pc)^2 + (mc^2)^2` — the identity a relativistic particle's kinematics should be checked against.
- **Mass–energy equivalence**: `E = m c^2` — the specific case Curie's Q-value (§9) already needs.
- **Lorentz transformation** (boost along x): `t' = gamma(t - vx/c^2)`, `x' = gamma(x - vt)` — for any two-frame comparison.
- **Proper time**: `dtau = dt/gamma` — the invariant clock a relativistic particle actually experiences.
- **Relativistic force law**: `F = d(gamma m v)/dt`, which does not reduce to `F = ma` except at low `v/c` — the one place Newton's second law (§1) genuinely stops applying and this engine must take over.
- **Relativistic Doppler shift**: `f_obs = f_src sqrt((1-beta)/(1+beta))` (receding case) — for any electromagnetic signal (Faraday, §3) emitted by a fast-moving source.
- **Einstein field equations** (the frontier case, named for completeness per §10's "eventually curved spacetime if there is ever a reason to model it"): `G_mu_nu = (8 pi G/c^4) T_mu_nu` — not expected to be needed soon, but the equation this engine's name ultimately refers to.

---

## 11. Lavoisier — chemistry / reaction & species engine

### Already authored

| Law | Equation | Source |
|---|---|---|
| Molar mass from composition | `M = sum(atom_count_i · atomic_weight_i)` | `engine_toy/organic_species.py` |
| Charge bookkeeping | explicit per-species charge property, `q=0` for combustion products | `engine_toy/organic_species.py` |
| Soot yield | `formed = anchor_frac · (prop/anchor_index)`; `oxidised = min(0.95, 0.85 max(0,1-phi))`; `soot = formed(1-oxidised) + formed·unburnt` | `engine_toy/combustion_products.py` |
| Boundary species/phase routing | species -> canonical chemical state lookup at the engine/chemistry interface | `engine_toy/chemistry_boundary.py` |
| **Construction-time conservation identities** (the load-bearing existing design, not yet wired to a live reaction network) | element conservation `A @ nu[r] = 0` (A = elemental composition matrix, nu = signed stoichiometry); charge conservation `z @ nu[r] = 0` | `turing/docs/AUDIT_MULTISPECIES_INORGANIC_CHEMISTRY.md` (design audit of the intended compendium; the identities are stated as *construction-time* checks a reaction row must pass before reaching SymPy) |

### Canonical roster

- **Law of mass action (reaction rate)**: `r = k · prod_i [C_i]^{n_i}` — the kinetic law the audit document's stoichiometric rows are meant to feed, not yet wired to an actual rate expression anywhere.
- **Arrhenius rate constant**: `k = A exp(-E_a/(R T))` — the temperature dependence every kinetic rate above needs, consuming Gibbs's `T` (§6).
- **Equilibrium constant from Gibbs energy**: `K_eq = exp(-Delta G_rxn / (R T))` — the literal Lavoisier–Gibbs coupling §11 of the main proposition already names ("Chemistry couples to Gibbs for thermodynamic driving state").
- **Species mass-balance ODE**: `dC_i/dt = sum_r nu_{i,r} · rate_r` — the equation the construction-time identities above are the *conservation constraint on*, not yet the ODE itself.
- **Elemental and charge conservation, restated as the running invariant** (not just a construction-time check): `sum_i (A_{e,i} C_i) = const` for every element `e`; `sum_i (z_i C_i) = const` — the live-simulation version of the audit document's `A @ nu = 0`/`z @ nu = 0`, i.e. what a Noether-style auditor (§14) should verify every step, not only at reaction-row authoring time.
- **Faraday's laws of electrolysis** (flagged per the naming note above — chemistry, not the Faraday field engine): `m = (Q/F) · (M/z)`, with `F` the Faraday constant (charge per mole of electrons) — needed the moment an electrochemical process (battery, electrolysis, corrosion) enters the chemistry roster.
- **Combustion stoichiometry**: balanced fuel + oxidizer -> products from elemental conservation alone, with equivalence ratio `phi = (fuel/air)_actual / (fuel/air)_stoichiometric` — `combustion_products.py`'s soot law already consumes a `phi` computed this way but the stoichiometric balance itself is not separately authored.

---

## 12. Fourier — thermal conduction / heat-transfer engine

### Already authored

| Law | Equation | Source |
|---|---|---|
| Combined sensible + latent storage | `E/kg = c_p |dT| + L_latent` (a Fourier term and a Gibbs term in one formula, matching §12's "thermal is a shared consumer" pattern exactly) | `engine_toy/thermal_storage.py` |
| Radiative sky cooling | `Qdot = A · q_clear · (1 - cloud_frac)` | `engine_toy/thermal_storage.py` |
| Flame quench distance | `delta_q = Pe · alpha / S_L` | `engine_toy/combustion_efficiency.py` |
| Combustion completeness (crevice/quench losses) | `efficiency = oxygen_limit · (1-crevice_loss) · (1-quench_loss)` | `engine_toy/combustion_efficiency.py` |
| Blackbody thermal emission (Planck's law, GPU-resident) | `blackbody_emission_rgb` samples Planck's law per kelvin value stored in a `ThermalChunk` SSBO, evaluated per-band in the render shader | `engine_toy/engine_mesh.py` (`blackbody_emission_rgb`); see `ENGINE_TOY_ARCHITECTURE_NOTES.md`'s "kelvins in an SSBO, Planck in the shader" section |
| Joule–Thomson / choked-flow / blowdown temperatures (shared with Gibbs, §6) | see §6 above | `engine_toy/thermal_emission.py` |
| Surface drying (heat-limited evaporation) | `mdot = h A (T_surface - T_boil)/L` | `engine_toy/surface_wetting.py` |

### Canonical roster

- **Fourier's law of conduction itself** (the namesake law, not yet authored as its own explicit statement even though the repository's thermal mesh clearly implements its discretization): `q = -k grad(T)`.
- **The heat equation**: `dT/dt = alpha lap(T) + source/(rho c_p)`, `alpha = k/(rho c_p)` — the PDE `q=-k grad(T)` implies once combined with energy conservation; the main proposition's §12 already describes the mesh solving "the heat equation" with resolvable `cp(state), k(state), rho(state)` — this is that equation, stated explicitly.
- **Newton's law of cooling (convection)**: `q = h A (T_surface - T_fluid)` — the convective counterpart already used implicitly in the surface-drying law above but not stated as its own general boundary condition.
- **Stefan–Boltzmann radiation**: `q = epsilon sigma T^4` — the integrated-over-all-wavelengths counterpart to the Planck's-law emission already computed per-band on the GPU; useful as the cheap 0-D radiative-loss estimate when a full spectral render is not needed.
- **View-factor radiative exchange** between two surfaces: `Q_{1->2} = sigma A_1 F_{1->2} (T_1^4 - T_2^4)` — needed once two hot bodies radiatively couple rather than one body radiating to a fixed sky.
- **Biot number**: `Bi = h L_c / k` — the criterion that decides whether a lumped-capacitance model (`dT/dt = -hA(T-T_inf)/(rho V c_p)`) is valid or whether spatial gradients inside the body must be resolved.
- **Heat-exchanger effectiveness–NTU method**: `epsilon = f(NTU, C_r)`, `NTU = UA/C_min` — for any counterflow/crossflow heat exchanger (the powertrain coolant loop the repository's `turret_production.py` comment already references as "the fluid engine's job").

---

## 13. Boltzmann — statistical mechanics / kinetic particle-distribution engine (reserved)

### Already authored

Nothing. No distribution-function or kinetic-transport law was found.

### Canonical roster

- **The Boltzmann transport equation itself** (the namesake law): `df/dt + v·grad_x(f) + (F/m)·grad_v(f) = (df/dt)_collision` — the general evolution of a particle distribution function `f(x,v,t)` under external force and collisions.
- **Maxwell–Boltzmann velocity distribution** (the equilibrium solution): `f(v) = n (m/(2 pi k T))^{3/2} exp(-m v^2/(2kT))`.
- **BGK collision operator** (the simplest closure making the transport equation tractable): `(df/dt)_collision = -(f - f_eq)/tau`, relaxation toward local Maxwell–Boltzmann equilibrium over a collision time `tau`.
- **Chapman–Enskog expansion**: the formal bridge from the Boltzmann equation to the Navier–Stokes equations (§4) in the small-Knudsen-number limit — the mathematical justification for exactly the promotion path (Boltzmann ⟷ Navier–Stokes) the main proposition's reserved-name rationale describes.
- **H-theorem**: `dH/dt <= 0`, `H = integral(f ln f)` — the microscopic origin of the second law's entropy-production requirement Gibbs (§6) states macroscopically.
- **Mean free path and Knudsen number**: `lambda = 1/(sqrt(2) n sigma)`, `Kn = lambda/L` — the regime-selection criterion that would decide, for a given region, whether Boltzmann, Navier–Stokes, or free-molecular flow is the right description; this is the practical "promote/reduce" trigger §13 of the main proposition asks every engine to expose.

---

## 14. Noether — conservation / invariant / physical-ledger verification engine (reserved)

### Already authored

Not as a unified cross-engine auditor, but its practical raw material
already exists piecemeal, per-engine, inside the dt-system contract
itself:

| What exists | Character | Source |
|---|---|---|
| Per-step error/invariant channels every compiled law already publishes | `max_vel, max_flux, div_inf, mass_err, dt_limit, energy_j, power_w` | `turing/src/common/dt_system/dt_scaler.py` (`Metrics`), consumed by every law in `symbolic_em_solvers.py`/`symbolic_atmosphere_model.py`/`symbolic_chamber_solvers.py` |
| Construction-time element/charge conservation check | `A @ nu = 0`, `z @ nu = 0` (Lavoisier, §11) | `turing/docs/AUDIT_MULTISPECIES_INORGANIC_CHEMISTRY.md` |
| Gauss's-law-consistent field staggering | `div(B)=0` enforced structurally by the Yee lattice (Faraday, §3), not asserted as a runtime check | `symbolic_em_solvers.py` |

None of these are unified under one cross-engine identity or law
table; each engine currently reports its own conservation honestly, but
nothing compares one engine's ledger against another's, or against the
symmetry that implies it.

### Canonical roster

- **Noether's theorem itself** (the namesake law, a meta-law rather than a per-system equation): every continuous symmetry of a system's action implies a conserved quantity —
  - time-translation invariance → **energy conservation**;
  - spatial-translation invariance → **linear momentum conservation**;
  - rotational invariance → **angular momentum conservation**;
  - gauge invariance (of the electromagnetic potential) → **electric charge conservation**.
- **The practical ledger this engine should assert**, in the vocabulary the repository already has a home for: `mass_err`, `div_inf`, and the `energy_j`/`power_w` pair are already *reported* per engine (`dt_scaler.Metrics`); Noether's job would be to *compare* — e.g. total system energy before/after a step across every active engine simultaneously, not merely each engine's own `energy_j` in isolation — which is a genuinely new function, not a relabeling of what already exists.
- **Discrete conservation identities per engine**, restated as what Noether would check rather than what each engine already self-reports:
  - Newton (§1): `sum(p_i) = const`, `sum(L_i) = const` for an isolated system;
  - Faraday (§3): `div(B) = 0` and total charge `integral(rho) dV = const`;
  - Lavoisier (§11): `sum_i(A_{e,i} C_i) = const` per element, `sum_i(z_i C_i) = const`;
  - Navier–Stokes (§4)/Bjerknes (§5): total mass, momentum, and energy across a closed control volume.
- **A cross-engine energy ledger**, the one identity no single engine's `energy_j`/`power_w` channel can express alone: `d(E_total)/dt = sum_k P_k^{in} - sum_k P_k^{out}` summed over every engine active in one coupled volume (the controlled chamber, §20 of the main proposition) — the practical form of "one physical fact, one authoritative state, many consumers" applied to energy itself.
