# Honorary equation set: coverage audit of the established systems

2026-09-22. Every established physics system in `engine_toy/` and `turing/`
(law modules, chemistry, DEC and cavity, dt_system physics, cells) was checked
against `engine_toy/honorary_engine_equation_catalogue.py` (612 equations,
22 engines). The check was read-only: docstrings plus the formulas in the
code, and each "missing" name was confirmed absent from the catalogue `.py`
and `.md` by text search. What follows is only what is NOT yet represented,
filed by the engine it belongs to. A system that is fully covered is not
listed (for example Maxwell/Yee, Schrödinger, Poisson, Noble–Abel/Vieille,
the Timoshenko members, modal and ringdown analysis, mass-action CRN and
membrane capacitance).

Each entry is **system** (where it lives): the laws it uses that the
catalogue lacks.

---

## Existing engines

### Otto (O): cyclic engines (~25, the largest gap)
- **Ideal cycles** (`thermo_cycles`, `derived_torque`, `engines`, `gas_turbine`): Otto η=1−r^(1−γ); Diesel η with cutoff ratio; Brayton η=1−r_p^((1−γ)/γ) and its regenerated form; Carnot bound.
- **Torque from MEP** (`engines`, `torque_curve`, `engine_sim`, `starter`, turing `abstract_ui_vehicles` powertrain): IMEP=VE·ρ·(LHV/AFR)·η; T=BMEP·V_d/(4π); Chen–Flynn FMEP=A+B·S_p+C·S_p²; mean piston speed 2Sn.
- **Breathing and tuning** (`engines`, `port_flow`, `throttle_body`, `valve_control`): Helmholtz f=(c/2π)√(A/(LV)); quarter-wave f=c/4L; port Mach index (Taylor Z); butterfly area fraction; effective (Miller/Atkinson) compression ratio.
- **Combustion timing** (`engine_cycle_sim`, `combustion_efficiency`, `cylinder_deactivation`): manifold filling/emptying lag; turbulent flame speed S_T=S_L(1+k·rpm) and MBT timing; Livengood–Wu knock integral; crevice fraction; Peclet quench distance; pumping-loss recovery.
- **Scavenging** (`two_stroke`, `derived_torque`): Hopkinson perfect-displacement and perfect-mixing trapping bounds; charging efficiency.
- **Positive-displacement machines** (`compressors`, `expander`, turing accessories): clearance volumetric efficiency; polytropic compressor work and discharge T; cutoff-expander MEP (isothermal and polytropic).

### Navier–Stokes (NS): flow, hydraulics, free surface (~22)
- **Shallow water** (turing `symbolic_fluid_model` "Shoal"): Saint-Venant equations with viscosity, drag and Coriolis; gravity wave speed √(gh).
- **Orifices and leaks** (`fluid_circuit_laws`, `hole_emitters`, `fittings`): SUBCRITICAL compressible orifice (Saint-Venant–Wanzel); Torricelli; Tate's drop law; Weber/Ohnesorge breakup map.
- **Turbomachinery similarity** (`hydraulics`, `station_cooling`, `dyno_brakes`, `couplings`, `air_movers`): affinity laws (Q∝n, P∝n³); water brake T=Kρ D⁵ω²; Föttinger coupling T=λρD⁵n²; actuator-disk momentum theory (T=ṁv_i, P=½ρA v_i³).
- **Separation** (`centrifuges`): Sigma theory (disc stack, tubular); Stokes cut size; Lapple cyclone d50.
- **Hydraulic lines** (`hydraulic_losses`, `actuators`): quadratic valve drop; seal breakaway vs running friction; hydraulic natural frequency ω_n=√(4βA²/(Vm)); parallel-plate slot flow; Bingham plastic (MR damper).
- **Fluid properties and particle fluids** (`hydraulics`, turing `bath/*`): Walther/ASTM D341 viscosity–temperature; Tait EOS; Boussinesq buoyancy; pressure-projection Poisson; CSF surface tension.
- **Films and pools** (turing `surface_step`/`pool_step`, `surface_wetting`): capillary length √(σ/ρg); sessile height 2l_c·sin(θ/2); Nusselt film velocity; sharp-crested weir Q=C·w·√(2g)·H^1.5.
- **Ventilation** (`air_volumes`): well-mixed CSTR dC/dt=(Q/V)(C_in−C)+S/V; air changes per hour.

### Bjerknes (B): atmosphere and cloud (~14)
- **Bulk microphysics** (turing `voxel_species_step`): Kessler autoconversion, accretion (q_r^0.875), rain re-evaporation.
- **Aerosol** (turing `aerosol_step`): Cunningham slip; Brownian coagulation kernel; Stokes–Einstein diffusivity; κ-Köhler critical supersaturation; washout.
- **Drops** (turing `droplet_step`, `symbolic_atmosphere`): Mason growth law; Newton-regime terminal velocity and the Stokes–Newton blend; Nu=2 sphere conduction.
- **Psychrometrics** (`air_treatment`, turing `symbolic_atmosphere_model`): Magnus–Tetens; humidity ratio w=0.622·p_v/(p−p_v); Chilton–Colburn kinetic+diffusive evaporation resistance.

### Gibbs (G): thermodynamic processes and solutions (~18)
- **Process relations** (turing balloon-tire gas, `symbolic_parts`, `oxygen_service`, `thermal_emission`, `gas_harvest`, `cryogenics`): polytropic pV^n=const and T·V^(n−1)=const; isentropic compression/expansion with efficiency; isothermal work mRT·ln(p/p₀) (single and multistage); turboexpander outlet T.
- **Refrigeration** (`refrigeration`): Carnot COP times second-law efficiency.
- **Mixtures** (`autoclave`, `cryogenics`): Dalton's law; Raoult's law; JT inversion criterion; CO₂ frost-point curve.
- **Solutions and colligative properties** (turing `salt_solution_step`, `thermal_storage`): ΔT_f=K_f·i·b; ΔT_b=K_b·i·b; Raoult in molality form; linear solubility in T; crystal growth, nucleation, dissolution.
- **Distillation** (`air_separation`): Fenske, Underwood, Gilliland.
- **Casting** (`foundry`, `moulding`): Chvorinov t=C(V/A)^n; compounding shrinkage.

### Lavoisier (L): chemistry (~9)
- **Equilibrium in T** (turing `compendium` EquilibriumLaw): van't Hoff ln K(T).
- **Activity** (turing `compendium` owner system): ionic strength I=½Σm z²; Davies activity coefficient; molality/molarity definitions; Kohlrausch σ=Λ_m·c.
- **Combustion bookkeeping** (`organic_species`, `combustion_products`, `otto_langen`, `spectacle`): O₂ demand x+y/4−z/2+w and stoichiometric AFR; CₓHᵧSᵤ product balance; flammability window LFL–UFL and φ; rich unburnt fraction 1−1/φ.
- **Deposition and ageing** (`fouling`, `hydraulics`, `crankcase_state`, `automatic_transmission`): Arrhenius-rule ageing ("life halves per 10 K"); threshold deposition rates; film deposit/scrape/burn balance; fuel-dilution boil-off.
- **Emissions** (`emissions`): Heywood φ-trends; catalyst light-off sigmoid and λ-window; CO→COHb uptake (Stewart), which is physiology.

### Faraday (F): electromagnetics and electrical machines (~25)
- **Transformers** (`transformers`): EMF V=4.44fNAB; leakage reactance and short-circuit current; regulation; Steinmetz core loss; copper loss.
- **Rotating machines** (`starter`, `dyno_brakes`, turing accessories, turing `VoiceCoil`): series-DC motor V=IR+KIω, T=KI²; eddy-current brake torque curve; voice-coil force F=Bl·i and Thiele–Small parameters.
- **Conductors and components** (`dc_power`, `traces`, `knees`, `rectifiers`, `battery_bank`, `electrical_network`): Pouillet R=ρL/A with temperature coefficient; Grover partial inductance; IPC-2221 ampacity; Shockley diode; voltage doubler; battery OCV(SOC), internal resistance and coulomb counting.
- **Guided waves and apertures** (`waveguides`): circular-guide Bessel cutoffs; rectangular TE/TM cutoffs; TE/TM wave impedance; Bethe small-hole polarizabilities and cross-section.
- **Cavities and radiators** (`cavities`, `emitters`, `chambers`, `applicators`, turing `dec_cavity`): driven curl-curl source form; complex-ε Q loss; modal expansion; Weyl mode count; wall-loss Q; modal LC equivalents; small-loop radiation resistance; available power; shielding effectiveness; cavity-perturbation frequency pull.
- **Waves and plasma** (turing `symbolic_em_solvers`): damped wave equation and its energy density; Townsend ionization; plasma frequency; Debye length; conductivity; two-body recombination.

### Fourier (FO): heat transfer (~10)
- **Convection** (`barrel_thermal`, `interior_ballistics`): Dittus–Boelter Nu=0.023Re^0.8Pr^0.4, laminar Nu=4.36.
- **Phase change and cryogenics** (`autoclave`, `cryogenics`): condensation heat ṁ·h_fg; boil-off ṁ=Q_leak/h_fg; MLI effective conductivity.
- **Conduction shapes** (`thermal_storage`, turing `surface_step`): buried-sphere shape factor 4πkrΔT; series resistance 1/U=Σ.
- **Solar** (`solar`): Kasten–Young air mass; Meinel irradiance; incidence cosine; isotropic-sky view factor; PV temperature derate.

### Newton (N): contact, friction, drivetrain dynamics (~15)
- **Tire and friction** (turing wheel contact, `engine_harm`): Stribeck μ(v); bristle/brush saturation F_lim·tanh(F/F_lim); load sensitivity; sidewall relaxation filter.
- **Regularized drivetrain** (`drivetrain_port`, `couplings`, turing powertrain): tanh-regularized clutch torque; clutch capacity T=μpA·N·r_m; slip heat T·Δω; backlash dead band; gear ratio and efficiency; reflected inertia J_eq=ΣJ(ω_i/ω_in)²; rolling resistance.
- **Mass properties** (`engine_mass_properties`, `rotating_inertia`, `operating_states`): I_ij=Σm(r²δ−r_i r_j); I=k·m·r²; ISO 21940 unbalance U=m·e.
- **Game-physics constraints** (turing dt_system engines and softbody): penalty contact k·pen−b·v_n; tension-only rope; asymmetric damper; XPBD compliant multiplier; linear (Stokes) drag.
- **Governor** (`governor`): centrifugal flyball balance against spring preload.

### Timoshenko (T): sections, joints, vibration (~15)
- **Section properties** (`milspec`, `surfaces`, `joints`): I=∫y²dA and parallel-axis theorem; radius of gyration; St-Venant torsion constant for a rectangle; Bredt–Batho closed section; tube section modulus; Cowper shear coefficients.
- **Stress and deflection** (`ring_joints`, `muzzle_reference`, `barrel_thermal`): σ=N/A+My/I; cantilever droop wL⁴/8EI; thermal-bow curvature κ=αΔT/d.
- **Vibration** (`machine_shake`, `wrench_paths`, `engine_mounts`, `component_mode_atlas`): harmonic response amplitude; unbalance force Uω²; base-excitation transmissibility; gyroscopic mode splitting; Craig–Bampton reduction.
- **Contact stiffness** (`feet`): Boussinesq rigid punch k=2aE/(1−ν²); rocking frequency.
- **Springs** (turing suspension): helical spring rate k=Gd⁴/(8D³n); motion ratio k_w=k·MR².

### Bragg (BR): material constitutive laws (~7)
- **Membranes and laminates** (turing `vehicle_balloon_tire`): St Venant–Kirchhoff energy with Green–Lagrange strain; orthotropic Q laminate; Rayleigh dissipation function.
- **Plasticity and fracture** (turing `vehicle_mechanical_material`): linear isotropic hardening; reduced von Mises form; ductility-exhaustion fracture; viscous strain-rate dissipation.

### Zeldovich (Z): blast (~8)
- **Blast scaling** (`ordnance`, `burst`): Hopkinson–Cranz Z=R/W^(1/3); Sadovsky overpressure; normal reflection P_r; scaled impulse; fireball radius; Gurney fragment/plate velocity; stored-gas burst energy.

### Tartaglia (TA): exterior ballistics (~2)
- **Flat fire and recoil** (`calibres`, `tasked_fire`): flat-drag closed form v=v₀e^(−kx); free-recoil momentum including the propellant gas.

### Archimedes (AR): fluid power, as Pascal (~4)
- **Cylinders and motors** (`actuators`, `outriggers`, `symbolic_parts`, `starter`): F=pA (and the rod side), v=Q/A; motor torque T=Δp·D/2π and speed Q/D; relief-valve loss P=Δp·Q.

---

## New engines needed

| proposed engine | field | systems | laws |
|---|---|---|---|
| **Willis** | machine elements: gearing, bolted joints, tribology | `seals`, `fasteners`, `ring_joints`, `slew_drive`, `wear`, `wear_debris`, `engine_harm`, `joints`, `pneumatic_gimbal` | gear forces F_t=T/r, F_r=F_t·tanα; Lewis bending; Hertz LINE contact; Willis planetary ratio; nut-factor torque T=K·d·F; tensile stress area; thread stripping; weld throat and pin shear; Reynolds film h≈k√(μU/P), λ ratio and lubrication regimes; Archard wear V=KFs/H; filter β-ratio capture; McKibben muscle force; Couette film torque |
| **Hodgkin** | membrane transport and biophysics | turing `cellsim/transport` (ghk, kedem_katchalsky, pumps), `cellsim/membranes` | Goldman–Hodgkin–Katz flux; Kedem–Katchalsky J_v and J_s; osmotic pressure Π=σRTΔC; Na/K-ATPase Michaelis–Menten/Hill saturation; Helfrich bending energy (κ/2)(2H−C₀)². Would also absorb membrane capacitance (F6_4) and Laplace tension (NS5_6) |
| **Janssen** (Janssen/Coulomb) | granular and soil mechanics | `materials_handling`, `outriggers`, `earthworks` | Janssen silo stress; Jenike arching; Beverloo discharge; soil bearing capacity; bulking factor and repose cone; Rittinger comminution energy |
| **Poncelet** | terminal ballistics | `ballistics` | energy perforation W=toughness·A·t/cosθ; ricochet critical angle; mushrooming; fluid-layer drag |
| **Emmons** | fire engineering | `fire`, `kiln` | pool burning rate; heat-release rate Q=ṁ·LHV·χ; point-source radiation χQ/(4πr²) and ignition flux; critical water-flow suppression |
| **Maxwell** (governor/control) | regulation and feedback | `aftermarket_idle_controller`, `ecu`, `hcu`, `ignition_driver`, `dyno_controller`, `regulator`, `governor` | PI law; pole placement from the linearized plant; hysteresis relay; rev limiter |

## Largest receivers

Otto ~25, Faraday ~25, Navier–Stokes ~22, Gibbs ~18, Newton ~15,
Timoshenko ~15, Willis (new) ~15, Bjerknes ~14, Fourier ~10, Lavoisier ~9,
Zeldovich ~8, Bragg ~7, Hodgkin (new) ~7, Janssen (new) ~6. Noether, Curie,
Einstein, Hamilton, Boltzmann and Coupling receive nothing from the
established systems.

## Correction to the annex

`ENGINE_EQUATION_ANNEX.md` says Navier–Stokes is not authored anywhere. The
depth-averaged form is: `turing/src/compiler/symbolic_fluid_model.py` authors
the shallow-water equations in SymPy.
