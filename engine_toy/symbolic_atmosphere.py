"""The dewar/atmosphere law: vapour, phase change and precipitation as
one symbolic program, in the same lane as symbolic_parts.py.

A dewar, a climate chamber, a fog bank and a raincloud are the same
object: a bounded gas space holding a condensable species near its own
phase boundary, exchanging mass and latent heat with whatever solid or
liquid surface bounds it. This module authors that ONE law, once, as
SymPy, and lowers it through the repository's own SymPy -> ProcessGraph
-> SSA -> LLVM pipeline (`compile_symbolic_program`), exactly the way
`symbolic_parts.py` already does for the strut and the cylinder.

The species-specific constants are NOT invented here. `phase_table.py`
already carries, per material, the transition temperature and signed
latent heat at one atmosphere; `cryogenics.py` already carries triple
points and boil-off composition. This module's equations take those as
plain float inputs (`triple_point_k`, `triple_point_pa`, `latent_j_kg`,
...), so any material in either table can run through the same law
without a special case, and the "exceptions" (no liquid phase, pour
point instead of freezing, a colligative depression) are handled by
which INPUTS a caller supplies, not by branches inside the law.

CONVENTIONS, same as symbolic_parts.py:
  `sympy.Max`/`sympy.Min`, never the `(x + Abs(x))/2` identity.
  `_abs`/`_step` are the smooth, branch-free spellings already proven
  there (sqrt(x*x+eps); no `sympy.sign`, which has no LLVM definition).
  Every equation is `Eq(Symbol(name), expr, evaluate=False)`.

THE PURE MATH, textbook by name, each a real equation with a source:

  Clausius-Clapeyron (integrated at constant latent heat, from one
  known point -- the triple point, since it is the one point every
  material in the table already states):
      e_s(T) = P_tp * exp( -L/Rspecific * (1/T - 1/T_tp) )
  This is the same relation validated earlier against water's steam
  table point (373.15 K) using a single latent heat: it recovers the
  right ORDER of magnitude and the right monotone curve, and it is
  exactly as good as the one latent-heat number handed to it -- a
  constant-L integration is the textbook's own first approximation,
  not a bug in this module. A caller wanting the NIST curve supplies a
  temperature-varying L from its own table; the equation does not care.

  Antoine equation, the standard empirical alternative when a species'
  three fitted constants (A, B, C) are known instead of a latent heat:
      log10(P) = A - B / (C + T)
  Included as a second saturation-pressure law so a species table entry
  may supply EITHER form; `dtype`/branchless selection is the caller's,
  because which constants exist is a fact about the species, decided
  once, not a per-tick predicate.

  Relative humidity / saturation ratio:
      S = e / e_s(T)
  Supersaturation:
      s = S - 1

  Magnus-Tetens dew point inversion (already the exact form
  air_treatment.py uses for water; generalised here to accept its own
  b/c constants so a non-water condensable can use the same inversion):
      gamma = ln(RH) + b*T_c/(c + T_c)
      T_d,c = c*gamma / (b - gamma)

  Koehler curve (Kelvin curvature term + Raoult solute term), the
  standard cloud-microphysics droplet-activation law, generalised so
  the solute term is driven by however many MOLES of inorganic ion the
  caller declares are dissolved in the droplet (a literal "tiny bit of
  inorganic ions" hook -- sea-salt, dust, any electrolyte -- enters
  through `solute_mol` and nothing else has to change):
      S(r) = exp( A/r - B/r^3 )
      A = 2*Mw*sigma / (R*T*rho_w)              (Kelvin term, per unit r)
      B = 3*i*solute_mol*Mw / (4*pi*rho_w)      (Raoult/solute term)
  where `i` is the van 't Hoff dissociation count (i=2 for NaCl, the
  common sea-salt/road-salt ion pair; i=1 for a non-dissociating
  solute). This is the same curve `Koehler (1936)` derives and every
  cloud-microphysics text since reprints; the Kelvin half alone is
  `p_sat(r) = p_sat(inf) * exp(2*sigma*Vm/(r*R*T))`, and this module's
  `A` is that exponent's coefficient with `Vm = Mw/rho_w` substituted.

  Hertz-Knudsen mass flux, the standard kinetic-theory law for net
  evaporation/condensation (positive leaves the surface, negative
  deposits onto it) across a pressure difference at the interface --
  this is what a solid emitter (frost, dry ice, a wet surface) and a
  deposition (frost forming, dew forming) both are, one signed flux:
      J = alpha * (P_sat(T_surface) - P_vapour) / sqrt(2*pi*M*Rgas_u*T_surface)
  `alpha` is the sticking/evaporation coefficient (near 1 for a clean
  liquid surface, well under 1 for a contaminated or crystalline one);
  `M` is the species' OWN molar mass, so this is the one place a
  literal periodic-table number (g/mol) enters the law directly.

  Latent-heat energy balance (what a phase change DOES to the
  temperature it happens in, the same accounting phase_table.py's
  docstring states in words -- condensation heats, evaporation cools):
      dT = -J * area * L * dt / (m * cp)
  `J` positive (net evaporation) removes heat from the gas/liquid it
  left, hence the minus sign; `J` negative (net deposition/condensation)
  adds it back. This is one line of the first law of thermodynamics
  applied to a phase change, nothing beyond it.

  Precipitation terminal velocity, Stokes' law regularised into the
  quadratic-drag regime so one expression covers both a fog droplet
  (Stokes) and a raindrop (Newton drag) without a branch -- the
  standard "sphere settling" result, combined smoothly:
      v_stokes = 2*rho_w*g*r^2 / (9*mu_air)
      v_newton = sqrt(8*rho_w*g*r / (3*Cd*rho_air))
      v_t = v_stokes*v_newton / sqrt(v_stokes^2 + v_newton^2 + eps)
  the harmonic-sum blend is a smooth minimum of the two regimes' own
  velocities, which is what a real drag curve does between them; it is
  not a new law, only a branch-free join of two textbook ones.

  Shader-facing optical signals, Beer-Lambert extinction driven by
  liquid/ice water content -- the standard radiative-transfer result
  used for real-time fog and cloud rendering:
      beta = k_ext * LWC                     (extinction coefficient)
      transmittance = exp(-beta * path_m)     (Beer-Lambert law)
  and the Rayleigh/Mie regime blend, from the dimensionless size
  parameter `x = pi*D/lambda` (the same parameter Mie theory itself is
  written in terms of): `x << 1` is Rayleigh (blue-sky) scattering,
  `x >= 1` is Mie (white, forward-peaked, cloud/fog) scattering, and the
  smooth blend weight below reproduces that crossover without a branch:
      mie_weight = x^4 / (1 + x^4)            (0 near x=0, 1 for x>>1)

  Triple-state shader fractions, read directly off the SAME state this
  law already advances -- solid/liquid/vapour mass fractions, each in
  [0, 1] and summing to 1 by construction, so a renderer can bl