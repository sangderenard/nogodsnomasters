"""Emitters on the holes damage makes: what passes through them.

A puncture (damage_state.Puncture) is geometry: a position, a
direction, a radius, in a part. An emitter is the physics on that
geometry: given what the part contains (its fluid circuit -- oil,
coolant, fuel, intake air, exhaust, compressed air -- or nothing) and
that circuit's live pressure, it regulates how much passes, in which
regime, and keeps the running tally of what has been lost. The same
object is the source for two consumers: the damage sound synth
(damage_sound.py -- a spray hisses, a pour glugs, a drip plinks, a gas
leak whistles) and the view (particles() -- droplets along the real
jet arc from the hole).

Regimes, decided from the real hole and pressure, not picked:
  "spray"  a liquid under pump/boost pressure: jet speed from
           Bernoulli, v = sqrt(2 dp / rho), through the orifice with a
           sharp-edged discharge coefficient
  "pour"   a liquid at atmospheric pressure with a gravity head above
           the hole: Torricelli, v = sqrt(2 g h), a continuous stream
  "drip"   a pour so slow that surface tension holds it into drops:
           below ~2 ml/s the stream breaks into drops of a known mass
           (Tate's law, m = 2 pi r sigma / g)
  "gas"    a compressible circuit venting: the same choked-orifice law
           the pneumatic system uses (drivetrain_graph.
           choked_orifice_mass_flow_kg_s); a hot exhaust leak is this
           at its own gas temperature
  "ingest" the NEGATIVE emitter: the volume behind the hole is below
           the pressure outside it (an intake plenum at part throttle,
           a crankcase under vacuum, the suction side of a pump), so
           the hole draws IN -- atmosphere through the same orifice law
           run the other way, plus whatever other emitters are spraying
           within its reach (their jets' particles captured by the
           hole's solid angle). What comes in mixes into the circuit's
           contents: every circuit's contents are a FluidMix, a mass
           tally of n fluids (its nominal fluid, air, engine-oil,
           coolant, fuel, particulate...) -- a vacuum leak leans the
           charge, an oil spray reaching a plenum hole fouls the intake,
           air drawn into a fuel line aerates the fuel.
  "none"   a hole in a dry part, or a circuit that has run out

Regulation: a circuit only supplies pressure while it has fluid; as
the emitters drain it the remaining volume falls, the gravity head with
it, and a pump on an empty circuit supplies nothing. `lost_l` per
circuit is the hook the engine's own consequences read (oil starvation,
coolant loss) -- exposed, not yet closed back into the thermal step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from drivetrain_graph import choked_orifice_mass_flow_kg_s

G = 9.81
ATM_PA = 101_325.0
DISCHARGE_COEFF = 0.62          # a sharp-edged, ragged projectile hole
DRIP_THRESHOLD_M3_S = 2.0e-6    # below this a stream breaks into drops
# Every fluid property comes from the one registry (fluids.py): adding
# a fluid is adding a row there, not editing five tables here.
from fluids import (SURFACE_TENSION_N_M, DENSITY_KG_M3, VISCOSITY_PA_S,
                    LIQUID_OF_CIRCUIT, GAS_CIRCUITS, fluid_key,
                    leak_material_for)  # noqa: F401  (re-exported for callers)
# the fraction of dipper-flung oil retained as bore-wall film (the rest
# runs back to the pan): calibrated so a 2 L single's dipper at 2000 rpm
# (64 g/s flung) lays the ~0.35 mg/s crankcase_state's film law expected
SPLASH_FILM_RETENTION_FRAC = 5.5e-6
REYNOLDS_TURBULENT = 2300.0      # pipe/jet transition, on the LIQUID's own Reynolds number
# jet breakup regimes are set by the GAS Weber number (the ambient air's
# inertia against the liquid's surface tension, We_g = rho_air v^2 d /
# sigma -- the classic Ohnesorge/Reitz map), not the liquid's:
WEBER_RAYLEIGH = 0.4             # below: a smooth column that only pinches into drops far downstream
WEBER_FIRST_WIND = 13.0          # below: a wavy column shedding drops off its surface
WEBER_SECOND_WIND = 40.0         # below: the column tears into droplets near the hole
                                 # above: atomisation -- a mist from the orifice lip


def flow_character(fluid: str | None, speed_m_s: float, diameter_m: float) -> str:
    """How the stream leaving (or entering) a hole actually behaves,
    judged from the fluid's own properties at this speed and size:
      liquid Reynolds  Re = rho v d / mu       -- laminar below ~2300
      gas Weber        We_g = rho_air v^2 d / sigma -- Rayleigh column
                       below 0.4, wavy (first wind-induced) below 13,
                       droplets (second wind-induced) below 40,
                       atomised mist above
    A cold-oil pour at 1 m/s is a laminar glassy rope; the same oil at
    3-4 m/s off a pump line is a wavy stream shedding drops; a dipper
    flinging it at 13 m/s makes a mist; coolant at 1 m/s is already
    turbulent (its viscosity is forty times lower) but still a column.
    A gas has no free surface: its jet is laminar or turbulent only."""
    if fluid is None or speed_m_s <= 0.0:
        return "still"
    d = max(diameter_m, 1e-4)
    if fluid == "gas":
        re = AIR_DENSITY_KG_M3 * speed_m_s * d / VISCOSITY_PA_S["gas"]
        return "turbulent-jet" if re >= REYNOLDS_TURBULENT else "laminar-jet"
    rho = DENSITY_KG_M3.get(fluid, 900.0)
    re = rho * speed_m_s * d / VISCOSITY_PA_S.get(fluid, 0.001)
    we_g = AIR_DENSITY_KG_M3 * speed_m_s * speed_m_s * d / SURFACE_TENSION_N_M.get(fluid, 0.04)
    flow = "turbulent" if re >= REYNOLDS_TURBULENT else "laminar"
    if we_g >= WEBER_SECOND_WIND:
        return "atomised-mist"
    if we_g >= WEBER_FIRST_WIND:
        return f"{flow}-droplets"
    if we_g >= WEBER_RAYLEIGH:
        return f"{flow}-wavy-stream"
    return f"{flow}-stream"

INGEST_REACH_M = 0.25            # how far a hole can pull another emitter's jet from
AMBIENT_PARTICULATE_FRAC = 2e-5  # dust in bay air, by mass
AIR_DENSITY_KG_M3 = 1.2


@dataclass
class FluidMix:
    """What a circuit actually contains: a mass tally per fluid name.
    Starts as the circuit's nominal fluid; leaks remove proportionally,
    a negative emitter adds what it drew in."""
    kg: dict = field(default_factory=dict)

    @property
    def total_kg(self) -> float:
        return sum(self.kg.values())

    def add(self, fluid: str, kg: float) -> None:
        if kg > 0.0:
            self.kg[fluid] = self.kg.get(fluid, 0.0) + kg

    def remove(self, kg: float) -> None:
        tot = self.total_kg
        if tot <= 0.0 or kg <= 0.0:
            return
        f = max(0.0, 1.0 - kg / tot)
        for k in list(self.kg):
            self.kg[k] *= f

    def fractions(self) -> dict:
        tot = self.total_kg
        return {k: v / tot for k, v in self.kg.items() if tot > 0.0 and v > 0.0}

    def fouling_frac(self, nominal: str) -> float:
        """Everything that is not the circuit's own fluid, by mass."""
        return 1.0 - self.fractions().get(nominal, 0.0)


@dataclass
class HoleEmitter:
    identity: str
    part: str
    circuit: str | None
    fluid: str | None            # "engine-oil" | "coolant" | "fuel" | "water" | "gas" | None
    position: tuple[float, float, float]
    direction: tuple[float, float, float]   # outward, the way the hole faces
    radius_m: float
    through: bool
    height_above_low_m: float = 0.0          # gravity head reference: hole height above the part's low point
    # live state
    regime: str = "none"
    flow_m3_s: float = 0.0
    mass_flow_kg_s: float = 0.0
    jet_speed_m_s: float = 0.0
    drip_rate_hz: float = 0.0
    drop_mass_kg: float = 0.0
    gas_temp_k: float = 293.15
    #: Pressure this emitter last discharged FROM. Recorded rather than
    #: recomputed because the throttling drop needs it and the field has
    #: already worked it out -- see thermal_emission.emission_thermals.
    source_pressure_pa: float = ATM_PA
    spilled_kg: float = 0.0
    drip_phase: float = 0.0
    drips_this_step: int = 0
    ingested_kg: float = 0.0
    ingest_kg_s: float = 0.0          # mass drawn in this tick (air + captures), when regime == "ingest"
    character: str = "still"          # flow_character(): laminar/turbulent, stream/droplets/mist
    # fittings.py: something is screwed into this opening, so what comes
    # out goes down a hose to an appliance instead of spilling (or is fed
    # INWARD from the fitting's own supply). The FittingField moves the
    # flow; this flag stops the emitter spilling it twice.
    fitted: bool = False
    # a SPLASH emitter (kind="splash"): not a hole -- the crank's dipper
    # throwing sump oil at a bore bottom; source = the oil circuit, speed
    # = the throw's tip speed, set per tick by HoleEmitterField.step
    kind: str = "hole"
    cylinder: int = 0
    throw_radius_m: float = 0.0
    dip_depth_m: float = 0.0          # how deep the throw runs in the bath at full level
    # A CONTAINED VOLUME, with no circuit behind it. A destroyed gear
    # case, cooler or filter housing still had oil in it when it let
    # go, and that oil has to go somewhere even though it was never on
    # a pumped circuit: this emitter owns those litres and empties them
    # under their own gravity head. Zero means "ask the circuit", which
    # is what every plumbed hole does.
    contained_l: float = 0.0
    contained_head_m: float = 0.12
    # where it lands: the bore bottom it is aimed at, and the tally of
    # what has been laid on that wall as film (the rest drains back)
    target: str = ""
    target_position: tuple = ()
    deposited_kg: float = 0.0
    deposit_kg_s: float = 0.0
    # WHAT THIS ONE DISCHARGES INTO, and the cheapest honest idea of
    # that volume: the identity of the casing it sprays inside.
    #
    # Almost everything an engine sprays, it sprays INTO ITSELF. A rod
    # throwing oil off the big end, a squirter aimed at a piston crown, a
    # cam lobe flinging it round the box: none of that is lost, it hits a
    # wall and runs back to the sump. Only oil that reaches the outside
    # world is gone. Without this every internal emitter was billed to
    # `on_loss`, so an engine idling with a healthy oil bath drained its
    # own sump through the bottom of the block -- the fluid fell out of
    # the casting under gravity because nothing said the casting was
    # there.
    #
    # No geometry, deliberately: containment is a yes/no about one named
    # part, not a mesh test per droplet. It stops being yes the moment
    # that part is breached or taken off, which is the same question
    # engine_mesh.breached_casings already answers for the renderer.
    # A declared hint for where it sprays, used only when nothing has
    # asked the geometry yet.
    discharges_into: str = ""
    # WHERE THE SPRAY ACTUALLY LANDS, found by casting a few rays out of
    # the hole in its own cone and seeing what they hit: ((part, frac),
    # ...) plus the fraction that hit nothing at all and left the engine.
    # Resolved once against the ray mesh (the same one a projectile uses)
    # and re-resolved when the geometry changes, not per droplet.
    spray_hits: tuple = ()
    # WHERE IT LANDED, in world space: one impact point and surface
    # normal per ray that hit something, paired with the part it hit.
    # The part name alone is a coarse tally; a wetting engine puts fluid
    # on TRIANGLES, so it needs the point to find them. Keeping only the
    # name threw away the one thing that handoff actually needs.
    spray_points: tuple = ()
    escaped_frac: float = 1.0   # until something resolves it, assume it all leaves
    spray_resolved: bool = False
    # WETTING IS A MODE, not something every emitter does. A liquid jet
    # wets what it lands on; a gas leak reaches the same surfaces and
    # wets nothing. An emitter stays an emitter either way -- this says
    # whether the wetting engine should be handed anything at all.
    wets_surfaces: bool = True
    returned_l: float = 0.0     # running tally of what landed inside and drained back

    @property
    def area_m2(self) -> float:
        return math.pi * self.radius_m * self.radius_m

    def step(self, dt: float, pressure_pa: float, head_m: float, remaining_l: float, gas_temp_k: float) -> float:
        """Advance one tick; returns litres removed from the circuit."""
        self.drips_this_step = 0
        self.ingest_kg_s = 0.0
        if self.fluid is None:
            self.regime, self.flow_m3_s, self.mass_flow_kg_s, self.jet_speed_m_s, self.drip_rate_hz = "none", 0.0, 0.0, 0.0, 0.0
            return 0.0
        a = self.area_m2
        if pressure_pa < ATM_PA - 2_000.0:
            # the negative emitter: outside pushes in. Choked when the
            # inside is below the critical ratio, else the subsonic dp law
            if pressure_pa / ATM_PA < 0.528:
                # the same textbook choked relation as choked_orifice_mass_
                # flow_kg_s, with the ATMOSPHERE as the reservoir (that
                # helper guards against an atmospheric upstream because its
                # consumers dump into the atmosphere; here it is the source)
                gamma, r_gas, t0 = 1.4, 287.0, 293.15
                m = (DISCHARGE_COEFF * a * ATM_PA * math.sqrt(gamma / (r_gas * t0))
                     * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))
            else:
                m = DISCHARGE_COEFF * a * math.sqrt(2.0 * AIR_DENSITY_KG_M3 * (ATM_PA - pressure_pa))
            self.regime = "ingest"
            self.ingest_kg_s = m
            self.mass_flow_kg_s = m
            self.jet_speed_m_s = m / max(AIR_DENSITY_KG_M3 * a, 1e-9)
            self.flow_m3_s = 0.0
            self.drip_rate_hz = 0.0
            self.ingested_kg += m * dt
            return 0.0
        if remaining_l <= 0.0:
            self.regime, self.flow_m3_s, self.mass_flow_kg_s, self.jet_speed_m_s, self.drip_rate_hz = "none", 0.0, 0.0, 0.0, 0.0
            return 0.0
        if self.fluid == "gas":
            self.gas_temp_k = gas_temp_k
            self.source_pressure_pa = pressure_pa
            m = choked_orifice_mass_flow_kg_s(a, pressure_pa, gas_temp_k, DISCHARGE_COEFF)
            if m <= 0.0 and pressure_pa > ATM_PA * 1.02:
                # sub-critical venting: incompressible estimate on the small dp
                rho = pressure_pa / (287.0 * max(gas_temp_k, 100.0))
                m = DISCHARGE_COEFF * a * math.sqrt(2.0 * rho * (pressure_pa - ATM_PA))
            self.mass_flow_kg_s = m
            self.jet_speed_m_s = math.sqrt(1.4 * 287.0 * max(gas_temp_k, 100.0)) if m > 0.0 else 0.0
            self.regime = "gas" if m > 1e-6 else "none"
            self.flow_m3_s = 0.0
            return 0.0
        rho = DENSITY_KG_M3.get(self.fluid, 900.0)
        dp = max(0.0, pressure_pa - ATM_PA)
        if dp > 5_000.0:
            v = math.sqrt(2.0 * dp / rho)
            self.regime = "spray"
        else:
            h = max(0.0, head_m)
            v = math.sqrt(2.0 * G * h) if h > 0.0 else 0.0
            self.regime = "pour" if v > 0.0 else "none"
        q = DISCHARGE_COEFF * a * v
        if self.regime == "pour" and q < DRIP_THRESHOLD_M3_S:
            self.regime = "drip"
            sigma = SURFACE_TENSION_N_M.get(self.fluid, 0.04)
            self.drop_mass_kg = 2.0 * math.pi * self.radius_m * sigma / G
            self.drip_rate_hz = q * rho / max(self.drop_mass_kg, 1e-7)
            self.drip_phase += self.drip_rate_hz * dt
            self.drips_this_step = int(self.drip_phase)
            self.drip_phase -= self.drips_this_step
        else:
            self.drip_rate_hz = 0.0
        q = min(q, remaining_l / 1000.0 / max(dt, 1e-6))
        self.flow_m3_s = q
        self.mass_flow_kg_s = q * rho
        self.jet_speed_m_s = v
        self.spilled_kg += q * rho * dt
        self.character = flow_character(self.fluid, v, 2.0 * self.radius_m)
        return q * 1000.0 * dt

    def step_splash(self, dt: float, crank_omega_rad_s: float, fill_frac: float) -> None:
        """The dipper: the rod's big end (or its scoop) runs through the
        bath once per revolution and flings what it picks up at the
        bore above it. Throw speed is the tip speed of the throw; the
        picked-up volume is the dip's swept slice per revolution, which
        shrinks with the level -- an over-full bath drowns the crank, a
        low one is missed entirely. Nothing is removed from the circuit
        (it falls back into the pan)."""
        self.drips_this_step = 0
        self.ingest_kg_s = 0.0
        v = abs(crank_omega_rad_s) * self.throw_radius_m
        depth = self.dip_depth_m * max(0.0, fill_frac * 2.0 - 0.5)       # nothing below 25 % level, full at 75 %
        if v <= 0.05 or depth <= 0.0:
            self.regime, self.flow_m3_s, self.mass_flow_kg_s, self.jet_speed_m_s, self.character = "none", 0.0, 0.0, 0.0, "still"
            return
        rho = DENSITY_KG_M3.get(self.fluid, 870.0)
        # the swept slice: a chord of the throw circle, dip deep, dipper wide, per revolution
        chord = 2.0 * math.sqrt(max(0.0, 2.0 * self.throw_radius_m * depth - depth * depth))
        slice_m3 = chord * depth * (2.0 * self.radius_m) * 0.35        # ~a third of the swept slice actually leaves as splash
        revs_per_s = abs(crank_omega_rad_s) / (2.0 * math.pi)
        q = slice_m3 * revs_per_s
        self.regime = "splash"
        self.flow_m3_s = q
        self.mass_flow_kg_s = q * rho
        self.jet_speed_m_s = v
        self.character = flow_character(self.fluid, v, 2.0 * self.radius_m)
        # what stays on the bore wall as film: a wetted-wall retention of
        # the flung oil (the disclosed fraction that does not run straight
        # back down) -- crankcase_state's film law takes this per cylinder
        self.deposit_kg_s = self.mass_flow_kg_s * SPLASH_FILM_RETENTION_FRAC
        self.deposited_kg += self.deposit_kg_s * dt

    # spread half-angle of the stream by its character (radians): a
    # laminar rope stays a rope, droplets fan, a mist is a cloud
    SPREAD_RAD = {"laminar-stream": 0.02, "turbulent-stream": 0.05, "laminar-wavy-stream": 0.06, "turbulent-wavy-stream": 0.09,
                  "laminar-droplets": 0.12, "turbulent-droplets": 0.18, "atomised-mist": 0.35,
                  "laminar-jet": 0.08, "turbulent-jet": 0.2, "still": 0.0}

    def droplets(self, rng, n: int = 200, t_max_s: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
        """A cloud of n droplets: (positions (n,3), velocities (n,3)).
        Each is a sample of the real stream -- a random age along the
        arc and a random direction inside the character's cone, moving
        at the jet speed under gravity (a gas plume: no gravity, slowing
        as it entrains). Re-sampled every frame, so the stream flickers
        the way a real spray does."""
        if self.regime == "none" or self.jet_speed_m_s <= 0.0 or n <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        spread = self.SPREAD_RAD.get(self.character, 0.1)
        if self.regime == "splash":
            # the flung oil lands ON the bore bottom it is aimed at: the
            # cloud ends at that distance (what misses hits the case wall
            # just beyond it), a wide fan off the dipper
            reach = (float(np.linalg.norm(np.asarray(self.target_position) - np.asarray(self.position)))
                     if self.target_position else 0.08)
            t_max_s, spread = min(t_max_s, max(0.005, reach / max(self.jet_speed_m_s, 0.1))), max(spread, 0.3)
        p0 = np.asarray(self.position, dtype=float)
        d = np.asarray(self.direction, dtype=float)
        d = d / max(np.linalg.norm(d), 1e-9)
        # an orthonormal frame around the jet axis
        helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, helper); u /= max(np.linalg.norm(u), 1e-9)
        w = np.cross(d, u)
        t = rng.uniform(0.0, t_max_s, n)[:, None]
        ang = rng.uniform(0.0, 2 * np.pi, n)
        off = np.abs(rng.normal(0.0, spread, n))
        dirs = (d[None, :] * np.cos(off)[:, None] + (u[None, :] * np.cos(ang)[:, None] + w[None, :] * np.sin(ang)[:, None]) * np.sin(off)[:, None])
        # THE HOLE HAS A MOUTH, and the stream is as wide as it is.
        #
        # Every droplet used to start at the same point, so a burst core
        # plug and a pinhole produced identically thin threads and only
        # the individual drops got fatter. That is wrong for anything
        # bigger than a weep: fluid leaves across the WHOLE opening, so
        # the stream starts at the hole's diameter and only then breaks
        # up. Offsetting each droplet's origin by a random point on the
        # mouth disc costs one extra sample and is the difference
        # between a thread and a gush.
        #
        # sqrt of a uniform gives an even area distribution rather than
        # a crowd down the middle.
        mouth = float(getattr(self, "radius_m", 0.0) or 0.0)
        if mouth > 1e-4:
            m_r = mouth * np.sqrt(rng.uniform(0.0, 1.0, n))
            m_a = rng.uniform(0.0, 2 * np.pi, n)
            p0 = p0[None, :] + (u[None, :] * np.cos(m_a)[:, None]
                                + w[None, :] * np.sin(m_a)[:, None]) * m_r[:, None]
            p0 = p0.reshape(n, 3)
        else:
            p0 = p0[None, :]
        if self.regime == "ingest":
            v = min(self.jet_speed_m_s, 4.0)
            pos = p0 + dirs * (INGEST_REACH_M - np.minimum(INGEST_REACH_M, v * t))
            vel = -dirs * v
            return pos, vel
        if self.fluid == "gas":
            v = min(self.jet_speed_m_s, 3.0)
            # a plume slows as it entrains the bay air: v(t) = v0 / (1 + t/tau)
            tau = 0.08
            s = v * tau * np.log1p(t / tau)
            pos = p0 + dirs * s
            vel = dirs * (v / (1.0 + t / tau))
            return pos, vel
        v = self.jet_speed_m_s
        g = np.array([0.0, -G, 0.0])[None, :]
        pos = p0 + dirs * v * t + 0.5 * g * t * t
        vel = dirs * v + g * t
        return pos, vel

    def particles(self, n: int = 12, t_max_s: float = 0.35) -> np.ndarray:
        """Points along the jet's centre line: a ballistic arc at the jet
        speed along the hole's outward direction under gravity (y up in
        the graph frame). Empty when nothing is passing. (droplets() is
        the cloud; this is the axis, used for reach tests.)"""
        if self.regime == "none" or self.jet_speed_m_s <= 0.0:
            return np.zeros((0, 3))
        if self.regime == "splash":
            # a fan of drops off the dipper toward the bore: a short arc
            # at the throw speed, cut off where it meets the bore bottom
            t_max_s = min(t_max_s, 0.08)
        t = np.linspace(0.0, t_max_s, n)[:, None]
        p0 = np.asarray(self.position)[None, :]
        d = np.asarray(self.direction)[None, :]
        if self.regime == "ingest":
            # streamlines converging on the hole from its reach
            v = min(self.jet_speed_m_s, 4.0)
            return p0 + d * (INGEST_REACH_M - np.minimum(INGEST_REACH_M, v * t))
        v = self.jet_speed_m_s if self.fluid != "gas" else min(self.jet_speed_m_s, 3.0)
        g = np.array([0.0, -G, 0.0])[None, :] if self.fluid != "gas" else np.zeros((1, 3))
        return p0 + d * v * t + 0.5 * g * t * t


@dataclass
class HoleEmitterField:
    """Every emitter on the engine, stepped together against the live
    circuits.

    It does NOT own fluid state. The driving pressure behind a hole and
    the contents behind it belong to the engine's own circuits, so the
    field asks for both through callbacks the sim supplies
    (`pressure_of`, `remaining_of`) and reports every gram it takes back
    through `on_loss`, which the sim applies to the REAL reservoir --
    the tank's fill, the sump's oil, the receiver's charge. `lost_l`
    here is a running tally for display only; it is never the truth
    about how much is left."""
    emitters: list[HoleEmitter] = field(default_factory=list)
    lost_l: dict[str, float] = field(default_factory=dict)
    pressure_of = None       # (circuit_id, circuit) -> Pa really behind the hole
    remaining_of = None      # (circuit_id, circuit) -> litres actually left
    on_loss = None           # (circuit_id, litres) -> None, applied to the real reservoir
    mix: dict[str, FluidMix] = field(default_factory=dict)       # circuit identity -> its contents
    ingest_kg_s: dict[str, float] = field(default_factory=dict)  # circuit identity -> mass drawn in this tick
    # part identity -> mass of fluid laid on it by spray that landed
    # inside the engine. Not a loss: this is oil on the inside of the
    # castings, on its way back to the sump.
    wetted_kg: dict[str, float] = field(default_factory=dict)

    # A real jet is a cone, not a line: a handful of rays across it is
    # enough to say what it is pointed at and how much of it clears the
    # engine, which is the only question that changes the oil level.
    SPRAY_CONE_DEG = 18.0
    SPRAY_RAYS = 9
    SPRAY_REACH_M = 1.2              # past this, treat it as having left
    SPRAY_START_CLEARANCE_M = 0.004  # clear of the hole's own lip before testing

    @staticmethod
    def _is_self_hit(em: HoleEmitter, part: str) -> bool:
        """Whether this hit is the emitter's own hole or its own part.

        Mesh part names are the node identity with separators flattened
        (`node_powertrain_head_bank1_oil_fill`), so the comparison is
        made on that flattened form of the emitter's own identities
        rather than by looking for words inside a name."""
        def flat(x: str) -> str:
            return "node_" + str(x).replace("/", "_").replace(".", "_")
        own = {flat(em.part)}
        base = em.identity.rsplit(".", 1)[0] if "." in em.identity else em.identity
        own.add(flat(base))
        own.add(flat(em.identity))
        # the port's own drawn geometry is named for the port, not the node
        own.add("port_" + str(base).replace("powertrain.", "").replace(".", "_"))
        return part in own

    def resolve_spray(self, ray_mesh, only_unresolved: bool = True) -> int:
        """Cast each emitter's spray cone and record what it lands on.

        The engine already carries a ray mesh for projectiles; a jet of
        oil is asking the same question a bullet does -- what is in front
        of me, and do I get out. So this uses it, rather than inventing a
        second idea of where the inside of the engine is.

        Call it once after the graph is built, and again whenever the
        geometry changes (a cover comes off, a casing is holed): an
        emitter that used to spray onto the inside of a valve cover
        starts spraying at the sky the moment the cover is gone, and the
        oil it throws genuinely does start leaving the engine."""
        from engine_rays import Ray
        if ray_mesh is None:
            return 0
        done = 0
        half = math.radians(self.SPRAY_CONE_DEG) * 0.5
        for em in self.emitters:
            if only_unresolved and em.spray_resolved:
                continue
            origin = np.asarray(em.position, dtype=float)
            axis = np.asarray(em.direction, dtype=float)
            n = float(np.linalg.norm(axis))
            if n < 1e-9:
                continue
            axis = axis / n
            # any two directions across the axis, to fan the cone over
            tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(axis, tmp); u /= max(np.linalg.norm(u), 1e-9)
            v = np.cross(axis, u)
            tally: dict[str, int] = {}
            points: list = []
            escaped = 0
            for k in range(self.SPRAY_RAYS):
                if k == 0:
                    d = axis
                else:
                    ang = 2.0 * math.pi * (k - 1) / max(1, self.SPRAY_RAYS - 1)
                    d = axis * math.cos(half) + (u * math.cos(ang) + v * math.sin(ang)) * math.sin(half)
                    d = d / max(np.linalg.norm(d), 1e-9)
                # START CLEAR OF THE HOLE ITSELF. A port has its own
                # geometry in the mesh, and a ray launched from its
                # centre hits it immediately -- which reported every jet
                # as landing on the very hole it came out of, and so as
                # perfectly contained no matter what was really in front
                # of it.
                hit = ray_mesh.pick(Ray(origin + d * self.SPRAY_START_CLEARANCE_M, d))
                if hit is not None and self._is_self_hit(em, hit.part):
                    hit = None
                if hit is None or float(getattr(hit, "t", 0.0)) > self.SPRAY_REACH_M:
                    escaped += 1
                else:
                    tally[hit.part] = tally.get(hit.part, 0) + 1
                    points.append((hit.part,
                                   tuple(float(v) for v in hit.point),
                                   tuple(float(v) for v in hit.normal)))
            total = float(self.SPRAY_RAYS)
            em.spray_hits = tuple(sorted(((p, c / total) for p, c in tally.items()),
                                          key=lambda t: -t[1]))
            em.spray_points = tuple(points)
            em.escaped_frac = escaped / total
            em.spray_resolved = True
            done += 1
        return done

    def deposit_spray(self, em: HoleEmitter, litres: float) -> None:
        """Lay what landed onto the parts the rays actually hit.

        Kept as a running mass per part so anything that cares -- a
        thermal model asking which surfaces are oil-wetted, a view
        drawing wet castings -- reads it off one tally instead of
        tracking droplets."""
        if litres <= 0.0:
            return
        kg = litres / 1000.0 * DENSITY_KG_M3.get(em.fluid, 900.0)
        if not em.spray_hits:
            self.wetted_kg[em.part] = self.wetted_kg.get(em.part, 0.0) + kg
            return
        inside = sum(f for _, f in em.spray_hits) or 1.0
        for part, frac in em.spray_hits:
            self.wetted_kg[part] = self.wetted_kg.get(part, 0.0) + kg * (frac / inside)

    # Which declared fluid roles are a PUMPED LIQUID that will come out
    # of an open hole. Intake, exhaust and ignition ports are open by
    # design -- they are valve seats and plug holes, not leaks -- and a
    # crankcase breather is supposed to breathe.
    PUMPED_LIQUID_ROLES = {"oil": "oil", "coolant": "coolant", "fuel": "fuel"}

    def sync_open_ports(self, graph: dict) -> tuple[int, int]:
        """An emitter for every fluid port that is open RIGHT NOW.

        A line port is not a leak because of what it is called; it is a
        leak when the thing that closes it is not there. Every port
        declares its `closure` -- an oil filler's screw cap, a sump's
        drain plug, a gallery plug, the dipstick -- and this makes a hole
        out of any port that has a pumped liquid behind it and nothing in
        it. Take the cap off a running engine and oil comes out of the
        filler at whatever the gallery is making, which is exactly what
        happens if you do that to a real one.

        LIVE and idempotent: call it whenever the configuration changes.
        Putting the cap back takes the emitter away again, so a build
        that is reconfigured, hacked, or assembled wrong behaves the same
        way as one that was shot -- there is no separate code path for
        "the user did this on purpose".

        Where the oil GOES is not decided here. These emitters resolve
        their spray against the geometry like any other, so a filler
        under an intact valve cover sprays onto the inside of that cover
        and drains back, and the same filler with the cover off sprays
        into the engine bay and is gone."""
        want: dict[str, dict] = {}
        for n in graph.get("nodes", ()):
            circuit = self.PUMPED_LIQUID_ROLES.get(str(n.get("fluid_role") or ""))
            if circuit is None:
                continue
            if n.get("connected") or n.get("plugged"):
                continue
            want[f"{n['identity']}.open_port"] = dict(node=n, circuit=circuit)
        have = {em.identity: em for em in self.emitters if em.kind == "open-port"}
        # ports that have just been opened
        added = 0
        for ident, spec in want.items():
            if ident in have:
                continue
            n = spec["node"]
            d = n.get("port_direction") or (0.0, 1.0, 0.0)
            self.emitters.append(HoleEmitter(
                identity=ident, part=str(n.get("part") or n["identity"]), circuit=spec["circuit"],
                fluid={"oil": "engine-oil", "coolant": "coolant", "fuel": "fuel"}[spec["circuit"]],
                position=tuple(float(v) for v in n.get("reference_position", (0.0, 0.0, 0.0))),
                direction=tuple(float(v) for v in d),
                radius_m=float(n.get("port_radius_m") or 0.004), through=True, kind="open-port"))
            added += 1
        # ports that have just been closed again
        removed = 0
        for ident, em in have.items():
            if ident not in want:
                self.emitters.remove(em)
                removed += 1
        return added, removed

    def add_splash_from_graph(self, graph: dict) -> list[HoleEmitter]:
        """One splash emitter per `oil-splash-path` edge dressing.py
        declares (pan or trough -> each cylinder's bore bottom): placed
        at the bottom of that cylinder's own crank throw (cylinder_ports
        layout: crank centre, throw radius), aimed at the bore bottom
        port. These are the dipper, not holes; the edges stay in the
        graph as the circuit's declaration (the solver keys splash
        lubrication off them) but are no longer drawn as pipes."""
        from cylinder_ports import deserialize_layout
        layout_data = graph.get("cylinder_layout")
        if not layout_data:
            return []
        geoms = {g.number: g for g, _ in deserialize_layout(layout_data)}
        by_node = {n["identity"]: n for n in graph.get("nodes", ())}
        made = []
        for e in graph.get("edges", ()):
            if (e.get("constraint") or e.get("kind")) != "oil-splash-path":
                continue
            g = geoms.get(int(e.get("cylinder", 0)))
            target = by_node.get(e["b"])
            if g is None or target is None:
                continue
            centre = np.asarray(g.crank_centre, dtype=float)
            axis = np.asarray(g.axis, dtype=float)
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            r = float(g.crank_radius_m) or 0.04
            dip = centre - axis * r                                  # the throw's bottom dead centre, along the bore axis
            aim = np.asarray(target["reference_position"], dtype=float) - dip
            aim = aim / max(np.linalg.norm(aim), 1e-9)
            em = HoleEmitter(identity=f"{e['identity']}.dipper", part=e["a"], circuit="oil", fluid="engine-oil",
                             position=tuple(float(v) for v in dip), direction=tuple(float(v) for v in aim),
                             radius_m=0.006, through=False, kind="splash", cylinder=int(g.number),
                             throw_radius_m=r, dip_depth_m=max(0.004, r * 0.25), target=e["b"],
                             target_position=tuple(float(v) for v in target["reference_position"]))
            self.emitters.append(em)
            made.append(em)
        return made

    def contents(self, cid: str, circuit, nominal: str) -> FluidMix:
        m = self.mix.get(cid)
        if m is None:
            vol_m3 = float(getattr(circuit, "volume_l", 1.0) or 1.0) / 1000.0
            if nominal == "gas":
                rho = float(getattr(circuit, "pressure_pa", ATM_PA)) / (287.0 * max(float(getattr(circuit, "temp_k", 293.15)), 100.0))
                nominal = cid
            else:
                rho = DENSITY_KG_M3.get(nominal, 900.0)
            m = FluidMix({nominal: vol_m3 * rho})
            self.mix[cid] = m
        return m

    def add_from_punctures(self, recorded, graph: dict, circuits, part_low_y: dict | None = None) -> list[HoleEmitter]:
        """One emitter per boundary hole (entry, and exit when it went
        through) of every puncture in a part that contains something."""
        by_node = {n["identity"]: n for n in graph.get("nodes", ())}
        circuit_of_part = {}
        for c in circuits:
            cid = circuit_identity(c)
            for nid in c.nodes:
                circuit_of_part.setdefault(nid, cid)
            for e in c.edges:
                circuit_of_part.setdefault(e["identity"], cid)
        from damage_state import owning_part
        made = []
        for identity, p in recorded:
            cid = circuit_of_part.get(identity)
            if cid is None:
                # a struck casting piece belongs to a real part: what
                # leaks out of it is what THAT part's circuit carries
                owner = owning_part(identity)
                if owner is not None:
                    cid = circuit_of_part.get(owner)
            if cid is None:
                cid = self._infer_circuit(identity, p.material)
            fluid = None
            if cid in LIQUID_OF_CIRCUIT:
                fluid = LIQUID_OF_CIRCUIT[cid]
            elif cid is not None and any(cid.startswith(g) for g in GAS_CIRCUITS):
                fluid = "gas"
            low_y = (part_low_y or {}).get(identity)
            if low_y is None:
                node = by_node.get(identity, {})
                # `.get(key, default)` hands back a stored None rather
                # than the default, and plenty of real nodes carry the
                # key with no value -- so fall back explicitly
                pos = node.get("reference_position") or p.entry_position_m
                half = node.get("body_half_extent_m") or (0.05, 0.05, 0.05)
                low_y = float(pos[1]) - float(half[1])
            sides = [("entry", p.entry_position_m, tuple(-c for c in p.direction))]
            if p.through and p.exit_position_m is not None:
                sides.append(("exit", p.exit_position_m, tuple(p.direction)))
            for side, pos, d in sides:
                em = HoleEmitter(identity=f"{identity}.hole_{len(self.emitters) + 1}_{side}", part=identity, circuit=cid,
                                 fluid=fluid, position=tuple(float(v) for v in pos), direction=tuple(float(v) for v in d),
                                 radius_m=float(p.radius_m), through=bool(p.through),
                                 height_above_low_m=max(0.0, float(pos[1]) - low_y))
                self.emitters.append(em)
                made.append(em)
        return made

    @staticmethod
    def _infer_circuit(identity: str, material: str) -> str | None:
        tail = identity.split(".")[-1]
        if any(k in identity for k in ("water_jacket", "coolant_face", "radiator", "_hopper")):
            return "coolant"
        if any(k in tail for k in ("sump", "oil_pan", "splash", "trough")) or material == "lube":
            return "oil"
        if "oil" in tail:
            return "oil"
        if any(k in tail for k in ("water", "coolant", "radiator")) or "water_jacket" in identity:
            return "coolant"
        if any(k in identity for k in ("hydraulic", "hydronic", "fan_drive_motor")):
            return "hydraulic"
        if any(k in identity for k in ("chiller", "condenser", "receiver_drier", "evaporator",
                                       "expansion_valve", "refrigerant")):
            return "refrigerant"
        if "nitrogen" in identity:
            return "nitrogen"
        if "fuel" in tail or material == "fuel":
            return "fuel"
        if "exhaust" in tail or material == "exhaust":
            return "exhaust"
        if any(k in tail for k in ("intake", "plenum", "throttle", "air_filter")) or material == "intake":
            return "intake-air"
        return None

    def step(self, dt: float, circuits, exhaust_temp_k: float = 293.15, crank_omega_rad_s: float = 0.0) -> None:
        by_id = {circuit_identity(c): c for c in circuits}
        # a pumped liquid circuit can only push through its holes what the
        # pump delivers (FluidCircuit.flow_lpm, its real speed-derived
        # flow): when the holes' Bernoulli demand at line pressure
        # exceeds that, the pressure the holes see collapses to whatever
        # drives exactly the pump's flow through the total hole area
        collapsed_pa: dict[str, float] = {}
        for cid, c in by_id.items():
            liquid = LIQUID_OF_CIRCUIT.get(cid)
            if liquid is None:
                continue
            holes = [e for e in self.emitters if e.circuit == cid and e.fluid == liquid]
            if not holes or float(getattr(c, "pressure_pa", ATM_PA)) <= ATM_PA + 5_000.0:
                continue
            rho = DENSITY_KG_M3.get(liquid, 900.0)
            dp = float(c.pressure_pa) - ATM_PA
            area = sum(e.area_m2 for e in holes)
            demand = DISCHARGE_COEFF * area * math.sqrt(2.0 * dp / rho)          # m3/s at full line pressure
            pump = float(getattr(c, "flow_lpm", 0.0) or 0.0) / 60000.0           # m3/s the pump makes
            if pump > 0.0 and demand > pump:
                v = pump / max(DISCHARGE_COEFF * area, 1e-9)
                collapsed_pa[cid] = ATM_PA + 0.5 * rho * v * v
        self.ingest_kg_s = {}
        for em in self.emitters:
            if em.fitted:
                em.regime = "fitted"
                em.flow_m3_s = em.mass_flow_kg_s = em.jet_speed_m_s = 0.0
                continue
            c = by_id.get(em.circuit) if em.circuit else None
            if em.kind == "splash":
                vol = float(c.volume_l) if c is not None else 1.0
                fill = max(0.0, vol - self.lost_l.get(em.circuit, 0.0)) / max(vol, 1e-6)
                em.step_splash(dt, crank_omega_rad_s, fill)
                continue
            if em.fluid is None:
                em.step(dt, ATM_PA, 0.0, 0.0, 293.15)
                continue
            if em.fluid == "gas":
                p = (float(self.pressure_of(em.circuit, c)) if self.pressure_of is not None
                     else (float(c.pressure_pa) if c is not None else ATM_PA))
                temp = exhaust_temp_k if em.circuit == "exhaust" else (float(c.temp_k) if c is not None else 293.15)
                em.step(dt, p, 0.0, 1e9, temp)
                em.character = flow_character("gas", em.jet_speed_m_s, 2.0 * em.radius_m) if em.regime != "none" else "still"
                if em.regime == "gas" and em.mass_flow_kg_s > 0.0 and self.on_loss is not None:
                    # a holed receiver really does bleed down: the mass
                    # leaving is mass the circuit no longer has
                    self.on_loss(em.circuit, em.mass_flow_kg_s * dt)
                if em.regime == "ingest" and c is not None:
                    self._ingest(em, c, dt)
                continue
            if em.contained_l > 0.0 or (em.circuit is None and em.kind == "contained"):
                # its own contents, emptying under their own head: no
                # pump behind it, so this stops when the part is empty
                head = em.contained_head_m * min(1.0, em.contained_l / 2.0)
                removed = em.step(dt, ATM_PA, head, em.contained_l, 293.15)
                em.contained_l = max(0.0, em.contained_l - removed)
                continue
            vol = float(c.volume_l) if c is not None else 1.0
            if self.remaining_of is not None:
                remaining = float(self.remaining_of(em.circuit, c))
                vol = max(vol, remaining)
            else:
                remaining = max(0.0, vol - self.lost_l.get(em.circuit, 0.0))
            # the pump pressurises only what it has; the gravity head
            # shrinks with the level (the hole sees the fluid above it)
            fill = remaining / max(vol, 1e-6)
            live_p = (float(self.pressure_of(em.circuit, c)) if self.pressure_of is not None
                      else (float(c.pressure_pa) if c is not None else ATM_PA))
            p = live_p if fill > 0.05 else ATM_PA
            p = min(p, collapsed_pa.get(em.circuit, p))
            if em.height_above_low_m > 0.0:
                head = max(0.0, em.height_above_low_m * (fill - 0.5) * 2.0) if fill > 0.0 else 0.0
            else:
                head = 0.05 * fill
            removed = em.step(dt, p, head, remaining, 293.15)
            if removed > 0.0:
                # SPLIT IT BY WHERE IT WENT. Whatever the spray rays
                # landed on inside the engine drains back to the sump --
                # real flow out of the hole, real pressure drop, no oil
                # gone from the machine. Only the fraction that found its
                # way out is actually lost. Billing the whole lot to
                # `on_loss` is what let a healthy engine drain its own
                # sump through the bottom of the block while idling.
                escaped = removed * max(0.0, min(1.0, em.escaped_frac))
                landed = removed - escaped
                if landed > 0.0:
                    em.returned_l += landed
                    self.deposit_spray(em, landed)
                if escaped > 0.0:
                    self.lost_l[em.circuit] = self.lost_l.get(em.circuit, 0.0) + escaped
                    if self.on_loss is not None:
                        self.on_loss(em.circuit, escaped)      # the REAL reservoir loses it
                    if c is not None:
                        self.contents(em.circuit, c, em.fluid).remove(escaped / 1000.0 * DENSITY_KG_M3.get(em.fluid, 900.0))
            elif em.regime == "ingest" and c is not None:
                self._ingest(em, c, dt)

    def _ingest(self, em: HoleEmitter, circuit, dt: float) -> None:
        """What a negative emitter pulls in this tick: bay air (with its
        dust), and the jets of every other emitter whose particles pass
        within reach -- captured in proportion to the hole's solid angle
        at that distance -- all added to the circuit's mix."""
        mix = self.contents(em.circuit, circuit, em.fluid)
        m_air = em.ingest_kg_s * dt
        mix.add("air", m_air * (1.0 - AMBIENT_PARTICULATE_FRAC))
        mix.add("particulate", m_air * AMBIENT_PARTICULATE_FRAC)
        hole = np.asarray(em.position)
        for other in self.emitters:
            if other is em or other.regime in ("none", "ingest") or other.mass_flow_kg_s <= 0.0:
                continue
            pts = other.particles(8, 0.3)
            if not len(pts):
                continue
            d = float(np.min(np.linalg.norm(pts - hole[None, :], axis=1)))
            if d > INGEST_REACH_M:
                continue
            capture = min(1.0, em.area_m2 / (math.pi * d * d + em.area_m2)) * 4.0
            captured = other.mass_flow_kg_s * min(1.0, capture) * dt
            mix.add(other.fluid or "gas", captured)
            em.ingested_kg += captured
            em.ingest_kg_s += captured / max(dt, 1e-9)
        self.ingest_kg_s[em.circuit] = self.ingest_kg_s.get(em.circuit, 0.0) + em.ingest_kg_s
        if em.fluid == "gas":
            # a gas circuit is flow-through (the engine draws the plenum
            # down every cycle, the pipe empties out the tail): its
            # contents hold a constant mass, so what came in displaces an
            # equal mass of the current blend -- the mix tracks the live
            # proportions instead of accumulating forever
            mix.remove(em.ingest_kg_s * dt)

    def splash_deposit_kg_s(self, n_cyl: int):
        """Per-cylinder film deposit rate from the dippers (index =
        cylinder - 1), or None when this engine has no splash emitters
        (the crankcase model then keeps its own uniform law)."""
        sp = [e for e in self.emitters if e.kind == "splash"]
        if not sp:
            return None
        out = np.zeros(max(1, n_cyl))
        for e in sp:
            if 0 < e.cylinder <= n_cyl:
                out[e.cylinder - 1] += e.deposit_kg_s
        return out

    def fouling(self) -> dict:
        """circuit -> {fluid: mass fraction} for every circuit whose
        contents are no longer purely its own fluid."""
        out = {}
        for cid, m in self.mix.items():
            fr = m.fractions()
            if len(fr) > 1:
                out[cid] = {k: round(v, 4) for k, v in sorted(fr.items(), key=lambda kv: -kv[1])}
        return out

    def summary(self) -> tuple:
        """The synth's view: (identity, fluid, regime, mass_flow, jet_speed, drips_this_step, radius, character)."""
        return tuple((em.identity, em.fluid, em.regime, em.mass_flow_kg_s, em.jet_speed_m_s, em.drips_this_step, em.radius_m,
                      em.character)
                     for em in self.emitters if em.regime != "none")


def circuit_identity(circuit) -> str:
    named = circuit.edge_identity          # computed once, see FluidCircuit
    if named is not None:
        return named
    names = " ".join(circuit.nodes)
    if "coolant" in names or "radiator" in names or "water_pump" in names:
        return "coolant"
    return circuit.kind_class


# ---------------------------------------------------------------------
# THE HANDOFF TO A WETTING ENGINE
# ---------------------------------------------------------------------
#
# An emitter is an emitter: it has a hole, a pressure behind it, a
# direction, and a rate. Where the fluid GOES once it lands is somebody
# else's problem, and a much bigger one -- spreading over map triangles,
# running downhill, pooling, drying, catching fire. This is the seam.
#
# What is handed over is deliberately small: the impact points the spray
# rays already found, the surface normal at each, and the mass arriving
# there per second. A wetting engine resolves those to triangles itself,
# because it owns the map and this module does not.
#
# Splash deposits come through the same seam. The dipper emitters
# already carry a `target` and a running `deposited_kg` on a bore wall,
# which is the same statement -- fluid, arriving somewhere, at a rate --
# so a wetting engine that can take one can take the other.


def wetting_deposits(field: "HoleEmitterField", dt: float = 1.0) -> list:
    """Every point taking fluid right now, for a wetting engine.

    One entry per impact point: the part hit, the world position, the
    surface normal there, the fluid, and the mass landing on it this
    tick. Emitters that do not wet anything (a gas leak) are left out,
    and so are emitters whose spray has not been resolved against the
    geometry yet -- an unresolved one has no idea where it is pointing
    and should not be guessed at."""
    out: list = []
    for em in field.emitters:
        if not getattr(em, "wets_surfaces", True) or not em.spray_resolved:
            continue
        if em.regime in ("none", "gas"):
            continue
        landed_kg_s = max(0.0, em.mass_flow_kg_s) * (1.0 - em.escaped_frac)
        if landed_kg_s <= 0.0 or not em.spray_points:
            continue
        share = landed_kg_s / float(len(em.spray_points))
        for part, point, normal in em.spray_points:
            out.append({
                "emitter": em.identity,
                "part": part,
                "position": point,
                "normal": normal,
                "fluid": em.fluid,
                "kg": share * float(dt),
                "source": em.kind,
            })
    return out


def splash_deposits(field: "HoleEmitterField", dt: float = 1.0) -> list:
    """The same statement for the splash dippers.

    A dipper lays film on a bore wall rather than spraying at whatever
    is in front of it, so its target is declared rather than ray-found
    -- but it is still fluid arriving somewhere at a rate, which is all
    a wetting engine needs."""
    out: list = []
    for em in field.emitters:
        if em.kind != "splash" or not em.target:
            continue
        rate = max(0.0, float(getattr(em, "deposit_kg_s", 0.0)))
        if rate <= 0.0:
            continue
        out.append({
            "emitter": em.identity,
            "part": em.target,
            "position": tuple(em.target_position) if em.target_position else em.position,
            "normal": (),
            "fluid": em.fluid,
            "kg": rate * float(dt),
            "source": "splash",
        })
    return out
