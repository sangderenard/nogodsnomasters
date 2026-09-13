"""Explosive ordnance: a charge you place, with a fuze.

You should not have to find the explodey part on an engine. Put a
charge somewhere -- on a named part, at a point, or stuck to whatever
a shot was aimed at -- give it a fuze, and let the blast decide what
fails.

  `Explosive`   a real filling: TNT, C4/PE4 (plastic), Semtex, PETN,
                RDX, ANFO, black powder, det cord. Each declares its
                TNT equivalence (energy relative to TNT), its Gurney
                velocity sqrt(2E) (what its detonation products can
                throw a casing at) and its detonation velocity.
  `Charge`      mass of filling, optional casing mass (bare = pure
                blast, cased = blast plus fragments), where it sits,
                and its FUZE:
                  "command"  fires when you say so (detonate())
                  "timer"    fires timer_s after it is armed
                  "delay"    fires delay_s after something disturbs it
                             (a contact that is deliberately late)
                  "contact"  fires the moment it is disturbed -- placed
                             against a part it fires on arming; loose,
                             it fires when a projectile strikes it
                             (impact_sensitivity_j)
  `OrdnanceField` holds the placed charges, steps their fuzes, and
                detonates them.

What a detonation does is read from the real charge, never a constant:

  BLAST -- the free-air spherical relation (Sadovsky) for peak
  overpressure at scaled distance Z = R / W^(1/3), W in kg TNT
  equivalent:
      dP [MPa] = 0.085/Z + 0.3/Z^2 + 0.8/Z^3
  reflected at a surface facing the charge by the standard normal-
  reflection relation P_r = 2 dP + 6 dP^2 / (dP + 7 p0), with the
  specific impulse from the same scaling. Each part then gets that
  impulse over its own presented area: the plate takes velocity
  v = i_s / (areal density), and fails when that exceeds its own
  material's fracture velocity scale sqrt(2 * toughness / density) --
  the classic impulsive-loading plate criterion, so a thin alloy cover
  ten centimetres away shreds while a cast-iron block a metre off is
  merely dented, out of the materials the parts already declare.

  FRAGMENTS -- a cased charge throws its casing at the Gurney velocity
  V = sqrt(2E) * (M/C + 0.5)^(-1/2) (cylindrical), broken into a
  Mott-style spread of masses; each significant fragment is fired
  through the projectile engine (the same cascade the burst fragments
  use), so a cased charge genuinely swiss-cheeses what is around it.

Everything downstream is the machinery that already exists: parts that
fail go through `EngineCycleSim.burst_part` (absent, contents out,
their own fragments, a hole emitter on the open end), parts that are
merely holed get real punctures in `damage_state`, and `node_effects`
reads the consequences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

ATM_PA = 101_325.0
TNT_J_PER_KG = 4.184e6


@dataclass(frozen=True)
class Explosive:
    name: str
    tnt_equivalence: float        # energy relative to TNT, by mass
    gurney_m_s: float             # sqrt(2E), the casing-throwing constant
    detonation_m_s: float
    density_kg_m3: float
    impact_sensitivity_j: float   # struck with at least this much, it goes off
    note: str = ""

    @property
    def cj_pressure_pa(self) -> float:
        """Chapman-Jouguet detonation pressure, P = rho D^2 / 4 -- the
        physical ceiling on what this filling can push with. The free-air
        blast relations below are only valid outside the charge itself;
        pressed against a wall, this is what that wall actually sees."""
        return self.density_kg_m3 * self.detonation_m_s ** 2 / 4.0

    @property
    def fireball_radius_per_kg13_m(self) -> float:
        """Fireball radius scaling, R = k W^(1/3): the disclosed k ~ 3.0
        m/kg^(1/3) for a condensed high explosive's own luminous
        products -- what can light a fuel cloud."""
        return 3.0


EXPLOSIVES: dict[str, Explosive] = {e.name: e for e in (
    Explosive("tnt", 1.00, 2440.0, 6900.0, 1630.0, 60.0, "the reference filling"),
    Explosive("c4", 1.34, 2680.0, 8040.0, 1601.0, 200.0, "PE4/C4 plastic: RDX in a plasticiser, very insensitive"),
    Explosive("semtex", 1.25, 2700.0, 7600.0, 1430.0, 180.0, "RDX/PETN plastic"),
    Explosive("petn", 1.27, 2930.0, 8400.0, 1770.0, 25.0, "det cord / booster filling, sensitive"),
    Explosive("rdx", 1.60, 2930.0, 8750.0, 1820.0, 30.0, "bare RDX"),
    Explosive("anfo", 0.82, 1600.0, 3200.0, 840.0, 500.0, "bulk blasting agent, needs a booster"),
    Explosive("black powder", 0.55, 700.0, 600.0, 1700.0, 5.0, "deflagrates rather than detonates; very sensitive"),
    Explosive("det cord", 1.27, 2930.0, 7000.0, 1500.0, 20.0, "PETN core, grams per metre"),
)}

FUZES = ("command", "timer", "delay", "contact")


def get_explosive(name: str) -> Explosive:
    key = name.strip().lower()
    aliases = {"pe4": "c4", "plastic": "c4", "plastique": "c4", "comp b": "tnt", "gunpowder": "black powder",
               "cordtex": "det cord", "primacord": "det cord"}
    key = aliases.get(key, key)
    if key not in EXPLOSIVES:
        raise KeyError(f"unknown explosive {name!r}; known: {', '.join(EXPLOSIVES)}")
    return EXPLOSIVES[key]


# ---------------------------------------------------------------------------
# blast relations
# ---------------------------------------------------------------------------

def overpressure_pa(w_tnt_kg: float, r_m: float) -> float:
    """Sadovsky's free-air spherical peak overpressure at range r."""
    if w_tnt_kg <= 0.0:
        return 0.0
    r = max(r_m, 0.05)
    z = r / (w_tnt_kg ** (1.0 / 3.0))
    mpa = 0.085 / z + 0.3 / (z * z) + 0.8 / (z * z * z)
    return mpa * 1e6


def reflected_pa(incident_pa: float) -> float:
    """Normal reflection off a surface facing the charge."""
    p = max(0.0, incident_pa)
    return 2.0 * p + 6.0 * p * p / (p + 7.0 * ATM_PA)


# Sadovsky's companion specific-impulse scaling, i_s = K W^(2/3) / R,
# with the disclosed K that reproduces the standard free-air curve in
# the near field this model lives in (metres, kilograms).
IMPULSE_K_PA_S = 200.0


def specific_impulse_pa_s(w_tnt_kg: float, r_m: float) -> float:
    if w_tnt_kg <= 0.0:
        return 0.0
    return IMPULSE_K_PA_S * (w_tnt_kg ** (2.0 / 3.0)) / max(r_m, 0.05)


def gurney_velocity_m_s(explosive: Explosive, charge_kg: float, casing_kg: float) -> float:
    """Cylindrical Gurney: what the casing leaves at."""
    if charge_kg <= 0.0 or casing_kg <= 0.0:
        return 0.0
    return explosive.gurney_m_s / math.sqrt(casing_kg / charge_kg + 0.5)


@dataclass
class Charge:
    explosive: str = "c4"
    mass_kg: float = 0.25
    casing_kg: float = 0.0
    # where it is: a graph part it is stuck to (its position is read
    # live), or an absolute point
    attached_to: str | None = None
    offset_m: tuple = (0.0, 0.0, 0.0)
    position: tuple | None = None
    fuze: str = "command"
    timer_s: float = 5.0
    delay_s: float = 0.0
    armed: bool = False
    fired: bool = False
    identity: str = "charge"
    # live fuze state
    elapsed_s: float = 0.0
    disturbed: bool = False
    _delay_left: float | None = None

    @property
    def spec(self) -> Explosive:
        return get_explosive(self.explosive)

    @property
    def tnt_kg(self) -> float:
        return self.mass_kg * self.spec.tnt_equivalence

    def where(self, graph: dict) -> np.ndarray:
        if self.attached_to:
            for n in graph.get("nodes", ()):
                if n["identity"] == self.attached_to and n.get("reference_position") is not None:
                    return np.asarray(n["reference_position"], dtype=float) + np.asarray(self.offset_m, dtype=float)
            for e in graph.get("edges", ()):
                if e["identity"] == self.attached_to:
                    pos = {n["identity"]: n.get("reference_position") for n in graph.get("nodes", ())}
                    a, b = pos.get(e["a"]), pos.get(e["b"])
                    if a is not None and b is not None:
                        return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0 + np.asarray(self.offset_m, dtype=float)
        return np.asarray(self.position if self.position is not None else (0.0, 0.0, 0.0), dtype=float)

    def arm(self) -> None:
        self.armed = True
        self.elapsed_s = 0.0
        if self.fuze == "contact" and self.attached_to:
            self.disturbed = True          # placed against something: it is already in contact

    def disturb(self, energy_j: float = 1e9) -> bool:
        """Something struck or moved it. Returns True if that starts a
        firing sequence (a contact fuze needs the filling's own impact
        sensitivity to be exceeded when it is struck)."""
        if self.fired or self.fuze not in ("contact", "delay"):
            return False
        if energy_j < self.spec.impact_sensitivity_j and not self.armed:
            return False
        self.disturbed = True
        if self.fuze == "delay" and self._delay_left is None:
            self._delay_left = self.delay_s
        return True

    def step_fuze(self, dt: float) -> bool:
        """Advance the fuze; True when it fires this tick."""
        if self.fired or not self.armed:
            return False
        self.elapsed_s += dt
        if self.fuze == "timer":
            return self.elapsed_s >= self.timer_s
        if self.fuze == "contact":
            return self.disturbed
        if self.fuze == "delay" and self._delay_left is not None:
            self._delay_left -= dt
            return self._delay_left <= 0.0
        return False

    def describe(self) -> str:
        cased = f", {self.casing_kg * 1000:.0f} g casing" if self.casing_kg > 0 else ", bare"
        where = self.attached_to or (f"at {tuple(round(v, 2) for v in self.position)}" if self.position else "unplaced")
        return (f"{self.identity}: {self.mass_kg * 1000:.0f} g {self.explosive} ({self.tnt_kg * 1000:.0f} g TNT eq){cased}, "
                f"{self.fuze} fuze, on {where}" + (" [FIRED]" if self.fired else (" [armed]" if self.armed else "")))


@dataclass
class BlastHit:
    part: str
    range_m: float
    incident_pa: float
    reflected_pa: float
    impulse_pa_s: float
    plate_velocity_m_s: float
    failure_velocity_m_s: float
    failed: bool


@dataclass
class OrdnanceField:
    charges: list = field(default_factory=list)
    log: list = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))

    def place(self, charge: Charge, arm: bool = True) -> Charge:
        charge.identity = charge.identity if charge.identity != "charge" else f"charge_{len(self.charges) + 1}"
        if arm:
            charge.arm()
        self.charges.append(charge)
        self.log.append(f"placed {charge.describe()}")
        return charge

    def step(self, dt: float, sim) -> list:
        """Advance every fuze; detonate what fires."""
        fired = []
        for c in list(self.charges):
            if c.step_fuze(dt):
                fired.append(self.detonate(c, sim))
        return fired

    def detonate(self, charge: Charge, sim) -> dict:
        """Fire one charge into the live engine."""
        if charge.fired:
            return {}
        charge.fired = True
        graph = sim._drivetrain.graph
        centre = charge.where(graph)
        w = charge.tnt_kg
        hits = blast_on_graph(graph, centre, w, skip=set(getattr(sim.state, "absent_parts", ())),
                              ceiling_pa=charge.spec.cj_pressure_pa)
        failed = [h for h in hits if h.failed]
        report = {"charge": charge.identity, "tnt_kg": w, "centre": tuple(float(v) for v in centre),
                  "hits": hits, "failed": [h.part for h in failed], "fragments": 0}
        self.log.append(f"{charge.identity} detonated: {charge.mass_kg * 1000:.0f} g {charge.explosive} "
                        f"({w * 1000:.0f} g TNT eq) at {charge.attached_to or 'a point'}; "
                        f"{len(failed)} parts failed of {len(hits)} loaded")
        # the sound first: the blast itself, sized by the real energy
        sim.damage_events.append({"kind": "blast", "energy_j": w * TNT_J_PER_KG, "part": charge.attached_to or charge.identity,
                                  "volume_m3": 0.05})
        # the fireball: a real ignition source for anything flammable it
        # reaches (the sim decides what that means -- see ignite_within)
        fireball_m = charge.spec.fireball_radius_per_kg13_m * (w ** (1.0 / 3.0))
        report["fireball_m"] = fireball_m
        self._make_fireball(sim, centre, fireball_m, charge)
        # the parts the blast destroyed: each goes through the existing
        # burst path (absent, contents out, its own fragments, an open
        # end on its circuit, and its own cascade). A fuel part inside
        # the fireball is lit; outside it, it just splits and pours.
        for h in failed:
            try:
                sim.burst_part(h.part, ignited=(h.range_m <= fireball_m))
            except Exception as exc:                     # a part with no node of its own
                self.log.append(f"  {h.part}: could not burst ({exc})")
        lit = sim.ignite_within(centre, fireball_m, f"{charge.identity} fireball")
        if lit:
            self.log.append(f"  fireball {fireball_m:.2f} m lit: {', '.join(p.split('.')[-1] for p in lit)}")
        # casing fragments through the projectile engine
        if charge.casing_kg > 0.0:
            report["fragments"] = self._throw_casing(charge, sim, centre)
        return report

    def _make_fireball(self, sim, centre: np.ndarray, radius_m: float, charge: Charge) -> None:
        """The luminous detonation products as a real expanding cloud,
        handed to the burst field so it flies, slows and settles with
        everything else: products leave at roughly a third of the
        filling's detonation velocity and decelerate against the air,
        and the soot behind them lingers."""
        from burst import BurstCloud, PartBurst
        v0 = charge.spec.detonation_m_s / 3.0
        n = 260
        dirs = self.rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        # a shell of products already a little way out at first sight
        r0 = radius_m * self.rng.uniform(0.02, 0.25, n)[:, None]
        speeds = v0 * self.rng.uniform(0.25, 1.0, n)[:, None]
        flame = BurstCloud(fluid="flame", pos=centre[None, :] + dirs * r0, vel=dirs * speeds,
                           mass=np.full(n, 1e-5), radius=np.full(n, max(0.01, radius_m * 0.06)),
                           settled=np.zeros(n, bool), regime="burst", character="atomised-mist")
        m = 90
        sdirs = self.rng.normal(size=(m, 3)); sdirs /= np.linalg.norm(sdirs, axis=1)[:, None]
        soot = BurstCloud(fluid="soot", pos=centre[None, :] + sdirs * (radius_m * 0.3), vel=sdirs * (v0 * 0.08),
                          mass=np.full(m, 1e-6), radius=np.full(m, max(0.02, radius_m * 0.12)),
                          settled=np.zeros(m, bool), regime="burst", character="atomised-mist")
        pts = [n_["reference_position"] for n_ in sim._drivetrain.graph["nodes"] if n_.get("reference_position") is not None]
        floor_y = min(p[1] for p in pts) - 0.25 if pts else -0.5
        b = PartBurst(part=f"{charge.identity}.fireball", position=tuple(float(v) for v in centre),
                      energy_j=charge.tnt_kg * TNT_J_PER_KG, floor_y=floor_y, max_speed_m_s=float(v0))
        b.clouds = [flame, soot]
        sim.bursts.bursts.append(b)

    def _throw_casing(self, charge: Charge, sim, centre: np.ndarray) -> int:
        """The casing at its Gurney velocity, as real projectiles."""
        if sim.ray_mesh_factory is None:
            return 0
        from ballistics import ProjectileState
        from engine_rays import Ray
        spec = charge.spec
        v0 = gurney_velocity_m_s(spec, charge.mass_kg, charge.casing_kg)
        if v0 <= 0.0:
            return 0
        # a Mott-style spread: many small pieces, a few large ones
        n = int(np.clip(charge.casing_kg * 200.0, 8, 60))
        raw = self.rng.lognormal(0.0, 0.9, n)
        masses = raw / raw.sum() * charge.casing_kg
        dirs = self.rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        speeds = v0 * self.rng.uniform(0.75, 1.15, n)
        rm = sim.ray_mesh_factory()
        sim._cascade_depth += 1
        fired = 0
        try:
            for i in range(n):
                m = float(masses[i])
                r = (3.0 * m / (4.0 * math.pi * 7850.0)) ** (1.0 / 3.0)   # a steel casing piece
                e = 0.5 * m * speeds[i] ** 2
                if e < 25.0:
                    continue
                ps = ProjectileState(mass_kg=m, diameter_m=2.0 * r, speed_m_s=float(speeds[i]),
                                     direction=tuple(float(x) for x in dirs[i]), length_m=2.5 * r,
                                     yaw_rad=float(self.rng.uniform(0.2, math.pi / 2)),
                                     tumble_rad_s=float(self.rng.uniform(50.0, 400.0)),
                                     hardness_pa=1.2e9, integrity=0.9)
                start = centre + dirs[i] * 0.02
                pen = rm.penetrate(Ray.from_points(start, start + dirs[i]), ps.energy_j, 2.0 * r,
                                   projectile=ps, fluid_by_part=sim.ballistic_fluid_for_part)
                rec = sim.apply_penetration(pen)
                fired += 1
                if rec:
                    sim.cascade_log.append(
                        f"casing {m * 1000:.0f} g at {speeds[i]:.0f} m/s ({e:.0f} J): "
                        + "; ".join(f"{ident.split('.')[-1]} {q.damage_mode}" for ident, q in rec))
        finally:
            sim._cascade_depth -= 1
        return fired

    def summary(self) -> list[str]:
        return [f"  ORDNANCE {c.describe()}" for c in self.charges]


# ---------------------------------------------------------------------------
# the blast field over the graph
# ---------------------------------------------------------------------------

# a part's own impulsive-failure scale: sqrt(2 * toughness / density),
# the velocity a plate of that material cannot be given without tearing
FAILURE_VELOCITY_FLOOR_M_S = 8.0


def blast_on_graph(graph: dict, centre: np.ndarray, w_tnt_kg: float, skip: set | None = None,
                   ceiling_pa: float = 0.0) -> list[BlastHit]:
    """Every part the blast loads, and whether it survives.

    The impulse lands on the part's own WALL, not on its whole mass:
    a thin-walled box (the material profile's declared shell_wall_m)
    presents rho*t kilograms per square metre of skin, however heavy
    the whole assembly is -- which is why a quarter-kilo of plastic on
    an alloy plenum opens it while the same charge only rings a solid
    cast-iron block. Failure is the classic impulsive-loading
    criterion: the imparted skin velocity i_s/(rho t) against the
    material's own fracture velocity sqrt(2 U / rho). `ceiling_pa` is
    the filling's CJ pressure -- the free-air relations are not valid
    inside the charge, and nothing can push harder than that."""
    from ballistics import material_profile
    skip = skip or set()
    hits: list[BlastHit] = []
    for n in graph.get("nodes", ()):
        ident = n["identity"]
        pos = n.get("reference_position")
        if pos is None or ident in skip:
            continue   # a body-side part (the tank, the radiator) is still a real part a blast reaches
        mass = float(n.get("mass_kg", 0.0) or 0.0)
        if mass <= 0.0:
            continue
        half = n.get("body_half_extent_m")
        if n.get("drum_radius_m"):
            r = float(n["drum_radius_m"])
            half = [float(n.get("drum_length_m", 2 * r)) / 2.0, r, r]
        if not half:
            continue
        half = np.asarray(half, dtype=float)
        c = np.asarray(pos, dtype=float)
        d = float(np.linalg.norm(c - centre))
        # the largest face is what the blast pushes on
        area = float(max(4.0 * half[0] * half[1], 4.0 * half[1] * half[2], 4.0 * half[0] * half[2]))
        if area <= 0.0:
            continue
        r_eff = max(d - float(np.min(half)), 0.02)
        inc = overpressure_pa(w_tnt_kg, r_eff)
        if ceiling_pa > 0.0:
            inc = min(inc, ceiling_pa)
        if inc < 5_000.0:
            continue
        refl = reflected_pa(inc)
        i_s = specific_impulse_pa_s(w_tnt_kg, r_eff) * (refl / max(inc, 1.0))
        prof = material_profile(_material_key(n))
        # what the impulse actually accelerates: the skin of a shell, or
        # the whole lump when the part is solid
        areal_solid = mass / area
        wall = prof.shell_wall_m or 0.0
        areal = min(areal_solid, prof.density_kg_m3 * wall) if wall > 0.0 else areal_solid
        v_plate = i_s / max(areal, 0.05)
        v_fail = max(FAILURE_VELOCITY_FLOOR_M_S, math.sqrt(2.0 * prof.toughness_j_m3 / max(prof.density_kg_m3, 1.0)))
        hits.append(BlastHit(ident, d, inc, refl, i_s, v_plate, v_fail, v_plate >= v_fail))
    hits.sort(key=lambda h: h.range_m)
    return hits


def _material_key(node: dict) -> str:
    """The ballistics material profile that fits this node's declared
    material (the same names damage/ballistics already use)."""
    m = str(node.get("material", "") or "").lower()
    ident = node["identity"]
    if "cast-iron" in m or "crankcase" in ident or "engine" == ident.split(".")[-1]:
        return "case"
    if "aluminium" in m:
        return "cover"
    if "pressed-steel" in m or "steel" in m:
        return "fuel"          # the profile for a steel vessel wall
    if "exhaust" in ident:
        return "exhaust"
    if "intake" in ident or "plenum" in ident or "filter" in ident:
        return "intake"
    if "plastic" in m or "nylon" in m:
        return "ignition"
    return "cover"


def describe_catalogue() -> list[str]:
    return [f"{e.name:14s} TNTeq {e.tnt_equivalence:.2f}  Gurney {e.gurney_m_s:4.0f} m/s  VoD {e.detonation_m_s:4.0f} m/s  "
            f"impact {e.impact_sensitivity_j:5.0f} J  {e.note}" for e in EXPLOSIVES.values()]
