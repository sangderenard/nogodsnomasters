"""A part bursting: where its material and its contents go.

`burst_part(sim, identity, energy_j)` takes a graph node -- its mass,
its material, its fluid and volume, its position and size -- and turns
it into a `PartBurst`: the part is ABSENT from then on (the view drops
its triangles, its fluid leaves its circuit, an open end stays on the
circuit as a hole emitter), and its material and contents fly out as
fragments whose speed and range are set by the released energy and by
what they have to fly THROUGH:

  - the energy: a fraction of it (SOLID_ENERGY_FRAC) accelerates the
    casing fragments, a smaller one the fluid, the rest is heat/sound;
    fragment speed from the Gurney-style equal share, v = sqrt(2 E_f /
    M_f), jittered per fragment
  - the directions: sampled over the sphere and WEIGHTED BY THE DENSITY
    OF THE SURROUNDING SPACE -- a fragment starts far more readily into
    air (1.2 kg/m^3) than into a bath of oil (870) or into another
    casting (thousands): weight = rho_air / rho(dir); its launch speed
    is scaled by sqrt of the same ratio (the medium's inertia takes the
    rest as a pressure pulse)
  - the range: every fragment flies under gravity with quadratic drag
    in whatever medium it is in at each step, a = rho_med Cd A v^2 /
    (2 m): a 20 g steel chip in air carries metres, in oil centimetres;
    a fuel droplet stops within a hand's width of air. The bay floor
    (below the engine's own envelope) settles what reaches it.

`medium_density_at(point, graph)` is the density lookup: inside a node
that holds fluid -> that fluid's density; inside a node that is a solid
casting -> its material's density (a wall, effectively a stop); else
the bay air. `BurstField` steps every live burst and hands the view its
clouds (the same droplets() interface hole emitters use).

The real trigger built in (EngineCycleSim._check_burst_triggers): fuel
spraying from a punctured fuel part whose stream reaches an exhaust
part hotter than the fuel's autoignition temperature lights, and after
a short exposure the fuel part deflagrates -- energy = the fuel it
still holds (capped) x LHV x an open-air deflagration fraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from hole_emitters import DENSITY_KG_M3, G, AIR_DENSITY_KG_M3, flow_character, fluid_key

SOLID_ENERGY_FRAC = 0.35
FLUID_ENERGY_FRAC = 0.15
DRAG_CD = 0.9
SOLID_DENSITY = {"cast-iron-casting": 7100.0, "steel-pipe": 7800.0, "steel": 7800.0, "aluminium-casting": 2700.0,
                 "aluminium": 2700.0, "plastic": 1200.0, "rubber": 1100.0, "brass": 8400.0, "copper": 8900.0}
FUEL_LHV_J_KG = 44.0e6
FUEL_DEFLAGRATION_FRAC = 0.02        # open-air fuel-vapour deflagration: a few percent of LHV as blast
FUEL_BURST_CAP_KG = 0.05             # the vapour cloud that actually takes part
FUEL_AUTOIGNITION_K = {"fuel": 520.0}
FUEL_EXPOSURE_S = 0.6                # how long the spray must wet the hot part before it lights

# A real pressure vessel letting go ("high-pressure-canister" nodes: the
# pneumatic reserve and brake reservoirs, a nitrous bottle, a WMI tank,
# a gasholder, a fuel tank). Every one of them already declares what it
# holds -- capacity_kg, its own working/bottle pressure, its envelope --
# so the released energy is read, not invented:
#   a compressed GAS tank stores the isothermal expansion work of its
#   charge, W = m R T ln(p/p_atm) (conservative: the adiabatic figure
#   for air is ~30 % lower, the isothermal is the textbook upper bound
#   for a slow vessel failure and is what pressure-vessel safety work
#   quotes as the "air energy" of a receiver);
#   a LIQUEFIED gas (nitrous at 6.2 MPa is saturated liquid N2O) is a
#   BLEVE: the liquid flashes, and only the disclosed fraction below is
#   taken as mechanical work -- the rest goes to vapour heat.
R_AIR_J_PER_KG_K = 287.0
R_N2O_J_PER_KG_K = 188.9
BLEVE_WORK_FRAC = 0.10               # of the flash enthalpy that becomes blast; real BLEVE yields are a few percent
N2O_FLASH_ENTHALPY_J_KG = 3.8e5      # latent heat + sensible drop to 1 atm, saturated N2O
LIQUEFIED_BOTTLE_PA = 3.0e6          # above this a "bottle" of that size is holding a liquefied gas, not compressed air


def vessel_energy_j(node: dict, fill_frac: float = 1.0) -> float:
    """The stored energy a pressure vessel releases when it lets go,
    from its own declared charge and pressure. 0 if it declares none.

    What it holds decides which physics applies, and a LIQUID is the
    important distinction: a tank of water-methanol at 17 bar stores
    almost nothing (liquid is very nearly incompressible -- its stored
    energy is p^2 V / 2K, joules, not kilojoules), while the same
    pressure in compressed AIR is a real bomb, and a liquefied gas
    (nitrous at 62 bar is saturated liquid N2O) is a BLEVE."""
    cap = float(node.get("capacity_kg", 0.0) or 0.0) * max(0.0, min(1.0, fill_frac))
    p = float(node.get("bottle_pressure_pa", 0.0) or node.get("working_pressure_pa", 0.0) or 0.0)
    if cap <= 0.0 or p <= 1.2 * 101_325.0:
        return 0.0
    key = fluid_key(node.get("fluid")) or fluid_key(node.get("tank_kind")) or _contents_guess(node)
    if key in ("water", "coolant", "fuel", "engine-oil"):
        if p < LIQUEFIED_BOTTLE_PA:
            # an ordinary pressurised LIQUID tank: the elastic energy in
            # the liquid itself, p^2 V / (2 K) with K ~ 2.2 GPa -- real,
            # and genuinely negligible (a few joules). It fails by
            # splitting and dumping its contents, not by blast.
            vol_m3 = cap / DENSITY_KG_M3.get(key, 900.0)
            return p * p * vol_m3 / (2.0 * 2.2e9)
        return cap * N2O_FLASH_ENTHALPY_J_KG * BLEVE_WORK_FRAC
    if p >= LIQUEFIED_BOTTLE_PA:
        # a liquefied charge: the flash, not a gas expansion
        return cap * N2O_FLASH_ENTHALPY_J_KG * BLEVE_WORK_FRAC
    return cap * R_AIR_J_PER_KG_K * 293.15 * math.log(p / 101_325.0)


def _contents_guess(node: dict) -> str:
    """What an undeclared canister holds, from what it is called: the
    pneumatic/brake vessels are air, a nitrous bottle liquefied N2O, a
    water/methanol or fuel tank a liquid."""
    n = node.get("identity", "")
    if any(k in n for k in ("pneumatic", "brake_reservoir", "air", "receiver")):
        return "gas"
    if "nitrous" in n:
        return "nitrous"
    if any(k in n for k in ("water", "auxiliary_injection")):
        return "water"
    if "fuel" in n or "gasholder" in n:
        return "fuel"
    return "gas"


def vessel_contents_kg(node: dict, fill_frac: float = 1.0) -> float:
    return float(node.get("capacity_kg", 0.0) or 0.0) * max(0.0, min(1.0, fill_frac))


def contents_key(node: dict) -> str | None:
    """What a node actually holds, from its declared fluid when it has
    one and from what it is otherwise -- the one place that guess is
    made, so bursts, fires and emitters all agree."""
    return fluid_key(node.get("fluid")) or fluid_key(node.get("tank_kind")) or (
        _contents_guess(node) if node.get("kind") == "high-pressure-canister" or node.get("capacity_kg") else None)


# A tank fire is not a detonation: what can go off in one bang is the
# flammable VAPOUR standing in the ullage (the empty part of the tank),
# and a stoichiometric hydrocarbon/air mixture carries ~3.5 MJ per cubic
# metre. A brim-full tank has almost no ullage and barely pops; a nearly
# empty one holds the dangerous volume. The liquid itself then burns as
# a pool fire, which is not a blast and is not modelled as one here.
VAPOUR_ENERGY_J_PER_M3 = 3.5e6
VAPOUR_BLAST_FRAC = 0.15             # of that which becomes blast rather than a slow flame front


def fuel_vapour_energy_j(node: dict, fill_frac: float = 1.0, liquid_density: float = 745.0) -> float:
    """The bang an ignited fuel volume actually makes."""
    cap = float(node.get("capacity_kg", 0.0) or 0.0)
    vol_l = float(node.get("fluid_volume_l", 0.0) or 0.0)
    total_m3 = (cap / liquid_density) if cap > 0.0 else vol_l / 1000.0
    if total_m3 <= 0.0:
        return 0.0
    ullage_m3 = total_m3 * (1.0 - max(0.0, min(1.0, fill_frac)))
    return ullage_m3 * VAPOUR_ENERGY_J_PER_M3 * VAPOUR_BLAST_FRAC


def solid_density(material: str | None) -> float:
    if not material:
        return 3500.0
    for k, v in SOLID_DENSITY.items():
        if k in material:
            return v
    return 3500.0


def medium_density_at(points: np.ndarray, graph: dict, exclude: str = "") -> np.ndarray:
    """rho (kg/m^3) at each point: what a fragment is inside of."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    rho = np.full(len(pts), AIR_DENSITY_KG_M3)
    for n in graph.get("nodes", ()):
        if n["identity"] == exclude or n.get("chassis_side") or n.get("kind") in ("powertrain-mount", "engine-block-port"):
            continue
        pos = n.get("reference_position")
        if pos is None:
            continue
        c = np.asarray(pos, dtype=float)
        if n.get("drum_radius_m"):
            r = float(n["drum_radius_m"]); half = np.array([float(n.get("drum_length_m", r)) / 2.0, r, r])
        else:
            half = np.asarray(n.get("body_half_extent_m", (0.0, 0.0, 0.0)), dtype=float)
            if not half.any():
                continue
        inside = np.all(np.abs(pts - c[None, :]) <= half[None, :], axis=1)
        if not inside.any():
            continue
        if float(n.get("fluid_volume_l", 0.0) or 0.0) > 0.0 and n.get("fluid"):
            key = fluid_key(n["fluid"])
            d = DENSITY_KG_M3.get(key, AIR_DENSITY_KG_M3 if key == "gas" else 900.0)
        else:
            d = solid_density(n.get("material"))
        rho[inside] = np.maximum(rho[inside], d)
    return rho


@dataclass
class BurstCloud:
    """One family of fragments from one burst (the casing, or one
    fluid). Duck-typed like a hole emitter for the view: regime,
    fluid, radius_m, character, droplets()."""
    fluid: str                  # "solid" or a fluid name
    pos: np.ndarray             # (n, 3)
    vel: np.ndarray             # (n, 3)
    mass: np.ndarray            # (n,)
    radius: np.ndarray          # (n,)
    settled: np.ndarray         # (n,) bool
    regime: str = "burst"
    character: str = "turbulent-droplets"

    @property
    def radius_m(self) -> float:
        return float(np.mean(self.radius)) if len(self.radius) else 0.003

    @property
    def alive(self) -> int:
        return int((~self.settled).sum())

    def droplets(self, rng, n: int = 400, t_max_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        idx = np.arange(len(self.pos)) if len(self.pos) <= n else rng.choice(len(self.pos), n, replace=False)
        return self.pos[idx], np.where(self.settled[idx][:, None], 0.0, self.vel[idx])


@dataclass
class PartBurst:
    part: str
    position: tuple
    energy_j: float
    clouds: list = field(default_factory=list)
    age_s: float = 0.0
    floor_y: float = -1.0
    max_speed_m_s: float = 0.0
    max_range_m: float = 0.0

    def step(self, dt: float, graph: dict) -> None:
        self.age_s += dt
        for c in self.clouds:
            live = ~c.settled
            if not live.any():
                continue
            p, v, m, r = c.pos[live], c.vel[live], c.mass[live], c.radius[live]
            rho = medium_density_at(p, graph, exclude=self.part)
            speed = np.linalg.norm(v, axis=1)
            area = math.pi * r * r
            a_drag = 0.5 * rho * DRAG_CD * area * speed * speed / np.maximum(m, 1e-9)
            dv = -v / np.maximum(speed, 1e-9)[:, None] * np.minimum(a_drag, speed / max(dt, 1e-6))[:, None] * dt
            g = np.array([0.0, -G, 0.0]) if c.fluid != "gas" else np.zeros(3)
            v = v + dv + g[None, :] * dt
            p = p + v * dt
            # a wall (a solid medium) stops a fragment where it is; the
            # floor settles whatever reaches it
            hit_wall = rho > 2000.0
            below = p[:, 1] <= self.floor_y
            stop = hit_wall | below | (np.linalg.norm(v, axis=1) < 0.02)
            p[below, 1] = self.floor_y
            v[stop] = 0.0
            c.pos[live] = p; c.vel[live] = v
            idx = np.nonzero(live)[0]
            c.settled[idx[stop]] = True
            self.max_range_m = max(self.max_range_m, float(np.max(np.linalg.norm(p - np.asarray(self.position)[None, :], axis=1))))

    @property
    def alive(self) -> int:
        return sum(c.alive for c in self.clouds)


def _lognormal_masses(rng, total_kg: float, n: int, sigma: float = 0.8) -> np.ndarray:
    raw = rng.lognormal(0.0, sigma, n)
    return raw / raw.sum() * total_kg


def make_burst(graph: dict, identity: str, energy_j: float, rng, floor_y: float,
               fluid_kg_override: float | None = None) -> PartBurst:
    node = next(n for n in graph["nodes"] if n["identity"] == identity)
    pos = np.asarray(node["reference_position"], dtype=float)
    mass_kg = float(node.get("mass_kg", 1.0) or 1.0)
    material = node.get("material")
    fluid = node.get("fluid")
    fkey = fluid_key(fluid)
    if fluid_kg_override is not None:
        fluid_kg = fluid_kg_override
    elif node.get("kind") == "high-pressure-canister":
        # a real declared charge (capacity_kg), not a geometric volume
        fluid_kg = vessel_contents_kg(node, float(node.get("fill_level_frac", 1.0) or 1.0))
        if fkey is None:
            fkey = "fuel" if "fuel" in identity else "gas"
    else:
        fluid_kg = float(node.get("fluid_volume_l", 0.0) or 0.0) / 1000.0 * DENSITY_KG_M3.get(fkey, 1.2)
    half = np.asarray(node.get("body_half_extent_m", (0.03, 0.03, 0.03)), dtype=float)
    if node.get("drum_radius_m"):
        half = np.array([float(node.get("drum_length_m", 0.05)) / 2.0, float(node["drum_radius_m"]), float(node["drum_radius_m"])])
    size = float(np.linalg.norm(half))
    burst = PartBurst(part=identity, position=tuple(pos), energy_j=float(energy_j), floor_y=floor_y)

    def launch(n: int, total_kg: float, energy_frac: float, density: float, spread_pos: float):
        masses = _lognormal_masses(rng, total_kg, n)
        radii = (3.0 * masses / (4.0 * math.pi * density)) ** (1.0 / 3.0)
        dirs = rng.normal(size=(n, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        probe = pos[None, :] + dirs * (size + 0.03)
        rho_dir = medium_density_at(probe, graph, exclude=identity)
        # the surrounding density decides who goes where: resample
        # directions with weight rho_air / rho, and slow what does go into
        # something dense
        w = AIR_DENSITY_KG_M3 / rho_dir
        w = w / w.sum()
        pick = rng.choice(n, n, p=w)
        dirs, rho_dir = dirs[pick], rho_dir[pick]
        e_f = energy_j * energy_frac
        v0 = math.sqrt(2.0 * e_f / max(total_kg, 1e-6))
        speeds = v0 * rng.uniform(0.7, 1.3, n) * np.sqrt(AIR_DENSITY_KG_M3 / rho_dir)
        start = pos[None, :] + dirs * rng.uniform(0.0, spread_pos, n)[:, None]
        return BurstCloud(fluid="solid", pos=start, vel=dirs * speeds[:, None], mass=masses, radius=radii,
                          settled=np.zeros(n, bool)), float(np.max(speeds))

    n_solid = int(np.clip(12 + mass_kg * 6.0, 12, 90))
    solid, vmax = launch(n_solid, mass_kg * 0.6, SOLID_ENERGY_FRAC, solid_density(material), size * 0.5)
    solid.character = "turbulent-droplets"
    burst.clouds.append(solid)
    burst.max_speed_m_s = vmax
    if fluid_kg > 1e-4 and fkey:
        fname = fkey
        is_liquid = fname in DENSITY_KG_M3
        cloud, vf = launch(int(np.clip(fluid_kg * 4000.0, 60, 400)), fluid_kg, FLUID_ENERGY_FRAC,
                           DENSITY_KG_M3.get(fname, 900.0) if is_liquid else 1.2, size * 0.8)
        cloud.fluid = fname if is_liquid else "gas"
        cloud.radius = np.full(len(cloud.radius), 0.0012 if is_liquid else 0.004)
        cloud.character = flow_character(cloud.fluid, vf, 0.002)
        burst.clouds.append(cloud)
        burst.max_speed_m_s = max(burst.max_speed_m_s, vf)
    return burst


@dataclass
class BurstField:
    bursts: list = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(99))

    def step(self, dt: float, graph: dict) -> None:
        for b in self.bursts:
            b.step(dt, graph)

    def clouds(self) -> list:
        return [c for b in self.bursts for c in b.clouds if len(c.pos)]

    def summary(self) -> list[str]:
        return [f"  BURST {b.part.split('.')[-1]}: {b.energy_j / 1000:.1f} kJ, {sum(len(c.pos) for c in b.clouds)} fragments, "
                f"fastest {b.max_speed_m_s:.0f} m/s, reached {b.max_range_m:.2f} m, {b.alive} still flying"
                for b in self.bursts]
