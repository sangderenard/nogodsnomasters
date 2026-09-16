"""Fluid ON surfaces: how much clings, where it runs, how fast it dries.

hole_emitters has been producing wetting deposits for a consumer that
was never written. Its own comment says so -- `wetting_deposits` is
documented as "every point taking fluid right now, FOR A WETTING
ENGINE", and `splash_deposits` says a wetting engine that can take one
can take the other. This is that engine.

The unit of wetness is the TRIANGLE, and that is not an implementation
convenience -- it is the physics. A triangle has an area, so it can hold
a mass; it has a normal, so gravity resolves against it into a part that
presses the film onto the surface and a part that drags it along; and it
has neighbours, so what it cannot hold goes somewhere real instead of
disappearing.

HOW MUCH A SURFACE HOLDS, and why a ceiling drips but a floor puddles.

    A liquid film clings by surface tension and is pulled off by
    gravity. Those balance at the CAPILLARY LENGTH,

        l_c = sqrt(sigma / (rho . g))

    which for engine oil is about 1.9 mm and for petrol about 1.7 mm --
    the real reason a spill beads to roughly that depth and no more, on
    any surface, in any garage.

    The surface's tilt decides how much of that it actually keeps. With
    n the triangle's unit normal and up the vertical,

        n.up = +1   floor, facing up      holds a full capillary depth
        n.up =  0   vertical wall         holds only a clinging skin
        n.up = -1   ceiling, facing down  holds least, and drips

    so retention scales with the upward component of the normal, and the
    remainder is free to move. No triangle needs to be told what kind of
    surface it is; its own normal already says.

WHERE THE EXCESS GOES. Downhill, which is exactly the gravity vector
with the part along the normal removed:

    down_slope = g - (g.n) n,    normalised

That is an exact projection, not a heuristic, and it is zero on a
horizontal surface -- which is correct, because a puddle on a flat floor
does not run anywhere. What reaches a triangle's lower edge with nothing
below it LEAVES: it becomes a drip, and a drip is an emitter, so it goes
back to hole_emitters rather than vanishing. Re-emission closes the loop.

HOW FAST IT DRIES, in the only two regimes that matter.

  - ABOVE THE FLUID'S BOILING POINT the surface can supply heat faster
    than the film can absorb it, so drying is HEAT LIMITED and nothing
    else: every joule arriving boils its own worth of liquid.

        mdot = h.A.(T_surface - T_boil) / L

    This is why petrol flashes off a hot manifold and diesel sits there:
    petrol's T50 is 373 K and diesel's is 533 K, so the same exhaust
    pipe is above one and below the other. working_fluids carries both,
    and engine_cycle_sim is already asking a related question one step
    further along -- whether the surface is hot enough to IGNITE what
    landed on it.

  - BELOW IT, evaporation is limited by how fast vapour can diffuse
    away, which is slower by orders of magnitude and falls off steeply
    as the surface cools. Oil, whose boiling point is declared as 0.0
    because it cracks before it distils, never leaves this regime: it
    does not dry, it only runs off or burns. That asymmetry is why an
    oil leak stays a mess and a petrol spill does not.

WHAT THIS IS NOT. There is no "wetness" score and no decay constant. A
triangle holds a mass of a named fluid, that mass moves by projected
gravity, and it leaves by boiling at a rate set by the fluid's own
latent heat. Every quantity has a unit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Surface tension, N/m, at room temperature. Real measured values.
#: Oil and the light fuels are close enough that the capillary length
#: barely differs between them -- which is itself the reason a spill of
#: any of them beads to about the same depth.
#: FALLBACK ONLY. fluids.py is the registry and carries a surface
#: tension per row; this remains for anything not in it.
SURFACE_TENSION_N_PER_M = {
    "engine-oil": 0.032,
    "gear-oil": 0.033,
    "hydraulic-oil": 0.031,
    "water": 0.0728,
    "coolant": 0.050,
}
DEFAULT_SURFACE_TENSION_N_PER_M = 0.025      # light hydrocarbon fuels
GRAVITY_M_S2 = 9.80665

#: Convective coefficient from a hot surface into the film sitting on
#: it. engines.py already uses this same real natural-convection figure
#: for bare metal pipe in still air; a wetted surface is the same
#: exchange running the other way.
SURFACE_FILM_HTC_W_PER_M2K = 25.0

#: Below boiling, evaporation is diffusion limited. This is the fraction
#: of the heat-limited rate that a surface AT its boiling point achieves
#: by diffusion alone, and it falls off with the temperature deficit --
#: the standard exponential form, since vapour pressure follows
#: Clausius-Clapeyron and the flux follows the vapour pressure.
DIFFUSION_LIMITED_FRAC_AT_BOILING = 0.02

#: How much of a capillary depth a vertical or inverted surface still
#: clings to. A wall is not dry.
CLINGING_SKIN_FRAC_OF_CAPILLARY = 0.05

#: Steepness of the sub-boiling evaporation falloff. Vapour pressure
#: follows Clausius-Clapeyron so the flux is exponential in the
#: temperature deficit; this is that exponent's scale, and it is the
#: weakest number in this module -- the shape is right, the rate is not
#: anchored to a measured drying curve.
DIFFUSION_FALLOFF_EXPONENT = 8.0


def capillary_length_m(fluid: str, density_kg_m3: float) -> float:
    """sqrt(sigma / rho.g) -- the depth a spill beads to.

    Surface tension comes from fluids.py, which already carries one per
    row, rather than from the local table below. That table was a second
    copy of the same data and it did not know about the molten metals at
    all -- molten iron is 1.87 N/m against water's 0.073, so defaulting
    it to a light hydrocarbon's 0.025 would have spread a weld pool out
    like petrol instead of standing it up in a bead."""
    try:
        import fluids as _f
        row = _f.BY_KEY.get(fluid)
        if row is not None and row.surface_tension_n_m > 0.0:
            sigma = row.surface_tension_n_m
        else:
            sigma = SURFACE_TENSION_N_PER_M.get(fluid, DEFAULT_SURFACE_TENSION_N_PER_M)
    except Exception:
        sigma = SURFACE_TENSION_N_PER_M.get(fluid, DEFAULT_SURFACE_TENSION_N_PER_M)
    return math.sqrt(sigma / (max(density_kg_m3, 1.0) * GRAVITY_M_S2))


def retention_kg_per_m2(fluid: str, density_kg_m3: float, normal_up: float) -> float:
    """How much this surface keeps, per square metre.

    normal_up is n.up, from +1 (floor) through 0 (wall) to -1 (ceiling).
    A wall still keeps a thin clinging skin -- a vertical surface is not
    dry -- so the floor case scales the full capillary depth and the
    rest keeps only what surface tension alone can hold against a
    straight fall."""
    l_c = capillary_length_m(fluid, density_kg_m3)
    upward = max(0.0, float(normal_up))
    # the clinging skin a vertical or inverted surface keeps regardless:
    # one roughness-scale film, far below a capillary depth
    skin = CLINGING_SKIN_FRAC_OF_CAPILLARY * l_c
    return density_kg_m3 * (skin + (l_c - skin) * upward)


def downslope(normal: np.ndarray) -> np.ndarray:
    """Gravity with the surface-normal part removed: where a film runs.

    Exactly zero on a horizontal surface, which is why a puddle on a
    flat floor stays a puddle."""
    g = np.array([0.0, -1.0, 0.0], dtype=float) * GRAVITY_M_S2
    n = np.asarray(normal, dtype=float)
    nn = float(np.dot(n, n))
    if nn <= 1e-12:
        return np.zeros(3)
    n = n / math.sqrt(nn)
    slope = g - float(np.dot(g, n)) * n
    mag = float(np.linalg.norm(slope))
    return slope / mag if mag > 1e-9 else np.zeros(3)


def dry_rate_kg_s(fluid_spec, area_m2: float, surface_temp_k: float,
                  film_kg: float, htc: float = SURFACE_FILM_HTC_W_PER_M2K) -> float:
    """How fast this film leaves, in kg/s.

    Heat limited above the fluid's boiling point, diffusion limited
    below it, and identically zero for anything whose boiling point is
    declared 0.0 -- engine oil, which cracks before it distils and
    therefore never dries at all."""
    boil = float(getattr(fluid_spec, "boiling_point_k", 0.0) or 0.0)
    latent = float(getattr(fluid_spec, "latent_heat_j_per_kg", 0.0) or 0.0)
    if boil <= 0.0 or latent <= 0.0 or film_kg <= 0.0 or area_m2 <= 0.0:
        return 0.0
    t = float(surface_temp_k)
    if t >= boil:
        q = htc * area_m2 * (t - boil)
        return max(0.0, q / latent)
    # below boiling: the same heat-limited scale, cut by how far under
    # the boiling point the surface is. Clausius-Clapeyron makes vapour
    # pressure exponential in -1/T, so the flux is too.
    reference_q = htc * area_m2 * max(1.0, boil - t)
    deficit = (boil - t) / max(boil, 1.0)
    return max(0.0, reference_q / latent * DIFFUSION_LIMITED_FRAC_AT_BOILING
               * math.exp(-DIFFUSION_FALLOFF_EXPONENT * deficit))


@dataclass
class WetTriangle:
    """One triangle's wetness: what is on it and what that does."""
    index: int
    part: str
    area_m2: float
    normal: tuple
    fluid: str
    film_kg: float = 0.0
    temp_k: float = 293.15

    @property
    def normal_up(self) -> float:
        return float(self.normal[1])

    def retained_kg(self, density_kg_m3: float) -> float:
        return retention_kg_per_m2(self.fluid, density_kg_m3, self.normal_up) * self.area_m2

    def film_thickness_m(self, density_kg_m3: float) -> float:
        if self.area_m2 <= 0.0:
            return 0.0
        return self.film_kg / (max(density_kg_m3, 1.0) * self.area_m2)


@dataclass
class WettingField:
    """Every wet triangle on a machine, and the loop that keeps it real.

    Deposits in from hole_emitters, runoff between triangles, drips back
    out to hole_emitters as new emitters, drying to the air. Nothing is
    created and nothing is destroyed except by boiling, which is
    accounted in kilograms."""
    triangles: dict = field(default_factory=dict)      # index -> WetTriangle
    runoff_kg: float = 0.0                             # left the machine this step
    evaporated_kg: float = 0.0
    drips: list = field(default_factory=list)          # (position, fluid, kg) -> re-emission

    def deposit(self, entries, mesh=None) -> None:
        """Take hole_emitters.wetting_deposits / splash_deposits output.

        Exactly the contract those functions already declare:
        {part, position, normal, fluid, kg}. Nothing new is asked of the
        producer side, because the producer side was written for this."""
        for e in entries:
            idx = int(e.get("triangle", -1))
            if idx < 0:
                idx = self._nearest_triangle(e.get("position"), mesh)
            if idx < 0:
                continue
            t = self.triangles.get(idx)
            if t is None:
                n = e.get("normal") or (0.0, 1.0, 0.0)
                t = WetTriangle(index=idx, part=str(e.get("part", "?")),
                                area_m2=float(e.get("area_m2", 1e-4)),
                                normal=tuple(float(v) for v in n),
                                fluid=str(e.get("fluid", "engine-oil")))
                self.triangles[idx] = t
            t.film_kg += max(0.0, float(e.get("kg", 0.0)))

    def _nearest_triangle(self, position, mesh) -> int:
        if mesh is None or position is None:
            return -1
        centres = mesh.tri_vertices().mean(axis=1)
        d = centres - np.asarray(position, dtype=float)
        return int(np.argmin(np.einsum("ij,ij->i", d, d)))

    def step(self, dt: float, density_by_fluid: dict, fluid_specs: dict,
             neighbours: dict | None = None) -> None:
        """One tick: dry what can dry, then move what will not stay.

        Drying happens FIRST because a film that boils off never runs --
        on a hot surface that is the whole behaviour, and doing it the
        other way round would march fuel down an exhaust pipe that
        should have flashed it off where it landed."""
        for t in list(self.triangles.values()):
            rho = float(density_by_fluid.get(t.fluid, 850.0))
            spec = fluid_specs.get(t.fluid)
            if spec is not None:
                gone = dry_rate_kg_s(spec, t.area_m2, t.temp_k, t.film_kg) * dt
                gone = min(gone, t.film_kg)
                t.film_kg -= gone
                self.evaporated_kg += gone
            excess = t.film_kg - t.retained_kg(rho)
            if excess <= 0.0:
                continue
            t.film_kg -= excess
            below = (neighbours or {}).get(t.index)
            if below is None:
                # nothing under it: this leaves the machine as a drip,
                # and a drip is an emitter, not an accounting hole
                self.runoff_kg += excess
                self.drips.append((t.index, t.fluid, excess))
            else:
                nb = self.triangles.get(below)
                if nb is None:
                    self.runoff_kg += excess
                    self.drips.append((t.index, t.fluid, excess))
                else:
                    nb.film_kg += excess

    def total_kg(self) -> float:
        return sum(t.film_kg for t in self.triangles.values())

    def wettest(self, n: int = 5) -> list:
        return sorted(self.triangles.values(), key=lambda t: -t.film_kg)[:n]
