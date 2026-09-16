"""The combustion kernel: every cylinder's own burn, as light.

A baked "explosion sprite" per cylinder -- but baked in VOLUME, not
as a flat billboard: for each burn phase the kernel is the real gas
volume between the piston crown (placed by the same crank kinematics
cylinder_ports.py draws the piston with, at the crank angle that phase
actually occurs) and the head face, growing from a small core at the
igniter to the full bore as the flame front crosses the chamber. Its
colour is COMPOSED: a Planck blackbody term at the fuel's own flame
temperature, weighted by how sooty (luminous) that fuel burns, plus the
faint blue chemiluminescence a clean flame shows instead -- so a
diesel's burn is a yellow-white glare and methanol's is a pale blue
ghost, off the same two real terms. Residue is configurable per fuel
(CombustionVisual): a soot/oil-smoke/ash puff of a given colour and
opacity leaves the real exhaust port during the exhaust stroke, or
nothing does.

The frames are geometry only. Which frame each cylinder shows, and how
bright, is decided every tick from EngineCycleSim's OWN firing events
(combustion_state_from_sim): degrees since that cylinder last fired,
and the real strength it fired with (a misfire or a cut shows nothing,
a knocking cylinder shows a weakened burn). The renderer feeds the
frame's emission through the same live material layer the blackbody
glow uses (engine_gl_view._ThermalMaterialDB), and every live flame is
also a real point emitter -- the burn lights the crown and bore from
inside, which is the honest reason a running engine's pistons are
visible through a see-through block at all.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from engine_mesh import blackbody_emission_rgb, EngineMesh, _from_parts
from mesh_primitives import tube_mesh, capped_tube_mesh
from vehicle_mesh import SolidPart

N_BURN_PHASES = 8
N_SMOKE_PHASES = 4


@dataclass(frozen=True)
class CombustionVisual:
    """What a fuel's burn and its residue look like -- declared, per
    fuel family, overridable per engine (Engine.combustion_visual)."""
    flame_temp_k: float          # adiabatic-ish peak flame temperature driving the Planck term
    soot_luminosity: float       # 0 (clean, non-luminous) .. 1 (fully sooting, yellow blackbody flame)
    burn_duration_deg: float     # crank degrees the visible burn lasts
    residue_rgb: tuple           # smoke/ash colour (linear)
    residue_opacity: float       # 0 = leaves nothing visible
    residue_label: str


# real families, keyed by substrings of the fuel profile / engine kind:
#   diesel / multifuel  -- sooty, hot, black smoke under load
#   gasoline            -- moderately luminous, near-invisible exhaust
#   methanol / nitro    -- clean, pale-blue, next to nothing out the pipe
#   two-stroke premix   -- gasoline flame, blue-white oil smoke
#   coal gas / producer -- soft lean flame, thin grey
#   kerosene / jet      -- luminous, light grey
_VISUAL_BY_FAMILY = {
    "diesel": CombustionVisual(2300.0, 0.95, 60.0, (0.05, 0.05, 0.05), 0.55, "black soot"),
    "multifuel": CombustionVisual(2250.0, 0.9, 60.0, (0.08, 0.08, 0.08), 0.5, "dark soot"),
    "gasoline": CombustionVisual(2250.0, 0.55, 45.0, (0.55, 0.55, 0.6), 0.08, "faint grey"),
    "methanol": CombustionVisual(2100.0, 0.12, 40.0, (0.7, 0.7, 0.75), 0.02, "near-invisible"),
    "nitromethane": CombustionVisual(2400.0, 0.35, 35.0, (0.85, 0.75, 0.5), 0.12, "yellow nitro haze"),
    "two-stroke": CombustionVisual(2200.0, 0.55, 45.0, (0.6, 0.7, 0.9), 0.4, "blue-white oil smoke"),
    "coal-gas": CombustionVisual(2000.0, 0.2, 55.0, (0.6, 0.6, 0.62), 0.15, "thin grey"),
    "kerosene": CombustionVisual(2200.0, 0.7, 50.0, (0.5, 0.5, 0.52), 0.2, "light grey"),
    "propane": CombustionVisual(2200.0, 0.2, 45.0, (0.7, 0.7, 0.72), 0.02, "near-invisible"),
    "hydrogen": CombustionVisual(2400.0, 0.02, 35.0, (0.9, 0.9, 0.95), 0.0, "none (water vapour)"),
}
_DEFAULT_VISUAL = _VISUAL_BY_FAMILY["gasoline"]


def family_for(engine, fuel_profile: str | None = None) -> str:
    """WHICH combustion family this engine is burning in, by name.

    `visual_for` resolves the same question and hands back appearance;
    anything that needs the family itself -- how much soot the burn
    really makes, say, which is a mass and not a brightness -- asks for
    it here rather than reaching for a visual property and hoping it
    stands in."""
    prof = (fuel_profile or getattr(engine, "preferred_fuel_profile", "") or "").lower()
    arch = getattr(engine, "architecture", None)
    for key in ("diesel", "multifuel"):
        if key in prof:
            return key
    if arch is not None and getattr(arch, "two_stroke", False):
        return "two-stroke"
    for key in _VISUAL_BY_FAMILY:
        if key in prof:
            return key
    kind = (getattr(engine, "fuel_kind", "") or "").lower()
    for key in _VISUAL_BY_FAMILY:
        if key in kind:
            return key
    return "gasoline"


def visual_for(engine, fuel_profile: str | None = None) -> CombustionVisual:
    """The engine's own declared override first, else the fuel family
    the live fuel profile (or its preferred one) names."""
    declared = getattr(engine, "combustion_visual", None)
    if declared is not None:
        return declared
    prof = (fuel_profile or getattr(engine, "preferred_fuel_profile", "") or "").lower()
    arch = getattr(engine, "architecture", None)
    # a compression-ignition two-stroke (a big marine crosshead diesel)
    # is a diesel burn with diesel soot -- the "two-stroke" family here
    # means the premix oil-in-fuel kind, so the fuel family wins first
    for key in ("diesel", "multifuel"):
        if key in prof:
            return _VISUAL_BY_FAMILY[key]
    if arch is not None and getattr(arch, "two_stroke", False):
        return _VISUAL_BY_FAMILY["two-stroke"]
    for key, vis in _VISUAL_BY_FAMILY.items():
        if key in prof:
            return vis
    kind = (getattr(engine, "fuel_kind", "") or "").lower()
    for key, vis in _VISUAL_BY_FAMILY.items():
        if key in kind:
            return vis
    return _DEFAULT_VISUAL


def burn_shape(phase: float) -> float:
    """Relative heat-release intensity over the burn, 0..1: a fast rise
    as the flame front accelerates, a slower tail as it quenches at the
    walls -- the real shape of a Wiebe-style burn, not a flat pulse."""
    if phase <= 0.0 or phase >= 1.0:
        return 0.0
    rise = 1.0 - math.exp(-6.0 * (phase / 0.35) ** 2) if phase < 0.35 else 1.0
    tail = 1.0 if phase < 0.35 else math.exp(-3.5 * ((phase - 0.35) / 0.65) ** 1.5)
    return rise * tail


def flame_emission_rgb(vis: CombustionVisual, phase: float, strength: float) -> tuple[float, float, float]:
    """The composed colour: sooty blackbody at the (phase-scaled) flame
    temperature, plus the clean flame's blue chemiluminescence. Both
    scale with the real strength of this particular firing."""
    s = burn_shape(phase) * max(0.0, min(1.5, strength))
    if s <= 0.0:
        return (0.0, 0.0, 0.0)
    # the gas temperature follows the heat release: peak at full shape
    temp = 700.0 + (vis.flame_temp_k - 700.0) * (0.55 + 0.45 * burn_shape(phase))
    r, g, b = blackbody_emission_rgb(temp)
    soot = vis.soot_luminosity
    # a clean flame's CH/C2 bands: blue-violet, dim next to any soot
    chem = (1.0 - soot) * 0.9
    return (s * (soot * r + chem * 0.25), s * (soot * g + chem * 0.35), s * (soot * b + chem * 1.0))


def smoke_shape(phase: float) -> float:
    """Residue puff: appears as the exhaust valve opens, spreads, thins."""
    if phase <= 0.0 or phase >= 1.0:
        return 0.0
    return math.sin(math.pi * phase) ** 0.7


@dataclass
class CombustionFrames:
    """Baked per-cylinder frames: flame[cyl][k] for k in burn phases,
    smoke[cyl][k] for k in smoke phases (None when the fuel leaves no
    visible residue). Cylinder keys are the layout's own numbers."""
    flame: dict[int, list[EngineMesh]]
    smoke: dict[int, list[EngineMesh]]
    burn_duration_deg: float
    visual: CombustionVisual


def _chamber(g, crank_angle_deg: float):
    """(crown top point, head face point, bore radius, axis) for this
    cylinder at this crank angle -- the same kinematics cylinder_ports
    places the piston with, so the kernel sits exactly on the crown."""
    from cylinder_ports import piston_pin_distance_m, WALL_FRAC_OF_BORE
    axis = np.asarray(g.axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    base = np.asarray(g.base, dtype=np.float64)
    r = g.bore_m / 2.0
    wall = g.bore_m * WALL_FRAC_OF_BORE
    head_face = base + axis * (g.length_m - wall)
    if g.crank_radius_m > 0.0 and g.rod_length_m > 0.0:
        d = piston_pin_distance_m(g.crank_radius_m, g.rod_length_m, crank_angle_deg + g.throw_angle_deg)
        crown = np.asarray(g.crank_centre, dtype=np.float64) + axis * (d + r * 0.45)
    else:
        crown = base + axis * (wall + (g.length_m - 2.0 * wall) * 0.8)
    # never let the kernel poke through the head: clamp to a sliver
    h = float(np.dot(head_face - crown, axis))
    if h < r * 0.05:
        crown = head_face - axis * (r * 0.05)
    return crown, head_face, r * 0.96, axis


def bake_combustion_frames(layout, firing_angles_deg: dict[int, float], visual: CombustionVisual,
                           exhaust_ports: dict[int, tuple[np.ndarray, np.ndarray, float]] | None = None) -> CombustionFrames:
    """layout: cylinder_ports.deserialize_layout(graph["cylinder_layout"]);
    firing_angles_deg: EngineCycleSim._firing_angle_deg (cycle degrees
    at which each cylinder fires); exhaust_ports: cyl -> (position,
    outward direction, radius) for the residue puff, from the layout's
    own real exhaust PortSpec."""
    flame: dict[int, list[EngineMesh]] = {}
    smoke: dict[int, list[EngineMesh]] = {}
    from cylinder_ports import ATMOSPHERIC, ROTARY, EXPANDER
    for g, ports in layout:
        if g.kind in (ATMOSPHERIC, ROTARY, EXPANDER):
            # nothing burns in an expander's cylinder (steam / compressed
            # air -- the fire is in the boiler, not the bore); the rotary
            # and the atmospheric have no crown chamber to bake into (yet)
            continue
        fire = firing_angles_deg.get(g.number, 0.0)
        frames = []
        for k in range(N_BURN_PHASES):
            phase = (k + 0.5) / N_BURN_PHASES
            crank = fire + phase * visual.burn_duration_deg
            crown, head_face, r, axis = _chamber(g, crank)
            # the flame front: a small core at the head (where the
            # igniter is) growing to the full bore -- radius by the cube
            # root of burnt fraction, real for a spherical kernel
            grow = min(1.0, 0.12 + 0.88 * (max(phase, 0.02) / 0.45) ** (1.0 / 3.0))
            kr = r * grow
            top = head_face - axis * (r * 0.02)
            bottom = crown + (top - crown) * (1.0 - min(1.0, grow)) * 0.6 if grow < 1.0 else crown
            vtx, nrm = capped_tube_mesh(bottom, top, kr, sides=18)
            part = SolidPart(vertices=vtx, normals=nrm, thermal_group=f"flame_c{g.number}_p{k}", name=f"cyl{g.number}_flame_kernel_p{k}")
            frames.append(_from_parts([part], moving=True))
        flame[g.number] = frames
        if visual.residue_opacity > 0.0 and exhaust_ports and g.number in exhaust_ports:
            pos, direction, pr = exhaust_ports[g.number]
            direction = np.asarray(direction, dtype=np.float64)
            direction = direction / max(np.linalg.norm(direction), 1e-9)
            sframes = []
            for k in range(N_SMOKE_PHASES):
                phase = (k + 0.5) / N_SMOKE_PHASES
                travel = pr * (0.5 + 4.0 * phase)
                radius = pr * (0.6 + 1.6 * phase)
                start = np.asarray(pos, dtype=np.float64) + direction * (pr * 0.2)
                end = start + direction * travel
                vtx, nrm = capped_tube_mesh(start, end, radius, sides=12)
                part = SolidPart(vertices=vtx, normals=nrm, thermal_group=f"smoke_c{g.number}_p{k}", name=f"cyl{g.number}_smoke_puff_p{k}")
                sframes.append(_from_parts([part], moving=True))
            smoke[g.number] = sframes
    return CombustionFrames(flame=flame, smoke=smoke, burn_duration_deg=visual.burn_duration_deg, visual=visual)


def exhaust_ports_from_layout(layout) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    out = {}
    for g, ports in layout:
        for p in ports:
            if p.fluid_role == "exhaust":
                out[g.number] = (np.asarray(p.position, dtype=np.float64), np.asarray(p.direction, dtype=np.float64), float(p.radius_m))
                break
    return out


@dataclass(frozen=True)
class CylinderBurnState:
    cylinder: int
    burn_phase: float | None      # 0..1 while burning, None otherwise
    strength: float               # the real strength this firing had
    smoke_phase: float | None     # 0..1 during the exhaust stroke, None otherwise


def combustion_state_from_sim(sim, burn_duration_deg: float) -> list[CylinderBurnState]:
    """Read every cylinder's live burn off the sim's own firing
    bookkeeping: total crank degrees since its last real firing event,
    and the strength that event actually had. No schedule -- a cut
    ignition, a misfire, a stalled crank all simply show nothing."""
    out = []
    total = float(getattr(sim, "_total_crank_deg", 0.0))
    stalled = bool(getattr(sim.state, "stalled", False))
    cycle = float(getattr(sim.engine.architecture, "cycle_degrees", 720.0))
    for cyl, last_fire in getattr(sim, "_last_fire_total_deg", {}).items():
        strength = float(getattr(sim, "_last_strength", {}).get(cyl, 0.0))
        since = total - float(last_fire)
        burn = None
        smoke = None
        if not stalled and 0.0 <= since < burn_duration_deg and strength > 0.0:
            burn = since / burn_duration_deg
        # the exhaust stroke: the second half-cycle after the burn on a
        # four-stroke (power 0..180, exhaust 180..360 after TDC); a two-
        # stroke blows down right after the power stroke
        ex_start = 180.0 if cycle >= 700.0 else 100.0
        ex_len = 180.0 if cycle >= 700.0 else 80.0
        if strength > 0.0 and ex_start <= since < ex_start + ex_len:
            smoke = (since - ex_start) / ex_len
        out.append(CylinderBurnState(cylinder=cyl, burn_phase=burn, strength=strength, smoke_phase=smoke))
    return out
