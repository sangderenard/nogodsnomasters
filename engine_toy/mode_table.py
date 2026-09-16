"""The inertial solve, evaluated once over the whole graph, as a table.

WHAT IT ANSWERS. For every state a machine can be in, and every point
along that state's curve: what it weighs and where, which feet it is
standing on, its three modes on those feet, how a rotor at speed
splits them, and what each running source does to the frame at the
speed it is running -- through the load path its body actually has to
the mounts, not through an assumed weld. Every one of those is a
function of the entire graph, and none of them needs to be evaluated
more than once per state, so this evaluates them once and keeps rows.

THE ROW IS THE PRODUCT. A deployment that runs live reads `Row` objects
straight from `evaluate`. A deployment that does not -- the ordinary
case, and the one the realtime path wants -- reads `bake`, which is
the same rows as plain data with the closed-form coefficients a frame
needs and nothing it does not, and `lookup`, which is a nearest-row
read plus one multiply. The election between them is the cycle's
`live_vibration`, made by the machine, not by this module.

WHAT THE LOAD PATH CHANGES. A source's unbalance force is scaled by its
body's transmissibility along its wrench path at the running frequency
before it is applied to the frame. A welded pump shakes the frame with
its whole unbalance; a pump on isolators shakes it less above the
isolator frequency and MORE at it; a pump that reaches no mount shakes
the frame not at all and is a problem on the row instead. The rigid
tensor could see none of those, and the difference between them is the
difference between a machine that is fine and one that hammers its
feet every start.

THE MODEL'S OWN BOUNDARIES, carried on each row as hazards rather than
silently crossed: a running speed within a tenth of a mode is a
crossing; a foot leaving the floor or a short one being struck is
hammering, and past that point the numbers are boundary estimates; a
source whose body has no load path is unsupported and not applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import feet as feet_mod
import machine_shake as shake
from machine_package import mass_properties
from operating_states import Cycle, OperatingState, sources_of
from wrench_paths import wrench_paths

#: How close a running frequency has to be to a mode to count as
#: crossing it. A tenth is the half-power band of a lightly damped
#: mode with the ground damping this toy declares, so a speed inside it
#: is being amplified, not merely near.
CROSSING_BAND = 0.10


@dataclass
class SourceEffect:
    """What one running source does to the frame, in this row."""
    source: str
    rpm: float
    unbalance_kg_m: float
    path_kind: str
    transmission: float
    response: "shake.Response | None"
    hammering_w: dict = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return self.response is not None


@dataclass
class Row:
    """One state, one point on its curve, everything that follows."""
    state: str
    sample: float
    rigid_body: object
    stance: object
    model: shake.ShakeModel
    whirl: list
    angular_momentum_y: float
    effects: list
    hazards: tuple

    @property
    def frequencies_hz(self) -> tuple:
        return tuple(m.frequency_hz for m in self.whirl)

    @property
    def hammering_w(self) -> float:
        return sum(sum(e.hammering_w.values()) for e in self.effects)

    def describe(self) -> str:
        freqs = ", ".join(f"{f:.1f}" for f in self.frequencies_hz)
        line = (f"  {self.state:12s} @{self.sample:4.2f}  "
                f"{self.rigid_body.total_mass_kg:7.1f} kg  modes [{freqs}] Hz")
        for e in self.effects:
            if e.applied:
                line += (f"\n      {e.source.split('.')[-1]:14s} {e.rpm:6.0f} rpm "
                         f"x{e.transmission:4.2f} {e.path_kind:11s} "
                         f"{e.response.heave_m * 1e6:8.2f} um")
            else:
                line += (f"\n      {e.source.split('.')[-1]:14s} {e.rpm:6.0f} rpm "
                         f"NOT APPLIED ({e.path_kind})")
        if self.hammering_w > 0.0:
            line += f"\n      hammering {self.hammering_w:.3g} W"
        for h in self.hazards:
            line += f"\n      ! {h}"
        return line


@dataclass
class ModeTable:
    identity: str
    rows: list
    paths: dict
    problems: tuple
    live_vibration: bool = False

    def at(self, state: str, sample: float) -> Row:
        """The nearest row in the named state."""
        cands = [r for r in self.rows if r.state == state]
        if not cands:
            raise KeyError(f"{self.identity}: no rows for state {state!r}")
        return min(cands, key=lambda r: abs(r.sample - sample))

    def states(self) -> tuple:
        seen = []
        for r in self.rows:
            if r.state not in seen:
                seen.append(r.state)
        return tuple(seen)

    def hazards(self) -> list:
        return [(r.state, r.sample, h) for r in self.rows for h in r.hazards]

    def describe(self) -> str:
        lines = [f"{self.identity}: mode table, {len(self.rows)} rows over "
                 f"{len(self.states())} states, vibration "
                 f"{'live' if self.live_vibration else 'baked'}"]
        for p in self.problems:
            lines.append(f"  PROBLEM: {p}")
        lines.extend(r.describe() for r in self.rows)
        return "\n".join(lines)


def _charged_nodes(nodes, state: OperatingState) -> list:
    """The graph's bodies with this state's contents added on.

    A container's charge is mass AT the container -- point mass at its
    centre, because what the collector holds sits in the collector --
    and it changes the total, the CG, and the tensor, which is why a
    state with a full collector has different modes from an empty one."""
    if not state.charges_kg:
        return list(nodes)
    out = []
    for n in nodes:
        extra = state.charges_kg.get(n["identity"])
        if extra:
            n = dict(n)
            n["mass_kg"] = float(n.get("mass_kg", 0.0)) + float(extra)
        out.append(n)
    return out


def evaluate(doc: dict, feet, ground, cycle: Cycle, *,
             floor_height=0.0, frame_underside_y: float = 0.0) -> ModeTable:
    """Every state, every sample, once."""
    nodes = doc["nodes"]
    sources = sources_of(doc)
    paths = wrench_paths(doc)
    from wrench_paths import problems as path_problems
    problems = tuple(path_problems(paths, doc))
    rows = []
    for state, sample in cycle.samples():
        charged = _charged_nodes(nodes, state)
        rb = mass_properties(charged)
        stance = feet_mod.stand(feet, rb, ground, floor_height=floor_height,
                                frame_underside_y=frame_underside_y)
        model = shake.build(f"{doc.get('identity', 'machine')}:{state.name}",
                            stance, feet, rb, ground)
        # every running rotor's momentum, reacted along the vertical
        H = np.zeros(3)
        for s in sources:
            H += s.angular_momentum(state.fraction_for(s, sample))
        whirl = shake.whirl_modes(model, float(H[1]))
        hazards = []
        if stance.rocks:
            hazards.append(f"stance rocks: {stance.reason}")
        effects = []
        for s in sources:
            frac = state.fraction_for(s, sample)
            if frac <= 0.0:
                continue
            rpm = s.rpm(frac)
            hz = rpm / 60.0
            path = paths.get(s.identity)
            kind = path.kind if path else "unsupported"
            T = path.transmission(hz, s.mass_kg) if path else 0.0
            unbalance = state.unbalance_for(s)
            if kind == "unsupported":
                effects.append(SourceEffect(s.identity, rpm, unbalance, kind,
                                            0.0, None))
                hazards.append(f"{s.identity} runs at {rpm:.0f} rpm with no "
                               "load path to a mount")
                continue
            resp = shake.respond(model, rpm, unbalance_kg_m=unbalance * T,
                                 at=s.position)
            hammer = shake.hammering_power_w(model, resp)
            effects.append(SourceEffect(s.identity, rpm, unbalance, kind, T,
                                        resp, hammer))
            for m in whirl:
                if m.frequency_hz > 0.0 and abs(hz - m.frequency_hz) <= CROSSING_BAND * m.frequency_hz:
                    hazards.append(f"{s.identity} at {rpm:.0f} rpm crosses the "
                                   f"{m.kind} mode at {m.frequency_hz:.1f} Hz")
            if resp.lifts_off:
                hazards.append(f"{', '.join(resp.lifts_off)} leaves the floor "
                               f"under {s.identity}")
            if resp.strikes:
                hazards.append(f"{', '.join(resp.strikes)} is struck "
                               f"under {s.identity}")
        rows.append(Row(state=state.name, sample=sample, rigid_body=rb,
                        stance=stance, model=model, whirl=whirl,
                        angular_momentum_y=float(H[1]), effects=effects,
                        hazards=tuple(hazards)))
    return ModeTable(identity=str(doc.get("identity", "machine")), rows=rows,
                     paths=paths, problems=problems,
                     live_vibration=cycle.live_vibration)


# ---------------------------------------------------------------------
# the prebake
# ---------------------------------------------------------------------

def bake(table: ModeTable) -> dict:
    """The table as plain data a frame can read without any of this.

    Per row: the modes, and per running source the per-foot dynamic
    force and the heave at the speed it runs -- the closed-form answers,
    so the live path is a lookup and a multiply, not an eigenproblem.
    Nothing in here needs numpy, a graph, or an import of this module,
    which is what makes it bakeable into a deployment that elects not
    to carry the analysis."""
    rows = []
    for r in table.rows:
        rows.append({
            "state": r.state, "sample": r.sample,
            "mass_kg": float(r.rigid_body.total_mass_kg),
            "cg": [float(v) for v in r.rigid_body.center_of_gravity],
            "carrying": sorted(k for k, f in r.stance.forces.items() if f > 0.0),
            "static_n": {k: float(v) for k, v in r.stance.forces.items()},
            "modes_hz": [float(f) for f in r.frequencies_hz],
            "mode_kinds": [m.kind for m in r.whirl],
            "angular_momentum_y": r.angular_momentum_y,
            "sources": [{
                "source": e.source, "rpm": e.rpm, "path": e.path_kind,
                "transmission": e.transmission, "applied": e.applied,
                "heave_m": float(e.response.heave_m) if e.applied else 0.0,
                "slope_x": float(e.response.slope_x) if e.applied else 0.0,
                "slope_z": float(e.response.slope_z) if e.applied else 0.0,
                "amplification": float(e.response.amplification) if e.applied else 0.0,
                "foot_force_n": ({k: float(v) for k, v in e.response.foot_force_n.items()}
                                 if e.applied else {}),
                "hammering_w": {k: float(v) for k, v in e.hammering_w.items()},
            } for e in r.effects],
            "hazards": list(r.hazards),
        })
    return {"schema": "engine-toy-mode-table-v1", "identity": table.identity,
            "live_vibration": table.live_vibration,
            "problems": list(table.problems),
            "paths": {k: {"kind": p.kind, "mount": p.mount,
                          "stiffness_n_per_m": (None if p.stiffness_n_per_m == math.inf
                                                else p.stiffness_n_per_m),
                          "reaches": list(p.reaches),
                          "couple_arm_m": p.couple_arm_m}
                      for k, p in table.paths.items()},
            "rows": rows}


def lookup(baked: dict, state: str, sample: float) -> dict:
    """The live path: nearest row in the state, no arithmetic beyond it.

    Nearest rather than interpolated on purpose. Modes and hazards are
    not continuous across a lift-off, and a row half way between a
    quiet one and a hammering one would describe neither."""
    cands = [r for r in baked["rows"] if r["state"] == state]
    if not cands:
        raise KeyError(f"{baked.get('identity')}: no baked rows for {state!r}")
    return min(cands, key=lambda r: abs(r["sample"] - float(sample)))


def frame_motion(row: dict, phase_rad: float) -> dict:
    """The one multiply a frame does: where the frame is right now.

    Every source contributes its closed-form amplitude at its own phase;
    for one running source this IS the steady-state motion, and for
    several it is their sum on the assumption they are not locked --
    which is what unrelated motors on one frame do."""
    heave = sx = sz = 0.0
    for i, s in enumerate(row["sources"]):
        if not s["applied"]:
            continue
        c = math.cos(phase_rad * (s["rpm"] / max(row["sources"][0]["rpm"], 1e-9)) + i)
        heave += s["heave_m"] * c
        sx += s["slope_x"] * c
        sz += s["slope_z"] * c
    return {"heave_m": heave, "slope_x": sx, "slope_z": sz}
