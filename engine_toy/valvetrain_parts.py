"""Valvetrain PARTS: every valve, spring, retainer, follower/rocker and
pushrod as an individual, removable part with its own state -- not
decoration under the covers, but the things a game takes off, checks
for clearance, lets sag with age, and shims/tunes.

Per valve (from the cylinder layout's valve ports, one part set per
port of role intake/exhaust that is a poppet valve):
  valve      stem + head, on the port's own axis, lifted by the cam
  spring     a helix with a real STATE: free length, rate, installed
             height, permanent set (sag) and shim (tune) -- seat load
             and open load follow from those (F = k * compression)
  retainer   the disc the spring pushes against
  follower   a bucket tappet (overhead cam) or a rocker (pushrod), and
             the pushrod itself below it
Removability: each part has an axis-aligned bounding box, what it is
attached to, and the direction it comes out along; a part is CLEAR to
remove when its swept box along that direction meets nothing but the
parts it attaches to (and the parts that must come off first are
listed). That is the assembly rule a game needs: the cover comes off
before the springs, the retainer before the valve.

Spring state is real spring arithmetic, disclosed as first-order:
  seat_load = rate * (free_length - sag - installed_height + shim)
  open_load = seat_load + rate * lift
A spring that has sagged past its shim loses seat pressure, floats
earlier (engines.LifterSpring's own max_safe_rpm relation) and lets
the valve bounce -- which is what an old engine's "tired springs" are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from mesh_primitives import tube_mesh, cuboid_mesh, capped_tube_mesh

CRANK_AXIS = np.array([1.0, 0.0, 0.0])


def _unit(v):
    a = np.array(v, dtype=np.float64)
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else np.array([0.0, 1.0, 0.0])


def _perp(axis):
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    u = _unit(np.cross(axis, helper)); v = _unit(np.cross(axis, u))
    return u, v


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

@dataclass
class SpringState:
    free_length_m: float
    rate_n_per_m: float
    installed_height_m: float
    lift_m: float
    sag_m: float = 0.0          # permanent set -- grows with heat and cycles
    shim_m: float = 0.0         # a shim under the spring (tuning): raises seat load

    @property
    def seat_load_n(self) -> float:
        return max(0.0, self.rate_n_per_m * (self.free_length_m - self.sag_m - self.installed_height_m + self.shim_m))

    @property
    def open_load_n(self) -> float:
        return self.seat_load_n + self.rate_n_per_m * self.lift_m

    @property
    def coil_bind_margin_m(self) -> float:
        """Room left before the coils stack solid at full lift (a shim
        that is too thick binds the spring -- a real tuning limit)."""
        solid = self.free_length_m * 0.55
        return (self.installed_height_m - self.shim_m - self.lift_m) - solid

    def age(self, cycles: float, temp_k: float) -> None:
        """Permanent set: a disclosed first-order relaxation -- a few
        percent of free length over ~1e8 cycles at 400 K, accelerating
        with temperature (creep is thermally activated)."""
        accel = math.exp((temp_k - 400.0) / 60.0)
        limit = self.free_length_m * 0.05
        self.sag_m = min(limit, self.sag_m + limit * (cycles / 1.0e8) * accel)


@dataclass(frozen=True)
class MatingFace:
    """What a part actually seats on: the surface, its normal, what
    seals it and what holds it. This -- not a swept-box proof -- is
    the removability/assembly spec: a part comes off its face along
    the face normal once the parts listed as must_remove_first are
    gone, and it goes back on any face of the same kind."""
    face: str
    on_part: str
    normal: tuple
    seal: str = "none"
    fasteners: str = "none"


@dataclass
class RemovablePart:
    identity: str
    kind: str
    attached_to: tuple                 # identities this part mounts on (may stay in place)
    centre: np.ndarray
    half_extent: np.ndarray            # AABB half extents
    removal_direction: np.ndarray      # unit vector the part comes out along
    removal_travel_m: float            # how far it must travel to be clear
    state: dict = field(default_factory=dict)
    must_remove_first: tuple = ()      # parts that sit over it by design
    mates_to: MatingFace | None = None

    def aabb(self):
        return self.centre - self.half_extent, self.centre + self.half_extent

    def swept_aabb(self):
        """The volume the part passes THROUGH on its way out: from just
        past its own far face along the removal direction to the end
        of its travel -- so a neighbour it merely touches at rest is
        not a blocker, only something actually in the path is."""
        lo, hi = self.aabb()
        d = self.removal_direction
        start = np.abs(d) * (2.0 * self.half_extent) * 1.02
        step = d * self.removal_travel_m
        lo2, hi2 = lo + d * np.dot(np.abs(d), 2.0 * self.half_extent) * 1.02, hi + d * np.dot(np.abs(d), 2.0 * self.half_extent) * 1.02
        return np.minimum(lo2, lo2 + step), np.maximum(hi2, hi2 + step)


def _overlaps(lo_a, hi_a, lo_b, hi_b, tol: float = 5e-4) -> bool:
    return bool(np.all(lo_a < hi_b - tol) and np.all(lo_b < hi_a - tol))


def removal_blockers(part: RemovablePart, parts: list) -> list:
    """Parts whose boxes intersect this part's removal sweep, other
    than what it attaches to -- empty means clear to remove now."""
    lo, hi = part.swept_aabb()
    blockers = []
    for other in parts:
        if other is part or other.identity in part.attached_to:
            continue
        olo, ohi = other.aabb()
        if _overlaps(lo, hi, olo, ohi):
            blockers.append(other.identity)
    return blockers


# ---------------------------------------------------------------------
# Per-valve assembly geometry
# ---------------------------------------------------------------------

@dataclass
class ValveAssembly:
    identity: str
    cylinder: int
    port_name: str
    role: str
    seat: np.ndarray            # valve head centre (at the port)
    axis: np.ndarray            # stem axis, pointing out of the chamber
    head_radius_m: float
    stem_radius_m: float
    stem_length_m: float
    spring: SpringState
    valvetrain: str             # "pushrod" | "sohc" | "dohc"
    lift_now_m: float = 0.0     # instantaneous lift (from the cam phase)

    @property
    def tip(self) -> np.ndarray:
        return self.seat + self.axis * self.stem_length_m


def valve_assemblies(layout, engine=None, crank_angle_deg: float = 0.0, valve_state=None) -> list:
    """One assembly per poppet-valve port in the layout, with the
    spring dimensioned off the engine's own LifterSpring when given
    (rate, valvetrain mass) and off the bore otherwise."""
    out = []
    rate = float(getattr(getattr(engine, "lifter_spring", None), "spring_rate_n_per_mm", 45.0) or 45.0) * 1000.0
    for g, ports in layout:
        if g.kind not in ("spark-piston", "compression-piston") or g.valvetrain == "none":
            continue
        for p in ports:
            if p.port_kind not in ("intake-valve-port", "exhaust-valve-port"):
                continue
            axis = _unit(p.direction)
            head_r = p.radius_m
            lift = head_r * 0.55
            stem_len = g.bore_m * 0.95
            free = g.bore_m * 0.55
            installed = free * 0.82
            spring = SpringState(free_length_m=free, rate_n_per_m=rate, installed_height_m=installed, lift_m=lift)
            # a first-order lift trace: the valve opens for ~a quarter of the cycle, phased by the throw
            cycle = 720.0
            phase = ((crank_angle_deg + g.throw_angle_deg + (180.0 if p.fluid_role == "intake" else 540.0)) % cycle) / cycle
            lift_now = lift * max(0.0, math.sin(math.pi * (phase / 0.32))) if phase < 0.32 else 0.0
            if valve_state is not None and len(out) < len(valve_state.cylinder):
                spring = valve_state.spring_state(len(out))
            out.append(ValveAssembly(identity=f"powertrain.cylinder_{g.number}.{p.name}.valve", cylinder=g.number,
                                     port_name=p.name, role=p.fluid_role, seat=np.array(p.position), axis=axis,
                                     head_radius_m=head_r, stem_radius_m=max(0.003, head_r * 0.14),
                                     stem_length_m=stem_len, spring=spring, valvetrain=g.valvetrain, lift_now_m=lift_now))
    return out


def removable_parts(layout, engine=None, crank_angle_deg: float = 0.0, extra_parts: list | None = None,
                    valve_state=None) -> list:
    """The assembly manifest: every valvetrain part as a RemovablePart,
    plus the covers/plugs/injectors from the layout, so a game can ask
    what comes off, in what order, and whether it is clear."""
    parts: list = []
    for g, ports in layout:
        for p in ports:
            if p.port_kind in ("spark-plug-boss", "glow-plug-boss", "direct-injector-boss", "port-injector-boss",
                               "gas-injector-boss", "drain-cock", "lubricator-boss"):
                d = _unit(p.direction)
                parts.append(RemovablePart(
                    identity=f"powertrain.cylinder_{g.number}.{p.name}", kind=p.port_kind,
                    attached_to=(f"powertrain.cylinder_{g.number}.head",), centre=np.array(p.position) + d * 0.012,
                    half_extent=np.array([p.radius_m * 1.3] * 3) + np.abs(d) * 0.012, removal_direction=d,
                    removal_travel_m=0.05, state={"torque_nm": 25.0 if "plug" in p.port_kind else 30.0},
                    mates_to=MatingFace("plug_thread" if "plug" in p.port_kind else "injector_bore",
                                        f"cylinder_{g.number}.head", tuple(float(x) for x in d),
                                        seal="crush-washer" if "plug" in p.port_kind else "o-ring",
                                        fasteners="M14 thread" if "plug" in p.port_kind else "clamp")))
    for va in valve_assemblies(layout, engine, crank_angle_deg, valve_state):
        a = va.axis; u, v = _perp(a)
        cyl_head = f"powertrain.cylinder_{va.cylinder}.head"
        spring_top = va.seat + a * (va.stem_length_m * 0.85)
        spring_c = va.seat + a * (va.stem_length_m * 0.55)
        spring_r = va.head_radius_m * 0.7
        he_spring = np.abs(a) * (va.spring.installed_height_m / 2.0) + (np.abs(u) + np.abs(v)) * spring_r
        retainer = RemovablePart(identity=f"{va.identity}.retainer", kind="valve-retainer", attached_to=(va.identity,),
                                 centre=spring_top, half_extent=np.abs(a) * 0.004 + (np.abs(u) + np.abs(v)) * spring_r,
                                 removal_direction=a, removal_travel_m=0.03, state={"keepers_seated": True},
                                 mates_to=MatingFace("retainer_groove", va.identity, tuple(float(x) for x in a), seal="keepers", fasteners="keepers"))
        spring = RemovablePart(identity=f"{va.identity}.spring", kind="valve-spring", attached_to=(cyl_head, va.identity),
                               centre=spring_c, half_extent=he_spring, removal_direction=a, removal_travel_m=0.06,
                               state={"seat_load_n": va.spring.seat_load_n, "open_load_n": va.spring.open_load_n,
                                      "sag_mm": va.spring.sag_m * 1000.0, "shim_mm": va.spring.shim_m * 1000.0,
                                      "coil_bind_margin_mm": va.spring.coil_bind_margin_m * 1000.0},
                               must_remove_first=(retainer.identity,),
                               mates_to=MatingFace("spring_pocket", cyl_head, tuple(float(x) for x in a), seal="none", fasteners="preload"))
        # the valve's box is its STEM (thin); the wide head sits at the
        # seat, below everything else, and comes out downward with it
        valve = RemovablePart(identity=va.identity, kind=f"{va.role}-valve", attached_to=(cyl_head,),
                              centre=va.seat + a * (va.stem_length_m / 2.0),
                              half_extent=np.abs(a) * (va.stem_length_m / 2.0) + (np.abs(u) + np.abs(v)) * (va.stem_radius_m * 1.5),
                              removal_direction=-a, removal_travel_m=va.stem_length_m + 0.02,
                              state={"lift_mm": va.lift_now_m * 1000.0, "seat_recession_mm": 0.0},
                              must_remove_first=(spring.identity,),
                              mates_to=MatingFace("valve_guide", cyl_head, tuple(float(x) for x in -a), seal="valve-seat", fasteners="none"))
        if va.valvetrain in ("sohc", "dohc"):
            follower = RemovablePart(identity=f"{va.identity}.bucket", kind="bucket-tappet", attached_to=(cyl_head,),
                                     centre=spring_top + a * 0.012, half_extent=np.abs(a) * 0.012 + (np.abs(u) + np.abs(v)) * (spring_r * 1.1),
                                     removal_direction=a, removal_travel_m=0.03, state={"shim_mm": 2.5, "lash_mm": 0.20},
                                     mates_to=MatingFace("tappet_bore", cyl_head, tuple(float(x) for x in a), seal="none", fasteners="none"))
            parts.extend([follower, retainer, spring, valve])
        else:
            rocker_c = spring_top + a * 0.018
            rocker = RemovablePart(identity=f"{va.identity}.rocker", kind="rocker-arm", attached_to=(cyl_head,),
                                   centre=rocker_c + u * (va.head_radius_m * 0.4),
                                   half_extent=np.abs(a) * 0.012 + np.abs(u) * (va.head_radius_m * 1.1) + np.abs(v) * 0.01,
                                   removal_direction=a, removal_travel_m=0.05, state={"lash_mm": 0.25, "ratio": 1.6},
                                   mates_to=MatingFace("rocker_stud", cyl_head, tuple(float(x) for x in a), seal="none", fasteners="stud + nut"))
            pushrod = RemovablePart(identity=f"{va.identity}.pushrod", kind="pushrod", attached_to=(rocker.identity,),
                                    centre=rocker_c + u * (va.head_radius_m * 1.0) - a * (va.stem_length_m * 0.9),
                                    half_extent=np.abs(a) * (va.stem_length_m * 0.9) + (np.abs(u) + np.abs(v)) * 0.005,
                                    removal_direction=a, removal_travel_m=va.stem_length_m * 1.9, state={"straight": True},
                                    must_remove_first=(rocker.identity,),
                                    mates_to=MatingFace("lifter_cup", f"cylinder_{va.cylinder}.block", tuple(float(x) for x in a), seal="none", fasteners="none"))
            parts.extend([rocker, pushrod, retainer, spring, valve])
    if extra_parts:
        parts.extend(extra_parts)
    return parts


def assembly_manifest(parts: list, advisory_clearance: bool = False) -> list[str]:
    """The assembly spec: what each part seats on (face, seal,
    fasteners), what must come off first, and its state. The swept-
    box check is advisory only (advisory_clearance=True): the mating
    faces are the rule, not a geometric proof."""
    lines = []
    for p in parts:
        first = p.must_remove_first[-1].split(".")[-1] if p.must_remove_first else "-"
        mf = p.mates_to
        face = f"{mf.face} on {mf.on_part.replace('powertrain.', '')} [{mf.seal}, {mf.fasteners}]" if mf else "-"
        st = ", ".join(f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}" for k, v in p.state.items())
        adv = ""
        if advisory_clearance:
            blockers = removal_blockers(p, parts)
            adv = "  (sweep clear)" if not blockers else "  (sweep meets " + ", ".join(b.split(".")[-1] for b in blockers[:2]) + ")"
        lines.append(f"{p.identity.replace('powertrain.', ''):46s} {p.kind:16s} after: {first:9s} {face:58s} {st}{adv}")
    return lines


# ---------------------------------------------------------------------
# Meshes (covers-off detail)
# ---------------------------------------------------------------------

def _helix_parts(centre_base, axis, radius, height, turns, wire_r, name, SolidPart, segments_per_turn=6):
    u, v = _perp(axis)
    n = int(turns * segments_per_turn)
    pts = [centre_base + axis * (height * k / n) + u * (radius * math.cos(2 * math.pi * turns * k / n))
           + v * (radius * math.sin(2 * math.pi * turns * k / n)) for k in range(n + 1)]
    verts = []; norms = []
    for a, b in zip(pts, pts[1:]):
        vt, nm = tube_mesh(a, b, wire_r, sides=4)
        verts.append(vt); norms.append(nm)
    return SolidPart(vertices=np.concatenate(verts), normals=np.concatenate(norms), thermal_group=None, name=name)


def build_valvetrain_parts(layout, engine=None, crank_angle_deg: float = 0.0, spring_style: str = "helix") -> list:
    """spring_style: "helix" (a real coil, for close-ups and exports) or
    "cylinder" (a plain drum at the compressed height, for a game render:
    it still shows compression and costs a dozen triangles)."""
    from vehicle_mesh import SolidPart
    out = []
    for va in valve_assemblies(layout, engine, crank_angle_deg):
        a = va.axis; u, v = _perp(a)
        tag = f"cyl{va.cylinder}_{va.port_name}"
        seat = va.seat - a * va.lift_now_m      # an open valve drops into the chamber
        vt, nm = capped_tube_mesh(seat - a * 0.003, seat + a * 0.003, va.head_radius_m, sides=16)
        out.append(SolidPart(vertices=vt, normals=nm, thermal_group=None, name=f"{tag}_valve_head"))
        vt, nm = capped_tube_mesh(seat, seat + a * va.stem_length_m, va.stem_radius_m, sides=8)
        out.append(SolidPart(vertices=vt, normals=nm, thermal_group=None, name=f"{tag}_valve_stem"))
        comp = va.spring.installed_height_m - va.lift_now_m
        base = va.seat + a * (va.stem_length_m * 0.85 - va.spring.installed_height_m) + a * va.spring.shim_m
        if spring_style == "cylinder":
            vt, nm = tube_mesh(base, base + a * comp, va.head_radius_m * 0.7, sides=8)
            out.append(SolidPart(vertices=vt, normals=nm, thermal_group=None, name=f"{tag}_spring"))
        else:
            out.append(_helix_parts(base, a, va.head_radius_m * 0.7, comp, 5.0, va.stem_radius_m * 0.6,
                                    f"{tag}_spring", SolidPart))
        top = base + a * comp
        vt, nm = capped_tube_mesh(top, top + a * 0.006, va.head_radius_m * 0.75, sides=14)
        out.append(SolidPart(vertices=vt, normals=nm, thermal_group=None, name=f"{tag}_retainer"))
        if va.valvetrain in ("sohc", "dohc"):
            vt, nm = capped_tube_mesh(top + a * 0.006, top + a * 0.026, va.head_radius_m * 0.8, sides=14)
            out.append(SolidPart(vertices=vt, normals=nm, thermal_group=None, name=f"{tag}_bucket"))
        else:
            rc = top + a * 0.02
            vt, nm = cuboid_mesh(rc + u * (va.head_radius_m * 0.4),
                                 np.abs(a) * 0.008 + np.abs(u) * (va.head_radius_m * 1.1) + np.abs(v) * 0.008)
            out.append(SolidPart(vertices=vt, normals=nm, thermal_group=None, name=f"{tag}_rocker"))
            vt, nm = capped_tube_mesh(rc + u * (va.head_radius_m * 1.0), rc + u * (va.head_radius_m * 1.0) - a * (va.stem_length_m * 1.8),
                                      0.004, sides=8)
            out.append(SolidPart(vertices=vt, normals=nm, thermal_group=None, name=f"{tag}_pushrod"))
    return out
