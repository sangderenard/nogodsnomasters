"""Parts as PRISMS with wrench points at their ports.

THE PROBLEM THIS ENDS. A part in these graphs has been one node at its
own centroid. That is enough to hang a mesh on and it is wrong for
every other purpose, because the single most consequential fact about a
solid body is that force does not enter it at its middle -- force
enters a body WHERE SOMETHING TOUCHES IT, which is always a surface.

An audit of the turret found forty-one members passing through solid
metal, and every one of them had the same cause: a member "attached to
the breech" is a line drawn to the breech's centroid, so it reaches
through 266 mm of gun steel to get there. Moving coordinates never
fixed it, because the coordinates were not the problem. The problem was
that the thing being attached to had no surface to attach to.

WHAT A PRISM IS HERE. A part declares its own solid shape -- a box, a
cylinder, a tube -- and from that shape any point on its surface is
derivable. A PORT is such a point, with the outward normal there. And a
port is where a WRENCH applies: a force and a moment, at a stated point,
in a stated direction. That is the whole idea, and it means:

  - a joint cannot tunnel, because both ends of a mated pair are on the
    two surfaces that touch and the member between them has zero length
  - a load path is real, because the wrench enters at the port, travels
    through the body's own shell to the other ports, and leaves there
  - a tube has an INSIDE as well as an outside, so a spacer between two
    concentric tubes is a real member of real length rather than two
    coincident axis nodes with nothing between them
  - and nothing has to be recognised by the shape of its name

THE BODY IS A SHELL, NOT A HUB. Emitting a port and tying it to the
centroid would put the tunnelling member back, one level down. The
ports of a prism are tied to each other around the body's surface --
which is what the metal actually is -- and to the centroid only as the
mass reference it is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

FACES = ("-x", "+x", "-y", "+y", "-z", "+z")


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _frame(axis):
    """The axis plus two perpendiculars, so an angle around a round body
    means the same thing every time it is asked for."""
    a = _unit(axis)
    helper = (np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9
              else np.array([0.0, 1.0, 0.0]))
    u = _unit(np.cross(a, helper))
    v = _unit(np.cross(a, u))
    return a, u, v


@dataclass(frozen=True)
class PrismPort:
    """A point on a part's surface, and the outward normal there.

    This is the wrench point: whatever is bolted, welded, seated or
    pressed here applies its force and its moment AT THIS POINT, not at
    the body's centre of mass."""
    name: str
    position: np.ndarray
    direction: np.ndarray
    face: str
    surface: str = "outer"         # "outer" | "inner" | "end"

    @property
    def identity_suffix(self) -> str:
        return self.name


@dataclass
class Prism:
    """A part's actual solid shape, and the source of every port on it.

    `shape` is one of:
      "box"       half_extent (hx, hy, hz), axis ignored
      "cylinder"  radius, length along axis
      "tube"      outer_radius, inner_radius, length along axis -- and
                  it has an INNER surface, which is the whole point for
                  anything concentric
    """
    identity: str
    kind: str
    centre: tuple
    shape: str = "box"
    axis: tuple = (0.0, 0.0, 1.0)
    half_extent: tuple = (0.05, 0.05, 0.05)
    radius: float = 0.05
    inner_radius: float = 0.0
    length: float = 0.10
    material: str = "steel-plate"
    mass_kg: float | None = None
    attributes: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- #
    def _c(self):
        return np.asarray(self.centre, dtype=np.float64)

    def face_point(self, face: str, *, across=(0.0, 0.0)) -> PrismPort:
        """A point on one flat face of a box, `across` in the range
        [-1, 1] over the other two half-extents."""
        if self.shape != "box":
            raise ValueError(f"{self.identity}: face_point is for a box; "
                             f"this is a {self.shape} -- use end_point or "
                             f"side_point")
        sign = -1.0 if face[0] == "-" else 1.0
        ai = "xyz".index(face[1])
        p = self._c().copy()
        h = np.asarray(self.half_extent, dtype=np.float64)
        p[ai] += sign * h[ai]
        others = [i for i in range(3) if i != ai]
        for k, i in enumerate(others):
            p[i] += float(across[k]) * h[i]
        d = np.zeros(3)
        d[ai] = sign
        return PrismPort(face, p, d, face, "end")

    def end_point(self, end: str, *, radius_fraction: float = 0.0,
                  angle_rad: float = 0.0) -> PrismPort:
        """A point on the flat end of a cylinder or tube. `end` is
        "rear" (against the axis) or "front" (along it)."""
        a, u, v = _frame(self.axis)
        sign = -1.0 if end == "rear" else 1.0
        r = self.radius * float(radius_fraction)
        if self.shape == "tube" and radius_fraction:
            r = self.inner_radius + (self.radius - self.inner_radius) * float(
                radius_fraction)
        p = (self._c() + a * (sign * self.length / 2.0)
             + (u * math.cos(angle_rad) + v * math.sin(angle_rad)) * r)
        return PrismPort(f"{end}", p, a * sign, end, "end")

    def side_point(self, angle_rad: float, *, along: float = 0.0,
                   surface: str = "outer") -> PrismPort:
        """A point on the curved surface. `along` runs -1..+1 over the
        length. `surface` picks the outside or, for a tube, the BORE --
        which is where anything concentric inside it makes contact."""
        if self.shape == "box":
            raise ValueError(f"{self.identity}: a box has no curved side")
        a, u, v = _frame(self.axis)
        r = self.radius if surface == "outer" else self.inner_radius
        if surface == "inner" and self.shape != "tube":
            raise ValueError(f"{self.identity}: only a tube has a bore; "
                             f"this is a {self.shape}")
        radial = u * math.cos(angle_rad) + v * math.sin(angle_rad)
        p = self._c() + a * (float(along) * self.length / 2.0) + radial * r
        # outward from the METAL: on a bore, that points inward
        d = radial if surface == "outer" else -radial
        return PrismPort(f"side{math.degrees(angle_rad):.0f}", p, d,
                         "side", surface)

    def flat_on_side(self, angle_rad: float, *, half_width_fraction=0.62):
        """The machined flat a round body is held by: the chord cut at
        `angle_rad`, and how far across it is usable."""
        port = self.side_point(angle_rad)
        a, u, v = _frame(self.axis)
        radial = u * math.cos(angle_rad) + v * math.sin(angle_rad)
        tangent = _unit(np.cross(_unit(self.axis), radial))
        return {"centre": port.position, "normal": port.direction,
                "tangent": tangent, "axis": _unit(self.axis),
                "half_width_m": self.radius * float(half_width_fraction)}

    def contains(self, point, *, margin: float = 0.0) -> bool:
        """Is this point inside the solid? The interference test every
        member ought to pass and, until now, nothing ran."""
        p = np.asarray(point, dtype=np.float64) - self._c()
        if self.shape == "box":
            h = np.asarray(self.half_extent, dtype=np.float64) * (1.0 - margin)
            return bool(np.all(np.abs(p) <= h))
        a, u, v = _frame(self.axis)
        along = float(np.dot(p, a))
        radial = float(np.linalg.norm(p - a * along))
        if abs(along) > self.length / 2.0 * (1.0 - margin):
            return False
        if self.shape == "tube":
            return (self.inner_radius * (1.0 + margin) <= radial
                    <= self.radius * (1.0 - margin))
        return radial <= self.radius * (1.0 - margin)


def emit_prism(g, prism: Prism, ports: dict[str, PrismPort], *,
               motion_group: str | None = None, assembly: str | None = None,
               shell_radius: float = 0.004, web_radius: float = 0.003,
               port_kwargs: dict | None = None) -> dict[str, str]:
    """Write the body and its wrench points into a production graph.

    Emits the centroid (the mass reference), one node per port, a SHELL
    of members tying the ports to one another, and a web from each port
    to the centroid. The shell is what makes this a body: ports tied
    only to the centre would be a hub and spoke, and a load entering one
    port would reach another by travelling through the middle -- the
    exact thing this module exists to stop.

    THE SHELL AND THE WEB ARE NOT MEMBERS. They are the body's own
    metal, already accounted for by the body's declared shape and its
    material. Drawing them at a structural section says there is a cage
    of tubes here, and there is not -- so they are emitted at an
    insignificant radius and marked `body_internal` and `rigid` -- the
    flag that excuses a thing from resonance, because a body's own skin
    does not ring separately from the body. The part is SEEN
    by its declared prism geometry, which is the paper; a member is
    seen by ITS declared section, which is also the paper. Nothing is
    drawn at a radius picked to make a picture look right.

    Returns {port name: node identity}, so callers attach to ports by
    the name they gave them and never touch the centroid.
    """
    if motion_group is not None:
        g.motion_group = motion_group
    if assembly is not None:
        g.assembly = assembly
    geom = {"shape": prism.shape}
    if prism.shape == "box":
        geom["half_extent_m"] = tuple(float(v) for v in prism.half_extent)
    elif prism.shape == "cylinder":
        geom.update(drum_axis=tuple(float(v) for v in prism.axis),
                    drum_radius_m=float(prism.radius),
                    drum_length_m=float(prism.length),
                    half_extent_m=(prism.radius, prism.radius,
                                   prism.length / 2.0))
    else:
        geom.update(tube_axis=tuple(float(v) for v in prism.axis),
                    tube_outer_radius_m=float(prism.radius),
                    tube_inner_radius_m=float(prism.inner_radius),
                    drum_length_m=float(prism.length),
                    half_extent_m=(prism.radius, prism.radius,
                                   prism.length / 2.0))
    body_kw = dict(material=prism.material, **geom, **prism.attributes)
    if prism.mass_kg is not None:
        body_kw.setdefault("mass_kg", float(prism.mass_kg))
    g.node(prism.identity, [float(v) for v in prism.centre], prism.kind,
           **body_kw)

    made: dict[str, str] = {}
    pk = dict(port_kwargs or {})
    for name, port in ports.items():
        ident = f"{prism.identity}.port.{name}"
        g.node(ident, [float(v) for v in port.position],
               "structural-mount-port",
               material=prism.material,
               # THE WRENCH POINT. A force and a moment apply HERE, on
               # the surface, not at the body's centre of mass.
               wrench_point=True,
               surface_of=prism.identity,
               # A surface wrench point is not another little body.  It
               # follows this prism rigidly at its actual offset while all
               # attached members continue to apply force and moment there.
               solver_condensed_into=prism.identity,
               solver_condensed_mass=False,
               face=port.face, surface=port.surface,
               outward=[float(v) for v in _unit(port.direction)],
               **{"half_extent_m": (0.022, 0.022, 0.022),
                  "mass_in_total": False, "mass_kg": 0.4, **pk})
        made[name] = ident
        g.edge(f"{prism.identity}.web.{name}", ident, prism.identity,
               "rigid-distance", radius=web_radius, palette="rollbar-silver",
               body_internal="web", rigid=True,
               beam_solvable=False,
               load_path="wrench-point-into-the-body-it-is-a-surface-of")
    # THE SHELL: every port tied to the next, around the body
    names = list(ports)
    for i, name in enumerate(names):
        j = names[(i + 1) % len(names)]
        if j == name:
            continue
        g.edge(f"{prism.identity}.shell.{name}__{j}", made[name], made[j],
               "rigid-distance", radius=shell_radius,
               palette="rollbar-silver",
               body_internal="shell", rigid=True,
               beam_solvable=False,
               load_path="the-body's-own-skin-carrying-load-between-ports")
    return made


def interference(document: dict, prisms: list[Prism], *,
                 samples: int = 41, margin: float = 0.02) -> list[dict]:
    """Every member that passes through a declared solid.

    The test nothing in this project ran, which is why a structure could
    validate to 0.000% against analytical cantilevers and still have
    forty-one members buried in gun steel: a frame solver assembles
    stiffness from node positions and connectivity and has no concept of
    solid volume at all."""
    pos = {n["identity"]: np.asarray(n["reference_position"], dtype=np.float64)
           for n in document["nodes"]}
    out = []
    t = np.linspace(0.0, 1.0, samples)[:, None]
    for prism in prisms:
        own = {prism.identity}
        own |= {n["identity"] for n in document["nodes"]
                if n.get("surface_of") == prism.identity}
        for e in document["edges"]:
            if e["a"] in own or e["b"] in own:
                continue
            if e["a"] not in pos or e["b"] not in pos:
                continue
            p0, p1 = pos[e["a"]], pos[e["b"]]
            q = p0 + (p1 - p0) * t
            hits = sum(1 for point in q if prism.contains(point, margin=margin))
            if hits:
                out.append({"member": e["identity"], "body": prism.identity,
                            "samples": hits, "of": samples,
                            "assembly": e.get("assembly")})
    return out
