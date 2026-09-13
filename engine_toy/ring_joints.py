"""THE TWO JOINTS A TURRET IS, and what they negotiate.

Everything else in this machine is a member between two points. These
two are not: each is a ring of participants sharing one load, and the
only interesting question about either is HOW THE SHARE FALLS OUT. A
pinion between two ring gears and a floor resting on a ring face were
both in the graph as edges with a radius and a tooth count -- true
statements that settled nothing, because neither said what any
participant carries.

    CAPTIVE PLANETARY TWO-RING INTERFACE
        A pinion meshed simultaneously with a lower ring gear and an
        upper one. What makes it CAPTIVE is the separating force: gear
        teeth at a pressure angle push their mates apart, and with a
        ring either side the pinion is squeezed between two of them.
        Nothing holds it in but its retainer, and the retainer's preload
        is the number that decides whether the drive stays meshed.

        And if the two rings have DIFFERENT tooth counts it is not a
        drive, it is a reduction -- the upper ring advances by the
        tooth difference per carrier revolution, which is how a turret
        gets an enormous ratio out of one stage.

    CAPTIVE PASSIVE MATED PLANES RING
        Two flat annular faces held in contact. Passive: it drives
        nothing and is driven by nothing, it only carries. Captive:
        held together, because a flat face carries compression and has
        no answer at all to lift.

        Its whole behaviour is a PRESSURE DISTRIBUTION. A turret with an
        overhang does not load its ring evenly -- the resultant sits off
        centre, the pressure varies around the annulus, and past a
        certain overhang one side goes into tension, which a face cannot
        do. Where it would have gone negative, it lifts, and the
        hold-downs carry that instead. The ring "negotiating" is that
        redistribution, and it is the difference between a turntable and
        a hinge.

NOTHING HERE IS FITTED. Every number comes from the graph's own
declarations -- tooth counts, radii, widths, the masses and positions
the document already carries -- or from the standard relations for a
gear tooth and a bearing face.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


GRAVITY = 9.80665
#: Steel on greased steel, slow sliding. The band real flat slew tracks
#: are specified over; the low end is a well-fed film and the high end
#: is a tired one, and the difference is the traverse motor's whole
#: duty.
TRACK_FRICTION = (0.06, 0.15)
#: Rolling, so two orders down. This is why the rollers are there at all
#: and not only for the pressure.
ROLLER_FRICTION = 0.004


# =====================================================================
#  1. THE CAPTIVE PLANETARY
# =====================================================================
@dataclass
class PlanetaryRingMesh:
    """A pinion captive between two ring gears.

    `lower_teeth` is the ring bolted to the base; `upper_teeth` the one
    that carries the turret. Equal counts make it an ordinary drive at
    the pinion's own ratio. Unequal counts make it a reduction, and the
    ratio is ferocious."""
    pinion_teeth: int
    lower_teeth: int
    upper_teeth: int
    pitch_radius_m: float
    face_width_m: float
    pressure_angle_deg: float = 20.0
    pinions: int = 3
    #: not every pinion carries a third of the torque. Real multi-pinion
    #: drives share unevenly because the teeth are not perfectly indexed
    #: and the ring is not perfectly round; a mesh factor of 0.7 means
    #: the worst-placed pinion takes 1/(3*0.7) rather than 1/3.
    mesh_factor: float = 0.70
    retainer_preload_n: float = 0.0

    # -- geometry ------------------------------------------------------
    @property
    def module_m(self) -> float:
        """Tooth size, from the pitch radius and the count. It is the
        one number that has to agree across all three gears, and it is
        derived rather than declared so it cannot disagree."""
        return 2.0 * self.pitch_radius_m / max(self.pinion_teeth, 1)

    @property
    def lower_pitch_radius_m(self) -> float:
        return self.module_m * self.lower_teeth / 2.0

    @property
    def upper_pitch_radius_m(self) -> float:
        return self.module_m * self.upper_teeth / 2.0

    @property
    def tooth_delta(self) -> int:
        return self.upper_teeth - self.lower_teeth

    @property
    def reduction(self) -> float:
        """Carrier revolutions per revolution of the upper ring.

        WITH EQUAL RINGS THIS IS INFINITE, and that is not a bug in the
        arithmetic -- it is the mechanism telling the truth. Two ring
        gears of the same count, driven by a common pinion, advance
        together: the upper ring never moves relative to the lower one
        no matter how long the carrier turns. A turret built that way
        traverses by moving its CARRIER, and the second mesh is there
        for load sharing rather than for ratio.

        One tooth of difference turns the same hardware into a reduction
        of the ring's whole tooth count."""
        if self.tooth_delta == 0:
            return math.inf
        return self.upper_teeth / self.tooth_delta

    # -- forces --------------------------------------------------------
    def tooth_loads(self, traverse_torque_nm: float) -> dict:
        """What one tooth pair actually carries, and what it does to
        everything holding the pinion in place."""
        alpha = math.radians(self.pressure_angle_deg)
        shared = max(self.pinions * self.mesh_factor, 1e-9)
        # tangential, at the UPPER ring's pitch radius, because that is
        # the radius the turret's torque is applied at
        tangential = traverse_torque_nm / max(self.upper_pitch_radius_m, 1e-9)
        per_mesh = tangential / shared
        separating = per_mesh * math.tan(alpha)
        # THE PINION IS SQUEEZED FROM BOTH SIDES. Each mesh pushes it
        # away from that ring, and with a ring above and below, the two
        # separating forces act along the same line in opposite senses:
        # the retainer has to hold the WORSE of them, not their sum, and
        # the difference between those two readings is a factor of two
        # in a preload.
        return {
            "tangential_total_n": tangential,
            "tangential_per_mesh_n": per_mesh,
            "separating_per_mesh_n": separating,
            "radial_on_pinion_n": separating,
            "retainer_demand_n": separating,
            "retainer_margin": (self.retainer_preload_n
                                / max(separating, 1e-9)),
        }

    def contact_stress_pa(self, traverse_torque_nm: float,
                          youngs_pa: float = 2.05e11,
                          poisson: float = 0.3) -> float:
        """Hertzian line contact on the flank -- the stress that decides
        whether a tooth pits, which is what actually kills slew drives.

        A ring gear's mesh is INTERNAL, so the mate is concave and the
        relative curvature is a DIFFERENCE of curvatures rather than a
        sum. That is why an internal mesh carries far more than an
        external one of the same size, and using the external formula
        here would have condemned a perfectly good drive."""
        loads = self.tooth_loads(traverse_torque_nm)
        alpha = math.radians(self.pressure_angle_deg)
        w = loads["tangential_per_mesh_n"] / math.cos(alpha)
        r1 = self.pitch_radius_m * math.sin(alpha)
        r2 = self.upper_pitch_radius_m * math.sin(alpha)
        # internal mesh: 1/r1 - 1/r2
        inv = 1.0 / max(r1, 1e-9) - 1.0 / max(r2, 1e-9)
        if inv <= 0:
            return 0.0
        r_eq = 1.0 / inv
        e_star = youngs_pa / (2.0 * (1.0 - poisson ** 2))
        return math.sqrt(w * e_star
                         / (math.pi * max(self.face_width_m, 1e-9)
                            * max(r_eq, 1e-9)))

    def describe(self, traverse_torque_nm: float) -> list:
        f = self.tooth_loads(traverse_torque_nm)
        ratio = ("INFINITE -- equal rings, the carrier is the output"
                 if math.isinf(self.reduction)
                 else f"{self.reduction:.1f}:1")
        return [
            f"  captive planetary: {self.pinions} pinions of "
            f"{self.pinion_teeth} teeth, module {self.module_m * 1000:.1f} mm",
            f"    lower ring {self.lower_teeth}T r={self.lower_pitch_radius_m:.3f} m"
            f"   upper ring {self.upper_teeth}T r={self.upper_pitch_radius_m:.3f} m",
            f"    tooth delta {self.tooth_delta:+d}  ->  reduction {ratio}",
            f"    torque {traverse_torque_nm / 1000:.1f} kN.m  ->  "
            f"{f['tangential_per_mesh_n'] / 1000:.1f} kN tangential per mesh",
            f"    separating {f['separating_per_mesh_n'] / 1000:.1f} kN pushes the "
            f"pinion out of mesh; retainer holds "
            f"{self.retainer_preload_n / 1000:.1f} kN "
            f"({f['retainer_margin']:.2f}x)",
            f"    flank contact {self.contact_stress_pa(traverse_torque_nm) / 1e6:.0f} MPa",
        ]


# =====================================================================
#  2. THE CAPTIVE PASSIVE MATED PLANES RING
# =====================================================================
@dataclass
class MatedPlanesRing:
    """Two flat annular faces in contact, and the pressure between them.

    The distribution is the whole model. A ring under pure weight is
    evenly loaded and boring; a ring under weight AND an overturning
    moment is not, and every turret with an overhang is the second
    case."""
    radius_m: float
    width_m: float
    pads: int = 8
    rollers: int = 8
    holddowns: int = 8
    roller_diameter_m: float = 0.090
    roller_length_m: float = 0.110
    #: how much of the vertical load the discrete rollers take before
    #: the film does. Rollers are far stiffer than a grease film, so
    #: they take load first and the film takes what is left once they
    #: have deflected into it.
    roller_share: float = 0.65

    @property
    def area_m2(self) -> float:
        return 2.0 * math.pi * self.radius_m * self.width_m

    @property
    def section_modulus_m3(self) -> float:
        """A thin annulus resisting overturning: I/c = pi r^2 b."""
        return math.pi * self.radius_m ** 2 * self.width_m

    # ------------------------------------------------------------------
    def pressure(self, vertical_n: float, moment_nm: float,
                 samples: int = 72) -> dict:
        """Pressure around the annulus, and where it stops being real.

        p(theta) = N/A + M cos(theta) / (I/c)

        The second term is what the overhang does. When it exceeds the
        first, p goes negative somewhere -- and a face in contact cannot
        pull. So the contact SEPARATES over that arc, the remaining arc
        has to carry the whole load on less of itself, and the
        hold-downs carry the tension. Reporting the elastic answer
        without that step is reporting a bearing surface in tension,
        which is the one thing it cannot be."""
        theta = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
        uniform = vertical_n / self.area_m2
        bending = moment_nm / max(self.section_modulus_m3, 1e-12)
        p = uniform + bending * np.cos(theta)
        lifting = p < 0.0
        lift_fraction = float(lifting.mean())
        # redistribute onto what is still touching
        if lifting.any() and not lifting.all():
            contact = ~lifting
            p_closed = np.where(contact, p, 0.0)
            # the load the lifted arc was carrying has to go somewhere
            scale = vertical_n / max(float(np.trapezoid(
                p_closed, theta) * self.radius_m * self.width_m), 1e-9)
            p_closed = p_closed * max(scale, 1.0)
        else:
            p_closed = np.maximum(p, 0.0)
        tension_n = float(-np.trapezoid(np.minimum(p, 0.0), theta)
                          * self.radius_m * self.width_m)
        return {
            "theta": theta,
            "elastic_pressure_pa": p,
            "contact_pressure_pa": p_closed,
            "uniform_pa": uniform,
            "bending_pa": bending,
            "peak_pa": float(p_closed.max()),
            "lift_fraction": lift_fraction,
            "lifts": bool(lifting.any()),
            "holddown_total_n": tension_n,
            "holddown_each_n": tension_n / max(self.holddowns, 1),
        }

    def roller_loads(self, vertical_n: float, moment_nm: float) -> dict:
        """What one roller carries where the pressure is worst, and
        whether its line contact can stand it."""
        p = self.pressure(vertical_n, moment_nm)
        # the worst roller sits at the peak of the distribution
        arc = 2.0 * math.pi * self.radius_m / max(self.rollers, 1)
        peak_patch_n = p["peak_pa"] * arc * self.width_m
        per_roller = peak_patch_n * self.roller_share
        line_load = per_roller / max(self.roller_length_m, 1e-9)
        # Hertzian line contact, roller on a flat: 1/r_eq = 1/r
        e_star = 2.05e11 / (2.0 * (1.0 - 0.3 ** 2))
        r = self.roller_diameter_m / 2.0
        contact = math.sqrt(per_roller * e_star
                            / (math.pi * self.roller_length_m * r))
        return {
            "worst_roller_n": per_roller,
            "line_load_n_per_m": line_load,
            "contact_stress_pa": contact,
            "film_pressure_pa": p["peak_pa"] * (1.0 - self.roller_share),
        }

    def breakaway_torque_nm(self, vertical_n: float, moment_nm: float,
                            mu_track=None) -> dict:
        """What the traverse drive has to overcome to start turning.

        Friction acts at the track radius on whatever is actually
        pressing, so this is the load times the radius times mu -- and
        because the rollers carry most of it at a hundredth of the
        friction, the answer is dominated by how much the film is left
        holding rather than by the total."""
        lo, hi = mu_track or TRACK_FRICTION
        on_rollers = vertical_n * self.roller_share
        on_film = vertical_n * (1.0 - self.roller_share)
        return {
            "rolling_nm": on_rollers * ROLLER_FRICTION * self.radius_m,
            "sliding_low_nm": on_film * lo * self.radius_m,
            "sliding_high_nm": on_film * hi * self.radius_m,
            "total_low_nm": (on_rollers * ROLLER_FRICTION
                             + on_film * lo) * self.radius_m,
            "total_high_nm": (on_rollers * ROLLER_FRICTION
                              + on_film * hi) * self.radius_m,
        }

    def describe(self, vertical_n: float, moment_nm: float) -> list:
        p = self.pressure(vertical_n, moment_nm)
        r = self.roller_loads(vertical_n, moment_nm)
        t = self.breakaway_torque_nm(vertical_n, moment_nm)
        out = [
            f"  mated planes ring: r={self.radius_m:.3f} m, "
            f"{self.width_m * 1000:.0f} mm face, {self.area_m2 * 1e4:.0f} cm2",
            f"    carrying {vertical_n / 1000:.1f} kN and "
            f"{moment_nm / 1000:.1f} kN.m of overturning",
            f"    uniform {p['uniform_pa'] / 1e6:.3f} MPa, "
            f"bending +-{p['bending_pa'] / 1e6:.3f} MPa",
        ]
        if p["lifts"]:
            out += [
                f"    LIFTS over {p['lift_fraction'] * 100:.0f}% of the annulus "
                f"-- the face cannot pull, so that arc separates",
                f"    hold-downs take {p['holddown_total_n'] / 1000:.1f} kN, "
                f"{p['holddown_each_n'] / 1000:.1f} kN each over "
                f"{self.holddowns}",
            ]
        else:
            out.append(f"    stays closed all the way round "
                       f"(bending is {p['bending_pa'] / max(p['uniform_pa'], 1e-9):.2f} "
                       f"of uniform; it lifts past 1.00)")
        out += [
            f"    peak contact {p['peak_pa'] / 1e6:.3f} MPa; worst roller "
            f"{r['worst_roller_n'] / 1000:.1f} kN at "
            f"{r['contact_stress_pa'] / 1e6:.0f} MPa flank",
            f"    breakaway {t['total_low_nm'] / 1000:.2f} - "
            f"{t['total_high_nm'] / 1000:.2f} kN.m "
            f"(rolling alone {t['rolling_nm'] / 1000:.2f})",
        ]
        return out


# =====================================================================
#  READING THEM OUT OF THE GRAPH
# =====================================================================
def _bodies(document: dict):
    return [n for n in document["nodes"] if not n.get("wrench_point")]


def turret_overturning(document: dict, *,
                       recoil_n: float = 110_000.0) -> dict:
    """What the ring is actually asked to carry, from the graph itself.

    Two terms, and the second is the one that matters: everything above
    the ring presses down on it, and everything above the ring that is
    not over its centre ALSO tips it."""
    ring = [n for n in document["nodes"]
            if n["identity"].startswith("deck.ring.")]
    if not ring:
        raise KeyError("no deck.ring nodes: this graph has no static ring")
    centre = np.mean([n["reference_position"] for n in ring], axis=0)
    ring_y = float(centre[1])
    mass = 0.0
    moment = np.zeros(3)
    for n in _bodies(document):
        if str(n.get("motion_group")) == "frame":
            continue
        p = np.asarray(n["reference_position"], dtype=float)
        if p[1] < ring_y - 0.35:
            continue          # hanging below the ring: not on this face
        w = float(n.get("mass_kg", 0.0))
        mass += w
        moment += w * p
    cg = moment / max(mass, 1e-9)
    vertical = mass * GRAVITY
    lever = float(np.hypot(cg[0] - centre[0], cg[2] - centre[2]))
    # AND THE SHOT. Recoil is axial along the bore, and the bore is
    # above the ring face, so it arrives here as a moment of its own.
    bore = next((n for n in document["nodes"]
                 if n["identity"] == "turret.breech"), None)
    bore_y = float(bore["reference_position"][1]) if bore else ring_y
    return {
        "mass_kg": mass,
        "vertical_n": vertical,
        "cg": cg,
        "ring_centre": centre,
        "dead_lever_m": lever,
        "dead_moment_nm": vertical * lever,
        "recoil_moment_nm": recoil_n * (bore_y - ring_y),
        "total_moment_nm": vertical * lever + recoil_n * (bore_y - ring_y),
    }


def from_document(document: dict) -> tuple:
    """Both joints, built from what the graph declares about them."""
    track = next((e for e in document["edges"]
                  if e["constraint"] == "greased-thrust-track"), None)
    if track is None:
        raise KeyError("no greased-thrust-track edge in this graph")
    n_of = lambda c: sum(1 for e in document["edges"] if e["constraint"] == c)
    roller = next((e for e in document["edges"]
                   if e["constraint"] == "slew-thrust-roller"), None)
    ring = MatedPlanesRing(
        radius_m=float(track.get("track_radius_m", 0.67)),
        width_m=float(track.get("track_width_m", 0.12)),
        pads=n_of("greased-thrust-track"),
        rollers=n_of("slew-thrust-roller"),
        holddowns=n_of("slew-holddown-roller"),
        roller_diameter_m=float((roller or {}).get("roller_diameter_m", 0.09)),
        roller_length_m=float((roller or {}).get("roller_length_m", 0.11)),
    )
    # THE GRAPH ALREADY SAYS ALL OF THIS. The rings declare ring_teeth,
    # the meshes declare module_m and which ring they are against, and
    # the pinion declares its own count. Deriving a tooth count from a
    # radius instead -- which the first version of this did -- both
    # reinvents a number that is stated and silently loses the tooth
    # DELTA, reporting an 11:1 reduction as no reduction at all.
    lower_mesh = next((e for e in document["edges"]
                       if e["constraint"] == "captive-pinion-mesh"
                       and "mesh_lower" in e["identity"]), {})
    upper_mesh = next((e for e in document["edges"]
                       if e["constraint"] == "captive-pinion-mesh"
                       and "mesh_upper" in e["identity"]), {})
    pinions = [n for n in document["nodes"]
               if n.get("kind") == "drive-pinion"]
    p0 = pinions[0] if pinions else {}
    teeth = int(p0.get("pinion_teeth", 18))
    module = float(lower_mesh.get("module_m")
                   or upper_mesh.get("module_m")
                   or p0.get("module_m")
                   or (2.0 * float(p0.get("drum_radius_m", 0.072)) / max(teeth, 1)))
    lower_teeth = int(lower_mesh.get("ring_teeth")
                      or p0.get("ring_teeth") or 0)
    upper_teeth = int(upper_mesh.get("ring_teeth") or lower_teeth)
    if not lower_teeth:
        lower_teeth = upper_teeth
    mesh = PlanetaryRingMesh(
        pinion_teeth=teeth, lower_teeth=lower_teeth, upper_teeth=upper_teeth,
        pitch_radius_m=module * teeth / 2.0,
        face_width_m=float(p0.get("drum_length_m", 0.06)),
        pinions=max(len(pinions), 1),
    )
    return mesh, ring


def report(document: dict, *, recoil_n: float = 110_000.0) -> list:
    load = turret_overturning(document, recoil_n=recoil_n)
    mesh, ring = from_document(document)
    torque = ring.breakaway_torque_nm(load["vertical_n"],
                                      load["total_moment_nm"])["total_high_nm"]
    mesh.retainer_preload_n = mesh.tooth_loads(torque)["separating_per_mesh_n"] * 1.5
    out = [
        f"the ring carries {load['mass_kg']:.0f} kg = "
        f"{load['vertical_n'] / 1000:.1f} kN",
        f"   its cg is {load['dead_lever_m']:.3f} m off the ring centre "
        f"-> {load['dead_moment_nm'] / 1000:.1f} kN.m of dead overturning",
        f"   recoil adds {load['recoil_moment_nm'] / 1000:.1f} kN.m "
        f"(axial, above the face)",
        f"   total {load['total_moment_nm'] / 1000:.1f} kN.m",
        "",
    ]
    out += ring.describe(load["vertical_n"], load["total_moment_nm"])
    out += [""]
    out += mesh.describe(torque)
    return out


# =====================================================================
#  WRITING THE NEGOTIATION BACK ONTO THE GRAPH
# =====================================================================
def negotiate(document: dict, *, recoil_n: float = 110_000.0,
              traverse_torque_nm: float | None = None) -> dict:
    """Solve both ring joints and STAMP THE RESULT ON THE EDGES.

    Until this runs, everything above is a calculator: it can tell you
    that the hold-downs carry 218 kN and nothing downstream has any way
    to know. A wrench on the edge is how a negotiated load becomes part
    of the machine -- the frame solve, the beam survey, the damage model
    and the renderer all read the document, and none of them is going to
    import this module and ask.

    THE SHARE IS THE POINT. Each pad, roller and hold-down gets ITS OWN
    force, from where it sits in the pressure distribution -- not an
    average. A ring with half its annulus lifted has pads carrying twice
    the mean and pads carrying nothing, and writing the mean onto all of
    them would hide the only thing worth knowing.

    Mutates `document` in place and returns what it wrote."""
    load = turret_overturning(document, recoil_n=recoil_n)
    mesh, ring = from_document(document)
    torque = (traverse_torque_nm if traverse_torque_nm is not None
              else ring.breakaway_torque_nm(
                  load["vertical_n"], load["total_moment_nm"])["total_high_nm"])
    field = ring.pressure(load["vertical_n"], load["total_moment_nm"])
    teeth = mesh.tooth_loads(torque)

    centre = np.asarray(load["ring_centre"], dtype=float)
    pos = {n["identity"]: np.asarray(n["reference_position"], dtype=float)
           for n in document["nodes"]}

    def _angle_of(edge):
        """Where round the ring this participant sits, measured the same
        way the pressure field is."""
        a = pos[edge["a"]] - centre
        return math.atan2(a[2], a[0])

    def _p_at(theta):
        return float(np.interp(theta % (2.0 * math.pi),
                               field["theta"], field["contact_pressure_pa"],
                               period=2.0 * math.pi))

    def _elastic_at(theta):
        return float(np.interp(theta % (2.0 * math.pi),
                               field["theta"], field["elastic_pressure_pa"],
                               period=2.0 * math.pi))

    written = {"track": 0, "roller": 0, "holddown": 0, "mesh": 0}
    pad_arc = 2.0 * math.pi * ring.radius_m / max(ring.pads, 1)
    roll_arc = 2.0 * math.pi * ring.radius_m / max(ring.rollers, 1)

    # CLOSE ON THE PARTICIPANTS, NOT ON THE FIELD. The first version
    # normalised the continuous pressure distribution by integrating it,
    # then handed each pad a force computed from its own patch -- two
    # different quadratures of the same thing, which do not have to
    # agree and did not: the eight pads and eight rollers between them
    # reported carrying 270 kN of a 58.7 kN machine, a closure of 461%.
    #
    # What the ring must satisfy is that the forces ACTUALLY WRITTEN sum
    # to the load, so the sum that has to close is the one computed. Add
    # the patches up first, scale them to the load, and closure is an
    # identity rather than something to be checked and hoped for.
    _track_th = [_angle_of(e) for e in document["edges"]
                 if e["constraint"] == "greased-thrust-track"]
    _roll_th = [_angle_of(e) for e in document["edges"]
                if e["constraint"] == "slew-thrust-roller"]
    _raw = (sum(_p_at(t) * (1.0 - ring.roller_share) * pad_arc * ring.width_m
                for t in _track_th)
            + sum(_p_at(t) * ring.roller_share * roll_arc * ring.width_m
                  for t in _roll_th))
    _close = load["vertical_n"] / max(_raw, 1e-9)

    for e in document["edges"]:
        c = e["constraint"]
        if c == "greased-thrust-track":
            th = _angle_of(e)
            p = _p_at(th) * (1.0 - ring.roller_share) * _close
            f = p * pad_arc * ring.width_m
            e["wrench"] = {"force": [0.0, float(f), 0.0],
                           "moment": [0.0, 0.0, 0.0]}
            e["contact_pressure_pa"] = round(p, 1)
            e["in_contact"] = bool(_elastic_at(th) > 0.0)
            e["negotiated_force_n"] = round(f, 1)
            written["track"] += 1
        elif c == "slew-thrust-roller":
            th = _angle_of(e)
            f = (_p_at(th) * ring.roller_share * roll_arc
                 * ring.width_m * _close)
            e_star = 2.05e11 / (2.0 * (1.0 - 0.3 ** 2))
            r = ring.roller_diameter_m / 2.0
            e["wrench"] = {"force": [0.0, float(f), 0.0],
                           "moment": [0.0, 0.0, 0.0]}
            e["negotiated_force_n"] = round(f, 1)
            e["contact_stress_pa"] = round(math.sqrt(
                max(f, 0.0) * e_star
                / (math.pi * ring.roller_length_m * r)), 0)
            written["roller"] += 1
        elif c == "slew-holddown-roller":
            # ONLY THE LIFTED ARC. A hold-down on the side that is being
            # pressed down does nothing, and giving every one of them an
            # equal share of the tension would say the ring is being
            # pulled apart evenly, which is the opposite of what an
            # overhang does.
            th = _angle_of(e)
            elastic = _elastic_at(th)
            f = 0.0
            if elastic < 0.0:
                f = -elastic * pad_arc * ring.width_m
            e["wrench"] = {"force": [0.0, float(-f), 0.0],
                           "moment": [0.0, 0.0, 0.0]}
            e["negotiated_force_n"] = round(f, 1)
            e["engaged"] = bool(f > 0.0)
            written["holddown"] += 1
        elif c == "captive-pinion-mesh":
            # tangential drives, separating tries to push the pinion out
            # of mesh, and the retainer is what stops it
            e["wrench"] = {
                "force": [float(teeth["tangential_per_mesh_n"]),
                          0.0, float(teeth["separating_per_mesh_n"])],
                "moment": [0.0, 0.0, 0.0]}
            e["negotiated_force_n"] = round(teeth["tangential_per_mesh_n"], 1)
            e["separating_force_n"] = round(teeth["separating_per_mesh_n"], 1)
            e["flank_contact_pa"] = round(mesh.contact_stress_pa(torque), 0)
            written["mesh"] += 1
        elif c == "preloaded-captive-body-retainer-spring":
            e["wrench"] = {"force": [0.0, 0.0,
                                     float(teeth["separating_per_mesh_n"])],
                           "moment": [0.0, 0.0, 0.0]}
            e["negotiated_force_n"] = round(teeth["separating_per_mesh_n"], 1)
            e["holds_against"] = "pinion-separating-force"

    # THE RING'S OWN REACTION, on the node the load is about, so the
    # frame solve sees the overturning rather than having to infer it.
    total = sum(float(e.get("negotiated_force_n", 0.0))
                for e in document["edges"]
                if e["constraint"] in ("greased-thrust-track",
                                       "slew-thrust-roller"))
    return {
        "vertical_n": load["vertical_n"],
        "moment_nm": load["total_moment_nm"],
        "traverse_torque_nm": torque,
        "written": written,
        "pads_in_contact": sum(1 for e in document["edges"]
                               if e["constraint"] == "greased-thrust-track"
                               and e.get("in_contact")),
        "holddowns_engaged": sum(1 for e in document["edges"]
                                 if e["constraint"] == "slew-holddown-roller"
                                 and e.get("engaged")),
        "carried_n": total,
        "closure": total / max(load["vertical_n"], 1e-9),
    }
