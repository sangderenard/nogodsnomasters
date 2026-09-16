"""A floor-standing centrifuge as a real machine, in the prism vocabulary.

WHY THIS MACHINE. It is the one whose damage comes from being unlevel.
An autoclave stands still; a centrifuge turns a heavy bowl at several
thousand rpm on a vertical spindle, and a vertical rotor is the one
whose gyroscopic moment the feet can react -- so its rocking modes
split with speed, its run-up crosses them, and a foot wound short of
the floor turns a machine that hums into one that hammers. The mode
table is what shows that, and this is the machine to show it on.

THE FRAME IS DECLARED, NOT DRAWN. A machine frame is not a roll cage:
it is square hollow section welded into a box, or angle iron bolted at
the corners, with sheet cladding screwed to it every few inches. Those
are the sections and seams milspec and joints now carry, and this
module builds the same machine on either frame so the table can be
asked what the choice costs:

    square-tube-welded   6061-T6 square tube, welded: light and rigid
    angle-bolted         A36 angle, bolted corners: cheap and heavier,
                         and its least-axis stiffness is a third of
                         the tube's, which is why it wants bracing

The casing is sheet on a screwed seam in both cases, because that is
how a casing is fitted: it locates the panel and it is not a load path
anyone should trust, and the walk will say so through the seam's own
stiffness rather than through anyone's opinion.

WHAT IS NOT MODELLED, so that nobody reads it in: the frame's own
bending. The mode table stands a rigid frame on its feet; the members
here carry real sections so a beam solve can be run on them, but the
three-freedom reduction does not run one. A frame soft enough for its
own bending to matter would show up there, not here.
"""
from __future__ import annotations

import math

import numpy as np

from prism_bodies import Prism, emit_prism
from turret_production import ProductionGraph
from milspec import AngleSection, SquareTubeSection, SheetSection
from joints import Seam
from centrifuges import Centrifuge, _bowl_inertia, centrifuge_class
from operating_states import Cycle, OperatingState, run_up, coast

#: The two frames a centrifuge of this size is actually built on.
FRAMES = {
    "square-tube-welded": dict(
        section=SquareTubeSection(side_m=0.040, wall_m=0.003, material="6061t6",
                                  designation="40x40x3 SHS 6061-T6"),
        constraint="rigid-distance", joint="solid-welded"),
    "angle-bolted": dict(
        section=AngleSection(leg_m=0.040, thickness_m=0.004, material="a36",
                             designation="40x40x4 angle A36"),
        constraint="bolted-flange-mount", joint="bolted-flange"),
}

#: Casing sheet and how it is fastened: 1.2 mm steel, a #10 screw every
#: three inches, which is the ordinary tinsmith's pitch.
CASING = dict(sheet=SheetSection(width_m=0.30, thickness_m=0.0012, material="a36",
                                 designation="1.2 mm sheet"),
              seam=Seam(fastener="sheet-metal-screw", per_inch=1.0 / 3.0,
                        sheet_thickness_m=0.0012, fastener_diameter_m=0.0048))

FRAME_LENGTH_M = 0.70
FRAME_WIDTH_M = 0.60
FRAME_HEIGHT_M = 0.30
#: An electric motor of this class, by its frame size: a 7.5 kW 132
#: frame is about 55 kg and turns at 2900 rpm.
MOTOR_MASS_PER_KW = 7.5
MOTOR_RPM = 2900.0


def bowl_mass_kg(unit: Centrifuge) -> float:
    spec = unit.spec
    return unit.bowl_mass_kg or (unit.bowl_volume_m3 * spec.solidity
                                 * spec.bowl_density_kg_m3)


def build(identity: str = "plant.centrifuge", *, unit: Centrifuge | None = None,
          frame: str = "square-tube-welded",
          bowl_unbalance_kg_m: float | None = None,
          bowl_balance_grade_mm_s: float = 2.5,
          motor_offset_m: float = 0.14, bowl_offset_m: float = -0.20
          ) -> ProductionGraph:
    """The centrifuge, on the named frame.

    `bowl_unbalance_kg_m` is the residual unbalance the bowl is running
    with; left None it comes from the balance grade, which is the bowl
    as the balancing machine last saw it. A bowl with a sludge cake in
    it is not that bowl, and the cycle can say so per state."""
    if frame not in FRAMES:
        raise KeyError(f"unknown frame {frame!r}; declare one of: "
                       f"{', '.join(sorted(FRAMES))}")
    spec = FRAMES[frame]
    unit = unit or Centrifuge(kind="solid-bowl-batch", bowl_radius_m=0.150,
                              inner_radius_m=0.060, bowl_height_m=0.180,
                              speed_rpm=6000.0, motor_kw=7.5)
    g = ProductionGraph(identity)
    g.motion_group = "frame"

    hx, hy, hz = FRAME_LENGTH_M / 2.0, FRAME_HEIGHT_M / 2.0, FRAME_WIDTH_M / 2.0
    section = spec["section"]
    # the frame's own mass is its members: twelve edges of a box
    perimeter = 4.0 * (FRAME_LENGTH_M + FRAME_WIDTH_M + FRAME_HEIGHT_M)
    frame_mass = perimeter * section.mass_per_m_kg

    # ---- the frame: a box of the declared section --------------------
    # Emitted as a prism so it has ports -- the four feet on its
    # underside, the spindle seat and the motor pad on top -- and as
    # twelve real members of the declared section between its corners,
    # so a beam solve has the frame and not a picture of it.
    box = Prism(identity=f"{identity}.frame", kind="machine-frame",
                centre=(0.0, hy, 0.0), shape="box", half_extent=(hx, hy, hz),
                material=section.material, mass_kg=frame_mass,
                attributes={"in_view": "plant", "frame": frame,
                            "section": section.designation})
    feet = {"foot_fl": box.face_point("-y", across=(-hx * 0.9, -hz * 0.9)),
            "foot_fr": box.face_point("-y", across=(-hx * 0.9, +hz * 0.9)),
            "foot_rl": box.face_point("-y", across=(+hx * 0.9, -hz * 0.9)),
            "foot_rr": box.face_point("-y", across=(+hx * 0.9, +hz * 0.9)),
            # THE BOWL AND THE MOTOR BALANCE EACH OTHER ACROSS THE FRAME.
            # With the motor hung out at 220 mm and the bowl near the
            # middle the CG sat 90 mm aft and the front feet carried 47 N
            # of a 1040 N machine -- so they left the floor at the first
            # gram of unbalance, and every levelling result was buried
            # under a layout fault. A real purifier puts the bowl and the
            # motor either side of the frame's centre, and so does this.
            "spindle": box.face_point("+y", across=(bowl_offset_m, 0.0)),
            "motor_pad": box.face_point("+y", across=(motor_offset_m, 0.0)),
            "clad_l": box.face_point("-z"), "clad_r": box.face_point("+z")}
    f_nodes = emit_prism(g, box, feet, assembly="centrifuge",
                         port_kwargs={"bolt_radius_m": 0.008})
    # the feet are mounts; the rest of the ports are not
    for name in ("foot_fl", "foot_fr", "foot_rl", "foot_rr"):
        node = next(n for n in g.nodes if n["identity"] == f_nodes[name])
        node.update(port_role="structural-mount", joint=spec["joint"],
                    outward=(0.0, -1.0, 0.0))
    corners = {}
    for i, (sx, sy, sz) in enumerate(((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1),
                                      (-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1))):
        ident = f"{identity}.frame.corner.{i}"
        g.node(ident, (sx * hx, hy + sy * hy, sz * hz), "frame-corner",
               material=section.material, mass_kg=0.0, mass_in_total=False,
               joint=spec["joint"], half_extent_m=(0.02, 0.02, 0.02),
               in_view="plant")
        corners[i] = ident
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)):
        g.edge(f"{identity}.frame.member.{a}{b}", corners[a], corners[b],
               spec["constraint"], section=section, part_role="frame-member",
               body_internal=True, rigid=True)

    # ---- the bowl on its spindle -------------------------------------
    bm = bowl_mass_kg(unit)
    bowl = Prism(identity=f"{identity}.bowl", kind="centrifuge-bowl",
                 centre=(bowl_offset_m, FRAME_HEIGHT_M + 0.10 + unit.bowl_height_m / 2.0, 0.0),
                 shape="cylinder", axis=(0.0, 1.0, 0.0), radius=unit.bowl_radius_m,
                 length=unit.bowl_height_m, material="stainless-plate", mass_kg=bm,
                 attributes={"in_view": "plant",
                             "rotor_axis": (0.0, 1.0, 0.0),
                             "rotor_rated_rpm": float(unit.speed_rpm),
                             "rotor_inertia_kg_m2": float(_bowl_inertia(unit)),
                             "rotor_mass_kg": bm,
                             **({"rotor_unbalance_kg_m": float(bowl_unbalance_kg_m)}
                                if bowl_unbalance_kg_m is not None
                                else {"balance_grade_mm_s": bowl_balance_grade_mm_s}),
                             "runs_in": ("spin-up", "run", "coast")})
    b_ports = {"spindle": bowl.end_point("-")}
    b_nodes = emit_prism(g, bowl, b_ports, assembly="centrifuge")
    # the spindle: a short stiff column from the frame's seat to the
    # bowl's underside, in the frame's own section for want of a shaft
    g.edge(f"{identity}.spindle", f_nodes["spindle"], b_nodes["spindle"],
           "rigid-distance", radius=0.020, alloy="4340qt", part_role="spindle")

    # ---- the motor on its pad, driving by belt -----------------------
    mkg = unit.motor_kw * MOTOR_MASS_PER_KW
    motor = Prism(identity=f"{identity}.motor", kind="electric-motor",
                  centre=(motor_offset_m, FRAME_HEIGHT_M + 0.11, 0.0),
                  shape="cylinder", axis=(0.0, 1.0, 0.0), radius=0.09, length=0.22,
                  material="cast-iron", mass_kg=mkg,
                  attributes={"in_view": "plant",
                              "rotor_axis": (0.0, 1.0, 0.0),
                              "rotor_rated_rpm": MOTOR_RPM,
                              "rotor_shape": "solid-disc",
                              "rotor_mass_kg": mkg * 0.35, "rotor_radius_m": 0.05,
                              "balance_grade_mm_s": 6.3,
                              "runs_in": ("spin-up", "run")})
    m_ports = {"base": motor.end_point("-")}
    m_nodes = emit_prism(g, motor, m_ports, assembly="centrifuge")
    g.edge(f"{identity}.motor.mount", f_nodes["motor_pad"], m_nodes["base"],
           "bolted-flange-mount", radius=0.012, part_role="motor-feet")

    # ---- the casing: sheet on a screwed seam, both flanks ------------
    for side, port in (("l", "clad_l"), ("r", "clad_r")):
        z = -hz if side == "l" else +hz
        panel = Prism(identity=f"{identity}.casing.{side}", kind="casing-panel",
                      centre=(0.0, hy + 0.15, z + (0.0 if side == "l" else 0.0)),
                      shape="box", half_extent=(hx, 0.25, 0.0006),
                      material="steel-sheet",
                      mass_kg=2.0 * hx * 0.50 * CASING["sheet"].thickness_m
                      * CASING["sheet"].mat.density_kg_m3,
                      attributes={"in_view": "plant"})
        pn = emit_prism(g, panel, {"seam": panel.face_point("-y")},
                        assembly="centrifuge")
        g.edge(f"{identity}.casing.{side}.seam", f_nodes[port], pn["seam"],
               "screwed-seam", section=CASING["sheet"], seam=CASING["seam"],
               part_role="casing-seam")
    return g


def cycle(identity: str = "plant.centrifuge", *, sludge_unbalance_kg_m: float = 0.0,
          live_vibration: bool = False) -> Cycle:
    """Spin up, run, coast down.

    `sludge_unbalance_kg_m` is the cake. A bowl that has thrown solids
    unevenly runs with far more unbalance than it was balanced to, and
    that is the state the run is evaluated in; the spin-up is the clean
    bowl, because the cake builds during the run."""
    bowl = f"{identity}.bowl"
    over = {bowl: sludge_unbalance_kg_m} if sludge_unbalance_kg_m > 0.0 else {}
    return Cycle(machine=identity, states=(
        OperatingState("spin-up", curve=run_up(24), note="clean bowl, motor driving"),
        OperatingState("run", unbalance_kg_m=over, note="at speed, cake building"),
        OperatingState("coast", curve=coast(12), unbalance_kg_m=over,
                       speeds={f"{identity}.motor": 0.0},
                       note="motor off, bowl coasting with its cake"),
    ), live_vibration=live_vibration)
