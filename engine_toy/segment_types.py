"""The segment palette: every kind of thing an object can be made of.

This is the catalogue a chassis editor paints with. Each entry is one
SEGMENT TYPE from the production game's own vocabulary -- the constraint
strings `abstract_ui_vehicles._vehicle_mechanical_graph` actually
authors -- together with the parameters that kind carries and sensible
defaults for them.

The point of having it as data rather than as code is that objects stop
being things somebody wrote a builder function for. An object becomes a
list of nodes and a list of segments, each segment naming a type from
here, and anything that can produce such a list -- an editor, a script,
a generator -- can make objects the production physics will run.

EXPANDING IT IS THE NORMAL CASE. The production vocabulary is what the
game ships with; the `family="engine-toy"` entries at the bottom are
ours, added because a machine needed something the vehicle graph never
had (a prismatic guide, a commanded-length hoist ram). Adding a type is
adding a row here, and the editor picks it up with no further work.

FAMILIES, and what the physics does with each:

  structural   a load path with a rest length. Carries axial and shear,
               yields and fractures per the member material law.
  joint        constrains some degrees of freedom and frees others, with
               a six-axis bushing for the compliance that remains.
  spring       stores bounded energy and opposes relative velocity.
  actuator     a member whose REST LENGTH or ANGLE is commanded from
               outside; series compliance makes it backdrivable rather
               than a rigid position source.
  drive        transmits torque rather than force.
  routed       carries fluid or current, not load: no damage law, no
               bushing, and it does not hold anything up.
  breakable    a mount designed to fail first, at a declared load, so
               the failure happens where the designer chose.
  impulse      a load path for a discrete event rather than a continuous
               force -- a shot, an impact.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentType:
    key: str                       # the constraint string the graph carries
    family: str
    label: str                     # what to call it in a palette
    default_radius_m: float = 0.012
    palette_role: str = "rollbar-silver"
    # parameters this kind carries, with defaults an editor can offer
    parameters: tuple = ()
    note: str = ""
    origin: str = "production"     # "production" | "engine-toy"

    @property
    def carries_load(self) -> bool:
        return self.family not in ("routed",)

    def defaults(self) -> dict:
        return {name: value for name, value in self.parameters}


def _p(**kw) -> tuple:
    return tuple(kw.items())


SEGMENT_TYPES: tuple[SegmentType, ...] = (
    # ---------------- structural ----------------
    SegmentType("rigid-distance", "structural", "rigid member", 0.012, "rollbar-silver",
                note="the ordinary structural tube: a rest length it wants to keep"),
    SegmentType("rigid-offset", "structural", "rigid offset", 0.010, "rollbar-silver",
                note="a short rigid stand-off holding one node clear of another"),
    SegmentType("elastic-plastic-structural-post", "structural", "structural post", 0.016,
                "chassis-grey", note="a column meant to take compression and buckle honestly"),
    SegmentType("elastic-plastic-structural-body-pin", "structural", "body pin", 0.014,
                "chassis-grey", note="a pin loaded in bending"),
    SegmentType("rigid-bulkhead-frame-rail", "structural", "bulkhead rail", 0.022, "chassis-grey"),
    SegmentType("rigid-bulkhead-panel-spoke", "structural", "bulkhead spoke", 0.010, "chassis-grey"),
    SegmentType("rigid-brush-guard", "structural", "brush guard", 0.018, "chassis-grey"),
    SegmentType("rigid-caliper-mount", "structural", "caliper mount", 0.014, "chassis-grey"),
    SegmentType("rigid-rotor-mount", "structural", "rotor mount", 0.016, "chassis-grey"),

    # ---------------- joints ----------------
    SegmentType("rotational-bearing", "joint", "rotational bearing", 0.020, "rollbar-silver",
                parameters=_p(free_axis="member-axis", friction_torque_nm=1.8),
                note="frees rotation about its own axis, holds everything else"),
    SegmentType("gimbal-yaw-bearing", "joint", "yaw bearing (traverse)", 0.035, "rollbar-silver",
                parameters=_p(free_axis="world-vertical", friction_torque_nm=4.0),
                note="turns about the vertical: a turret's traverse"),
    SegmentType("gimbal-pitch-bearing", "joint", "pitch bearing (elevation)", 0.030,
                "rollbar-silver", parameters=_p(free_axis="member-lateral", friction_torque_nm=3.0),
                note="nose up and down: a turret's elevation, a trunnion"),
    SegmentType("actuated-damped-clutch-gimbal-base", "joint", "clutched gimbal base", 0.028,
                "actuator-yellow",
                parameters=_p(clutch_engagement=1.0, angular_stiffness_nm_per_rad=8200.0,
                              angular_damping_nm_s_per_rad=680.0, holding_torque_nm=3400.0),
                note="a gimbal that can be clamped: release it and the payload swings free"),
    SegmentType("six-axis-compliant-mount", "joint", "compliant mount", 0.016, "chassis-grey",
                parameters=_p(linear_stiffness_n_per_m=2.5e6, linear_damping_n_s_per_m=4.0e3),
                note="a rubber mount: stiff enough to locate, soft enough to isolate"),
    SegmentType("replaceable-sacrificial-knuckle-bushing", "joint", "sacrificial bushing", 0.014,
                "actuator-yellow",
                parameters=_p(yield_force_n=46_000.0, fracture_force_n=72_000.0),
                note="fails before the parts either side of it do, and is replaceable"),
    SegmentType("universal-joint-steering-shaft", "joint", "universal joint", 0.014,
                "rollbar-silver", note="torque around a corner"),

    # ---------------- springs and dampers ----------------
    SegmentType("spring-damper", "spring", "spring / damper", 0.018, "actuator-yellow",
                parameters=_p(stiffness_n_per_m=45_000.0, compression_damping_n_s_per_m=2_600.0,
                              rebound_damping_n_s_per_m=3_400.0, preload_compression_m=0.010),
                note="stores bounded energy, opposes relative velocity"),
    SegmentType("preloaded-bumper-shock-absorber", "spring", "bumper shock", 0.020,
                "actuator-yellow",
                parameters=_p(stiffness_n_per_m=120_000.0, maximum_force_n=38_000.0)),
    SegmentType("preloaded-captive-body-retainer-spring", "spring", "retainer spring", 0.012,
                "actuator-yellow",
                parameters=_p(stiffness_n_per_m=18_000.0, preload_compression_m=0.006,
                              maximum_force_n=2_400.0)),
    SegmentType("torsion-stabilizer", "spring", "anti-roll bar", 0.016, "rollbar-silver",
                parameters=_p(angular_stiffness_nm_per_rad=14_000.0),
                note="ties two corners together in roll and leaves them free in bump"),
    SegmentType("tension-limit-strap", "spring", "limit strap", 0.008, "chassis-grey",
                parameters=_p(free_length_m=0.18, maximum_force_n=26_000.0),
                note="does nothing at all until it is tight, then stops travel"),

    # ---------------- actuators ----------------
    SegmentType("linear-hydraulic-actuator", "actuator", "hydraulic ram", 0.030,
                "actuator-yellow",
                parameters=_p(commanded_rest_length_m=0.0, minimum_rest_length_m=0.0,
                              maximum_rest_length_m=0.500, bore_m=0.080, rod_m=0.045,
                              linear_stiffness_n_per_m=1.8e7, linear_damping_n_s_per_m=6.0e4,
                              holding_force_n=90_000.0, relief_force_n=140_000.0,
                              maximum_relief_stroke_m=0.020),
                note="REST LENGTH is commanded from outside; series relief keeps it "
                     "backdrivable instead of a rigid position source"),
    SegmentType("telescopic-hydraulic-diagonal", "actuator", "telescopic ram", 0.034,
                "actuator-yellow",
                parameters=_p(commanded_rest_length_m=0.0, stages=3,
                              maximum_rest_length_m=1.800),
                note="long stroke from a short body; force steps down as stages take over"),
    SegmentType("table-actuator-linear-link", "actuator", "linear link actuator", 0.016,
                "actuator-yellow", parameters=_p(commanded_rest_length_m=0.0)),
    SegmentType("rack-and-pinion-angle-to-translation", "actuator", "rack and pinion", 0.018,
                "actuator-yellow", parameters=_p(pinion_radius_m=0.025),
                note="turns an angle into a travel, or the other way about"),
    SegmentType("rack-translation", "actuator", "rack travel", 0.014, "actuator-yellow"),

    # ---------------- drive ----------------
    SegmentType("torque-shaft", "drive", "torque shaft", 0.020, "rollbar-silver"),
    SegmentType("direct-torque-shaft", "drive", "direct shaft", 0.022, "rollbar-silver"),
    SegmentType("constant-velocity-torque-shaft", "drive", "CV shaft", 0.020, "rollbar-silver",
                note="passes torque through an angle without the speed pulsing"),
    SegmentType("continuously-variable-torque-shaft", "drive", "CVT shaft", 0.020,
                "rollbar-silver", parameters=_p(ratio=1.0)),
    SegmentType("geared-timing-drive", "drive", "timing drive", 0.014, "rollbar-silver",
                parameters=_p(ratio=2.0)),
    SegmentType("accessory-drive-belt", "drive", "accessory belt", 0.008, "chassis-grey",
                parameters=_p(ratio=1.0, slip_torque_nm=45.0)),
    SegmentType("friction-brake-torque-couple", "drive", "friction brake", 0.024, "actuator-yellow",
                parameters=_p(maximum_torque_nm=4_200.0, applied_fraction=0.0)),
    SegmentType("starter-drive-engagement", "drive", "starter engagement", 0.012, "chassis-grey"),
    SegmentType("synchronized-positive-dog-clutch-bypass", "drive", "dog clutch", 0.020,
                "actuator-yellow", parameters=_p(engaged=0.0)),
    SegmentType("selectable-wheel-end-locking-hub", "drive", "locking hub", 0.018, "chassis-grey",
                parameters=_p(engaged=1.0)),
    SegmentType("steering-torque-shaft", "drive", "steering shaft", 0.014, "rollbar-silver"),
    SegmentType("steering-link", "drive", "steering link", 0.012, "rollbar-silver"),
    SegmentType("steering-assist-torque-coupling", "drive", "steering assist", 0.016,
                "actuator-yellow", parameters=_p(assist_torque_nm=180.0)),

    # ---------------- routed: fluid and current ----------------
    SegmentType("pressure-rated-hydraulic-line", "routed", "hydraulic hard line", 0.008,
                "hydraulic-red", parameters=_p(working_pressure_pa=25e6)),
    SegmentType("flexible-hydraulic-hose", "routed", "hydraulic hose", 0.010, "hydraulic-red",
                parameters=_p(working_pressure_pa=25e6)),
    SegmentType("rigid-brake-hard-line", "routed", "brake hard line", 0.004, "hydraulic-red"),
    SegmentType("flexible-brake-hose", "routed", "brake hose", 0.006, "hydraulic-red"),
    SegmentType("pressure-rated-air-line", "routed", "air line", 0.008, "pneumatic-blue",
                parameters=_p(working_pressure_pa=1.0e6)),
    SegmentType("low-pressure-air-line", "routed", "low-pressure air", 0.010, "pneumatic-blue"),
    SegmentType("flexible-air-line", "routed", "flexible air line", 0.008, "pneumatic-blue"),
    SegmentType("rigid-pneumatic-hard-line", "routed", "pneumatic hard line", 0.006,
                "pneumatic-blue"),
    SegmentType("flexible-pneumatic-hose", "routed", "pneumatic hose", 0.008, "pneumatic-blue"),
    SegmentType("oil-line", "routed", "oil line", 0.008, "lube-amber"),
    SegmentType("coolant-line", "routed", "coolant line", 0.012, "coolant-green"),
    SegmentType("nitrous-delivery-line", "routed", "nitrous line", 0.005, "pneumatic-blue"),
    SegmentType("intake-flow-path", "routed", "intake path", 0.030, "intake-blue"),
    SegmentType("exhaust-flow-path", "routed", "exhaust path", 0.028, "exhaust-orange"),
    SegmentType("heat-exchange-path", "routed", "heat exchange path", 0.016, "coolant-green"),
    SegmentType("insulated-copper-wire", "routed", "wire", 0.0035, "wire-copper",
                parameters=_p(maximum_current_a=48.0)),
    SegmentType("routed-energy-line", "routed", "energy line", 0.006, "wire-copper"),
    SegmentType("routed-tension-cable", "routed", "cable", 0.004, "chassis-grey"),
    SegmentType("sheathed-parking-brake-cable", "routed", "handbrake cable", 0.004, "chassis-grey"),

    # ---------------- breakable mounts ----------------
    SegmentType("breakable-six-axis-bolt-pattern", "breakable", "bolted mount", 0.016,
                "chassis-grey",
                parameters=_p(mount_yield_force_n=64_000.0, mount_fracture_force_n=96_000.0),
                note="fails at a declared load, so the failure is where you chose"),
    SegmentType("breakable-six-axis-braze-on", "breakable", "brazed mount", 0.012, "chassis-grey",
                parameters=_p(mount_yield_force_n=28_000.0, mount_fracture_force_n=41_000.0)),
    SegmentType("breakable-six-axis-body-pin-foot", "breakable", "body-pin foot", 0.016,
                "chassis-grey",
                parameters=_p(mount_yield_force_n=72_000.0, mount_fracture_force_n=118_000.0)),
    SegmentType("breakable-body-shell-mount", "breakable", "shell mount", 0.012, "chassis-grey"),

    # ---------------- impulse ----------------
    SegmentType("point-impulse-wrench-coupling", "impulse", "impulse coupling", 0.018,
                "actuator-yellow",
                parameters=_p(load_path="individual-shot-recoil-r-cross-impulse",
                              impulse_n_s=0.0),
                note="a discrete event's load path: a shot, an impact"),

    # =================================================================
    #  OURS -- added because a machine needed something the vehicle
    #  graph never had. This is the normal way this palette grows.
    # =================================================================
    SegmentType("prismatic-guide-strut", "joint", "prismatic guide", 0.028, "chassis-grey",
                parameters=_p(free_axis="world-vertical", lateral_stiffness_n_per_m=5.5e7,
                              lateral_damping_n_s_per_m=4.2e4),
                origin="engine-toy",
                note="frees one straight axis and holds the rest: what stops a hoisted "
                     "carriage swinging on the end of its ram"),
    SegmentType("oleo-pneumatic-strut", "spring", "oleo strut", 0.030, "actuator-yellow",
                parameters=_p(gas_charge_pressure_pa=2.0e6, gas_volume_m3=0.004,
                              orifice_area_m2=1.0e-4, polytropic_n=1.35, stroke_m=0.320),
                origin="engine-toy",
                note="gas spring and oil damper in one body: landing gear, hydropneumatic "
                     "suspension, and a gun's recoil system are all this part"),
    SegmentType("oleo-recoil-slide", "spring", "recoil slide", 0.036, "actuator-yellow",
                parameters=_p(slide_axis="member-axis", recoiling_mass_kg=170.0,
                              recoil_stroke_m=0.40, gas_charge_pressure_pa=5.0e5,
                              gas_volume_m3=0.008, orifice_area_m2=3.0e-4),
                origin="engine-toy",
                note="a barrel riding in machined ways: gas returns it to battery, "
                     "an orifice brake meters the return -- the real mechanism behind "
                     "a trunnion pin with a hole through it"),
    SegmentType("spherical-thrust-seat", "joint", "spherical thrust seat", 0.055,
                "rollbar-silver",
                parameters=_p(seat_radius_m=0.075, allowable_bearing_pa=120.0e6,
                              free_rotation=True, axial_stiffness_n_per_m=5.9e10),
                origin="engine-toy",
                note="stiff along the thrust axis, free in rotation. The one part that "
                     "makes a fine-aim stage and a firing load path coexist, because "
                     "recoil is axial and aiming is angular and a sphere can be rigid "
                     "to one while free to the other"),
    SegmentType("belleville-preload-stack", "spring", "preload stack", 0.038,
                "actuator-yellow",
                parameters=_p(preload_n=24_000.0, stack_rate_n_per_m=1.0e7,
                              free_length_m=0.090),
                origin="engine-toy",
                note="a stack of disc springs holding a joint closed. Enormous force "
                     "in a short stack, and nearly constant across its working travel, "
                     "which is what a preload has to be"),
    SegmentType("firing-lock-clamp", "structural", "firing lock", 0.048,
                "rollbar-silver",
                parameters=_p(engaged=True, friction_coefficient=0.15,
                              clamp_pressure_pa=40.0e6, clamp_area_m2=0.080,
                              engage_time_s=0.035, release_time_s=0.035),
                origin="engine-toy",
                note="a FRICTION clamp that holds wherever it is closed, which is the "
                     "whole point: a cone, a taper or any form-fitting detent would "
                     "drag the gun back to nominal as it shut and destroy the fine "
                     "correction it was closed to preserve. Closed, it ties the cradle "
                     "straight to the trunnion so the shot BYPASSES the fine stage "
                     "rather than being carried through it"),
    SegmentType("captive-pinion-mesh", "drive", "pinion / ring mesh", 0.026,
                "rollbar-silver",
                parameters=_p(pinion_teeth=18, ring_teeth=138, module_m=0.008,
                              pressure_angle_deg=20.0, drive_torque_nm=3800.0,
                              upper_ring_reference="same-body"),
                origin="engine-toy",
                note="one tooth mesh between a pinion and a ring. Author two of them "
                     "on the same pinion -- a ring under it and a ring over it -- and "
                     "the separating forces oppose, the load shares, and the shaft is "
                     "straddled instead of cantilevered; ground the upper ring to the "
                     "frame instead and the same hardware becomes a differential "
                     "planetary with an enormous flat reduction"),
    SegmentType("pinion-carrier-bearing", "joint", "pinion carrier bearing", 0.030,
                "rollbar-silver",
                parameters=_p(straddled=True, shaft_diameter_m=0.045),
                origin="engine-toy",
                note="what holds the pinion in mesh and takes the drive reaction; "
                     "straddled or cantilevered is the single parameter that decides "
                     "whether its shaft sees a quarter of the gap or the whole overhang"),
    SegmentType("commanded-rest-length-hoist-ram", "actuator", "hoist ram", 0.0625,
                "actuator-yellow",
                parameters=_p(commanded_rest_length_m=0.0, maximum_rest_length_m=0.850,
                              bore_m=0.125, rod_m=0.070, holding_force_n=180_000.0),
                origin="engine-toy",
                note="a hydraulic ram sized to lift an assembly rather than to trim it"),
)

BY_KEY = {t.key: t for t in SEGMENT_TYPES}
FAMILIES = ("structural", "joint", "spring", "actuator", "drive", "routed",
            "breakable", "impulse")


def of_family(family: str) -> tuple[SegmentType, ...]:
    return tuple(t for t in SEGMENT_TYPES if t.family == family)


def get(key: str) -> SegmentType:
    if key not in BY_KEY:
        raise KeyError(f"unknown segment type {key!r}")
    return BY_KEY[key]


# ---------------------------------------------------------------------
#  NODE KINDS -- what a point in the structure can BE
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class NodeType:
    key: str
    label: str
    default_mass_kg: float = 5.0
    fixed: bool = False
    note: str = ""


NODE_TYPES: tuple[NodeType, ...] = (
    NodeType("chassis-load-node", "load node", 20.0,
             note="an ordinary point in the structure that carries a wrench"),
    NodeType("roll-cage-node", "cage node", 12.0),
    NodeType("load-bearing-structure", "structure hub", 40.0),
    NodeType("structural-body-pin-frame-foot", "frame foot", 60.0, fixed=True,
             note="bolted to the world: the structure reacts against it"),
    NodeType("yaw-clutch-bearing", "yaw hub", 80.0),
    NodeType("pitch-bearing", "pitch trunnion", 40.0),
    NodeType("slew-ring-race-pad", "race pad", 30.0),
    NodeType("ring-gear", "ring gear", 120.0,
             note="a toothed ring. Laid over another one with a gap it makes the "
                  "pinion between them captive"),
    NodeType("drive-pinion", "pinion", 6.0),
    NodeType("pinion-carrier", "pinion carrier", 18.0,
             note="the body the drive unit bolts to, and what eats whatever "
                  "separating force a second ring did not cancel"),
    NodeType("recoiling-weapon-mass", "recoiling mass", 120.0,
             note="exempt from the usual connection count: it rides a bearing and a "
                  "recoil coupling, and that is all it should have"),
    NodeType("fluid-reservoir", "reservoir", 20.0),
    NodeType("high-pressure-canister", "pressure vessel", 30.0),
    NodeType("manifold", "manifold", 6.0),
    NodeType("electric-motor", "electric motor", 45.0),
    NodeType("gear-pump", "pump", 8.0),
    NodeType("spool-valve", "valve", 4.0),
    NodeType("fluid-cooler", "cooler", 6.0),
    NodeType("mass-and-volume-limited-magazine", "magazine", 50.0),
    NodeType("independent-fire-control-computer", "fire control", 9.0),
)
NODE_BY_KEY = {t.key: t for t in NODE_TYPES}
