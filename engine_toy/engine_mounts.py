"""Stage 4 of the "scientific bake": real mounting HARDWARE and an
automated POLICY that assigns it, built directly on top of block_
dynamics.py's own real modal solve (Stage 1/2: lumped mass/stiffness
network + generalized eigensolve) and its Stage 3 mount-candidate
search (select_mount_points) -- not a separate, disconnected mount
system with its own invented positions.

There are real, genuinely different engine-mounting TECHNIQUES in
actual practice -- a rubber (or hydraulic) isolator biscuit, a rigid
solid-bolted joint, a cradle/subframe carrying the engine through
stiffer bushings, a welded-in bar-cage attachment (the crate-engine
prism's own real load path, see engine_package.py/ENGINE_TOY_
ARCHITECTURE_NOTES.md's "the prism is a bar cage" section), and a
secondary torque strap/dogbone that only ever SUPPLEMENTS a compliant
primary mount, never replaces one. Which one is right is not a single
universal answer -- it depends on the real INSTALL CONTEXT (a road
engine, an aircraft firewall mount, a marine application, a rigid
industrial skid, or a supplied bar cage), same as a real mechanic or
airframe engineer would choose by context, not by picking the
"strongest" option regardless of application.

The real automated part: once an install context is declared,
`assign_mounting` resolves EVERY real mount point this engine's own
graph already declares (mount.engine_left/right, and -- only when the
transmission is actually being baked into this same crate -- mount.
transmission_left/right and mount.transfer_case_left/right) to a real
technique and hardware grade, using block_dynamics.py's own modal
result to decide which points are riding close to a resonance
antinode and need the firmer isolator grade (or a torque strap) rather
than guessing site-by-site. The install context still drives the base
technique -- a road engine does not switch to solid mounts just
because one bolt reads a bad number, the same way a real installer
wouldn't -- the modal data only refines the hardware GRADE within
that already-chosen family.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from block_dynamics import build_block_network, solve_block_modes, select_mount_points, dominant_excitation_hz
from drivetrain_graph import build_drivetrain_graph
from engine_mass_properties import rigid_body_properties, RigidBodyProperties
import engine_geometry


class MountingTechnique(Enum):
    RUBBER_ISOLATOR = "rubber-isolator"    # compliant biscuit/puck -- the common road-vehicle default
    CRADLE_SUBFRAME = "cradle-subframe"    # stiffer bushings through a real K-member/cradle, heavier engines
    SOLID_BOLTED = "solid-bolted"          # rigid, no isolation -- industrial skids, aircraft, race solid-mounts
    BAR_CAGE = "bar-cage"                  # welded/bolted directly into a supplied structural cage
    TORQUE_STRAP = "torque-strap"          # secondary reaction anchor; only ever supplements another technique


# Real, disclosed order-of-magnitude hardware numbers -- approximate
# published ranges for each real technique, not tuned to fit anything.
# `grade` distinguishes the modal-driven refinement within a technique
# (a firmer rubber compound at a point riding close to resonance)
# without needing a whole separate enum member for it.
@dataclass(frozen=True)
class MountHardware:
    technique: MountingTechnique
    grade: str                       # "standard" | "firm"
    stiffness_n_per_m: float | None  # None == effectively rigid (solid/bar-cage)
    max_static_load_n: float
    description: str


_HARDWARE: dict[tuple[MountingTechnique, str], MountHardware] = {
    (MountingTechnique.RUBBER_ISOLATOR, "standard"): MountHardware(
        MountingTechnique.RUBBER_ISOLATOR, "standard", 1.2e6, 15_000.0,
        "molded rubber biscuit mount, standard road-car durometer"),
    (MountingTechnique.RUBBER_ISOLATOR, "firm"): MountHardware(
        MountingTechnique.RUBBER_ISOLATOR, "firm", 2.6e6, 15_000.0,
        "firmer-durometer rubber biscuit -- riding close to a real "
        "resonance at idle, a softer compound would let this point "
        "shake more, not less"),
    (MountingTechnique.CRADLE_SUBFRAME, "standard"): MountHardware(
        MountingTechnique.CRADLE_SUBFRAME, "standard", 4.0e6, 30_000.0,
        "cradle/subframe bushing, heavier-engine duty"),
    (MountingTechnique.CRADLE_SUBFRAME, "firm"): MountHardware(
        MountingTechnique.CRADLE_SUBFRAME, "firm", 7.0e6, 30_000.0,
        "firmer cradle bushing at a real resonance-risk point"),
    (MountingTechnique.SOLID_BOLTED, "standard"): MountHardware(
        MountingTechnique.SOLID_BOLTED, "standard", None, 200_000.0,
        "rigid bolted joint, no isolation -- bolt shear is the real limit"),
    (MountingTechnique.BAR_CAGE, "standard"): MountHardware(
        MountingTechnique.BAR_CAGE, "standard", None, 200_000.0,
        "welded/bolted directly to a supplied structural cage member"),
    (MountingTechnique.TORQUE_STRAP, "standard"): MountHardware(
        MountingTechnique.TORQUE_STRAP, "standard", 3.0e6, 8_000.0,
        "reaction-only dogbone/strap bushing -- firmer than the "
        "primary mount by design, and never sized to carry the "
        "engine's own static weight alone"),
}

# Real, closed set of install contexts this policy actually resolves --
# not free text, so a typo can't silently fall through to a wrong
# default. Add a new context here (with real reasoning) rather than
# guessing at one ad hoc in select_technique.
INSTALL_CONTEXTS = ("automotive", "marine", "aircraft", "industrial_stationary", "bar_cage")

# A resonance-risk heuristic, disclosed rather than hidden in the
# comparison below: a candidate is "at risk" when the block's own
# dominant mode at idle sits within this margin of its real firing
# harmonic AND that candidate's own modal displacement (block_dynamics.
# select_mount_points' normalized |phi|) is more than half the mode's
# own peak -- i.e. it's genuinely riding toward an antinode of a mode
# that's genuinely close to being excited, not just "some number came
# back nonzero."
RESONANCE_RISK_MARGIN_HZ = 60.0
RESONANCE_RISK_DISPLACEMENT = 0.5
# Above this mass a road-going engine is real cradle/subframe territory
# (a K-member through stiffer bushings) rather than simple bolt-on
# rubber biscuits -- the same real reason heavy industrial diesels
# (e.g. the catalogue's Cat C18) sit in a cradle, not on rubber pucks.
CRADLE_MASS_THRESHOLD_KG = 500.0
# A strap only ever helps a COMPLIANT primary mount (it exists to
# resist the extra roll a soft mount allows under a torque spike); a
# rigid technique already resists that by construction, so a strap
# adds nothing there and is never recommended for one.
TORQUE_STRAP_SPECIFIC_TORQUE_NM_PER_KG = 1.1


def select_technique(engine, install_context: str = "automotive") -> MountingTechnique:
    """The context-driven base choice -- what a real installer picks
    FIRST, before any per-point vibration reading. Modal data (assign_
    mounting below) only ever refines the hardware GRADE within this
    family; it never flips a road engine onto solid mounts because one
    bolt reads a bad number, the same way a real installer wouldn't."""
    if install_context not in INSTALL_CONTEXTS:
        raise ValueError(f"unknown install_context {install_context!r}; expected one of {INSTALL_CONTEXTS}")
    if install_context == "bar_cage":
        return MountingTechnique.BAR_CAGE
    if install_context == "industrial_stationary":
        return MountingTechnique.SOLID_BOLTED
    if install_context == "aircraft":
        # a real aircraft engine mount is a rigid welded-tube truss/ring
        # (a "dynafocal" or bed mount) bolted straight to the firewall/
        # nacelle structure -- not rubber-isolated the way a car is
        return MountingTechnique.SOLID_BOLTED
    if engine.kind in ("atmospheric", "turbine"):
        # a real stationary gravity-piston engine or a skid-mounted gas
        # turbine bolts rigidly to its own bedplate/foundation
        # regardless of the declared install context
        return MountingTechnique.SOLID_BOLTED
    if engine.mass_kg >= CRADLE_MASS_THRESHOLD_KG:
        return MountingTechnique.CRADLE_SUBFRAME
    # marine diesels commonly ride flexible mounts to isolate hull-
    # borne noise/vibration, same real family as a road car
    return MountingTechnique.RUBBER_ISOLATOR


def wants_torque_strap(engine, technique: MountingTechnique) -> bool:
    if technique not in (MountingTechnique.RUBBER_ISOLATOR, MountingTechnique.CRADLE_SUBFRAME):
        return False
    specific_torque = engine.peak_torque_nm / max(engine.mass_kg, 1.0)
    boosted = engine.forced_induction.kind != "none" or engine.has_nitrous
    return specific_torque >= TORQUE_STRAP_SPECIFIC_TORQUE_NM_PER_KG or boosted


# ---------------------------------------------------------------------
# Universal stability gate: real, engine-agnostic, and cheap -- the
# mount points, projected onto the horizontal (X-Z) plane, must form a
# real polygon (at least 3 non-collinear points -- 2 points are a real
# line, which can never resist roll/tip-over no matter how stiff) that
# CONTAINS the assembly's own real center of gravity. This is the one
# check every real mount layout satisfies regardless of technique or
# application (a bolted industrial skid needs it exactly as much as a
# rubber-isolated road engine) -- not a bespoke per-engine heuristic.
# ---------------------------------------------------------------------

def _cross2(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Standard Andrew's monotone chain -- plain, small-n, no external
    dependency needed for the handful of real mount points this ever
    runs on."""
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 2:
        return np.array(pts)
    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross2(np.array(lower[-2]), np.array(lower[-1]), np.array(p)) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross2(np.array(upper[-2]), np.array(upper[-1]), np.array(p)) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _point_in_convex_polygon(pt: tuple[float, float], hull: np.ndarray) -> bool:
    n = len(hull)
    if n < 3:
        return False   # a line or a single point can't contain anything
    sign = None
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if abs(cross) < 1e-9:
            continue   # on the edge line -- inclusive, not a failure
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


@dataclass(frozen=True)
class StabilityResult:
    ok: bool
    reason: str
    point_count: int
    cg_xz: tuple[float, float]


def check_stability(positions: list[tuple[float, float, float]],
                    cg: tuple[float, float, float]) -> StabilityResult:
    cg_xz = (cg[0], cg[2])
    if len(positions) < 3:
        return StabilityResult(False, f"only {len(positions)} real support point(s) -- at least 3 "
                               "non-collinear points are needed to resist roll/tip-over", len(positions), cg_xz)
    pts_xz = np.array([[p[0], p[2]] for p in positions])
    hull = _convex_hull_2d(pts_xz)
    if len(hull) < 3:
        return StabilityResult(False, "real support points are collinear -- no real polygon to "
                               "carry the load", len(positions), cg_xz)
    if not _point_in_convex_polygon(cg_xz, hull):
        return StabilityResult(False, "center of gravity falls outside the support polygon -- "
                               "real tip-over risk", len(positions), cg_xz)
    return StabilityResult(True, "stable", len(positions), cg_xz)


# ---------------------------------------------------------------------
# Optional deeper check: Torque Roll Axis (TRA) rigid-body decoupling
# -- the real, standard automotive NVH method, not a bespoke addition.
# Only meaningful for a COMPLIANT technique (rubber isolator/cradle):
# a rigid mount has no compliance to decouple, and real industrial/
# aircraft installs never run this analysis in practice (see the
# design discussion this module's docstring reflects). Deliberately
# plain numpy, not turing's compiled eigh -- that path earned its
# keep on block_dynamics.py's own larger per-cylinder problem (n=3..24,
# "measured 126-174x faster"); this is a fixed, tiny n=6 rigid-body
# problem where reaching for the same machinery would be needless
# machinery for machinery's sake.
# ---------------------------------------------------------------------

def _skew(r: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


@dataclass(frozen=True)
class TRAResult:
    frequencies_hz: np.ndarray       # 6 real rigid-body mode frequencies, ascending
    dominant_hz: float               # the mode nearest this engine's own firing harmonic at idle
    resonance_margin_hz: float       # |dominant_hz - firing harmonic| -- bigger is safer


def evaluate_tra(engine, rigid_body: RigidBodyProperties,
                 mounts: list["MountAssignment"], rpm: float | None = None) -> TRAResult | None:
    """Builds the real 6x6 rigid-body mass/stiffness system (3
    translation + 3 rotation about the assembly's own real center of
    gravity) from each mount's own real position and (isotropic,
    disclosed-simplified -- MountHardware carries one scalar stiffness,
    not a real direction-dependent rate) stiffness, and solves the
    generalized eigenproblem for the 6 real rigid-body mode
    frequencies. Returns None when there's nothing real to decouple
    (fewer than 3 points, or every mount in the set is rigid)."""
    positions = [m.position for m in mounts]
    stiffnesses = [m.hardware.stiffness_n_per_m for m in mounts]
    if len(positions) < 3 or rigid_body.total_mass_kg <= 0.0:
        return None
    cg = np.array(rigid_body.center_of_gravity)
    K = np.zeros((6, 6))
    for pos, k in zip(positions, stiffnesses):
        if k is None:
            continue   # a rigid mount contributes no compliance to decouple
        r = np.array(pos) - cg
        A = np.zeros((3, 6))
        A[:, :3] = np.eye(3)
        A[:, 3:] = -_skew(r)
        K += k * (A.T @ A)
    if not np.any(K):
        return None

    M = np.zeros((6, 6))
    M[:3, :3] = rigid_body.total_mass_kg * np.eye(3)
    M[3:, 3:] = rigid_body.inertia_tensor_kg_m2

    w_rot, v_rot = np.linalg.eigh(rigid_body.inertia_tensor_kg_m2)
    inv_sqrt_rot = v_rot @ np.diag(1.0 / np.sqrt(np.maximum(w_rot, 1e-9))) @ v_rot.T
    M_inv_sqrt = np.zeros((6, 6))
    M_inv_sqrt[:3, :3] = np.eye(3) / math.sqrt(rigid_body.total_mass_kg)
    M_inv_sqrt[3:, 3:] = inv_sqrt_rot

    K_scaled = M_inv_sqrt @ K @ M_inv_sqrt
    K_scaled = 0.5 * (K_scaled + K_scaled.T)
    w2 = np.linalg.eigvalsh(K_scaled)
    freqs = np.sort(np.sqrt(np.maximum(w2, 0.0)) / (2.0 * math.pi))

    rpm_ = engine.idle_rpm if rpm is None else rpm
    excitation_hz = dominant_excitation_hz(engine, rpm_)
    real_freqs = freqs[freqs > 1.0]
    if len(real_freqs) == 0:
        return TRAResult(frequencies_hz=freqs, dominant_hz=0.0, resonance_margin_hz=float("inf"))
    idx = int(np.argmin(np.abs(real_freqs - excitation_hz)))
    return TRAResult(frequencies_hz=freqs, dominant_hz=float(real_freqs[idx]),
                     resonance_margin_hz=float(abs(real_freqs[idx] - excitation_hz)))


# ---------------------------------------------------------------------
# Universal subframe fallback -- for whatever real block-mount geometry
# genuinely can't achieve on its own (too short/wide, an asymmetric
# mass distribution, a single-cylinder unit with no real axial spread
# at all). A real, wide rectangular subframe sized directly off the
# assembly's own real mass-bearing extent GUARANTEES CG containment BY
# CONSTRUCTION (every corner spans at least as far as every real
# mass-bearing component that went into computing the same CG), not
# something checked afterward and hoped for.
# ---------------------------------------------------------------------

def _subframe_points(engine, graph: dict) -> list[tuple[str, tuple[float, float, float]]]:
    positions = [n["reference_position"] for n in graph["nodes"] if n.get("mass_in_total")]
    if not positions:
        positions = [[0.0, 0.0, 0.0]]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [abs(p[2]) for p in positions]
    x_min, x_max = min(xs), max(xs)
    x_margin = max(0.05, (x_max - x_min) * 0.1)
    half_z = max(zs) * 1.15 + 0.02
    y = min(ys) - engine_geometry.block_half_yz_m(engine)
    return [
        ("mount.subframe_front_left", (x_min - x_margin, y, -half_z)),
        ("mount.subframe_front_right", (x_min - x_margin, y, half_z)),
        ("mount.subframe_rear_left", (x_max + x_margin, y, -half_z)),
        ("mount.subframe_rear_right", (x_max + x_margin, y, half_z)),
    ]


@dataclass(frozen=True)
class MountAssignment:
    identity: str                          # the real graph node identity (or a synthetic strap name)
    role: str                              # "engine" | "transmission" | "transfer_case" | "torque_strap"
    position: tuple[float, float, float]
    hardware: MountHardware
    modal_displacement: float | None = None   # None where no modal candidate could be correlated
    resonance_margin_hz: float | None = None


_ROLE_BY_PREFIX = (
    ("mount.engine_", "engine"),
    ("mount.transmission_", "transmission"),
    ("mount.transfer_case_", "transfer_case"),
)


def _role_for(identity: str) -> str | None:
    for prefix, role in _ROLE_BY_PREFIX:
        if identity.startswith(prefix):
            return role
    return None


@dataclass(frozen=True)
class MountingPlan:
    mounts: list[MountAssignment]
    technique: MountingTechnique
    stability: StabilityResult
    rigid_body: RigidBodyProperties
    tra: TRAResult | None = None
    subframe_used: bool = False


def assign_mounting(engine, install_context: str = "automotive", include_transmission: bool = False,
                    transmission_mass_kg: float = 0.0, rpm: float | None = None) -> MountingPlan:
    """The real, automated policy: resolve every real mount point this
    engine's own graph declares to a technique + hardware grade, then
    run it through the universal CG-in-support-polygon stability gate
    (falling back to a real subframe when the direct points can't pass
    it) and, for a compliant technique, the optional deeper Torque
    Roll Axis rigid-body decoupling check.

    include_transmission gates whether mount.transmission_*/mount.
    transfer_case_* are assigned at all -- most crate-engine installs
    don't carry a transmission (see engine_package.py's own supplied-
    elsewhere default), so by default only the engine's own primary
    mounts (mount.engine_left/right) are returned. Passing True (a
    player choosing to bake more into one crate) also assigns the
    driveline's own real mount points, correlated against block_
    dynamics's own modal candidate for the bellhousing/transmission
    region when a real transmission_mass_kg is supplied (see block_
    dynamics.build_block_network's own docstring on why that number
    can't be fabricated here)."""
    technique = select_technique(engine, install_context)
    graph = build_drivetrain_graph(engine)
    node_by_id = {n["identity"]: n for n in graph["nodes"]}

    network = build_block_network(engine, include_transmission=include_transmission,
                                  transmission_mass_kg=transmission_mass_kg)
    modal_by_identity: dict[str, tuple[float, float]] = {}
    if network.identities:
        result = solve_block_modes(network)
        for cand in select_mount_points(engine, network, result, rpm=rpm):
            modal_by_identity[cand.identity] = (cand.modal_displacement, cand.resonance_margin_hz)

    # crank_shaft.front feeds mount.engine_*'s own risk reading (the
    # front timing-cover end); crank_shaft.rear/transmission feed the
    # driveline end -- both real MOUNT_CANDIDATE_IDENTITIES stations,
    # not a new lookup.
    front_risk = modal_by_identity.get("powertrain.crank_shaft.front")
    rear_risk = modal_by_identity.get("powertrain.crank_shaft.rear")
    trans_risk = modal_by_identity.get("powertrain.transmission", rear_risk)

    def _grade(risk: tuple[float, float] | None) -> str:
        if risk is None:
            return "standard"
        displacement, margin = risk
        at_risk = margin < RESONANCE_RISK_MARGIN_HZ and displacement > RESONANCE_RISK_DISPLACEMENT
        return "firm" if at_risk else "standard"

    def _hardware_for(technique_: MountingTechnique, risk: tuple[float, float] | None) -> MountHardware:
        grade = _grade(risk) if technique_ in (MountingTechnique.RUBBER_ISOLATOR, MountingTechnique.CRADLE_SUBFRAME) else "standard"
        return _HARDWARE[(technique_, grade)]

    out: list[MountAssignment] = []
    for identity, node in node_by_id.items():
        role = _role_for(identity)
        if role is None:
            continue
        if role in ("transmission", "transfer_case") and not include_transmission:
            continue
        risk = front_risk if role == "engine" else trans_risk
        hw = _hardware_for(technique, risk)
        out.append(MountAssignment(
            identity=identity, role=role, position=tuple(node["reference_position"]),
            hardware=hw, modal_displacement=(risk[0] if risk else None),
            resonance_margin_hz=(risk[1] if risk else None)))

    rigid_body = rigid_body_properties(engine, graph)

    # The universal gate: does this technique's own real support points
    # (a torque strap never counts -- it isn't rated to carry static
    # weight, see its own hardware description) actually contain the
    # assembly's real center of gravity? Skipped for BAR_CAGE: a
    # supplied structure's own stability is the vehicle/game's real
    # responsibility, not this toy's to second-guess (same reasoning
    # as correlate_mounts never inventing cage geometry of its own).
    support_positions = [m.position for m in out if m.role in ("engine", "transmission", "transfer_case")]
    stability = check_stability(support_positions, rigid_body.center_of_gravity)
    subframe_used = False

    if technique is not MountingTechnique.BAR_CAGE and not stability.ok:
        # A genuinely "strange" build -- too few/too collinear real
        # mount points to ever contain the CG on their own (a very
        # short engine, an asymmetric accessory load, a single-
        # cylinder unit). A subframe sidesteps the whole question:
        # sized directly off the real mass-bearing extent, so
        # containment is guaranteed BY CONSTRUCTION, not hoped for
        # after the fact -- see _subframe_points' own docstring.
        technique = MountingTechnique.CRADLE_SUBFRAME
        subframe_hw = _HARDWARE[(MountingTechnique.CRADLE_SUBFRAME, "standard")]
        out = [MountAssignment(identity=name, role="engine", position=pos, hardware=subframe_hw)
              for name, pos in _subframe_points(engine, graph)]
        support_positions = [m.position for m in out]
        stability = check_stability(support_positions, rigid_body.center_of_gravity)
        subframe_used = True

    if wants_torque_strap(engine, technique):
        strap_hw = _HARDWARE[(MountingTechnique.TORQUE_STRAP, "standard")]
        mounts = engine_geometry.mount_points(engine)
        front = mounts["front_mount"]
        top_y = engine_geometry.block_half_yz_m(engine) * 2.0
        out.append(MountAssignment(
            identity="mount.torque_strap", role="torque_strap",
            position=(front[0], top_y, 0.0), hardware=strap_hw,
            modal_displacement=(front_risk[0] if front_risk else None),
            resonance_margin_hz=(front_risk[1] if front_risk else None)))

    out.sort(key=lambda m: m.identity)

    tra = None
    if technique in (MountingTechnique.RUBBER_ISOLATOR, MountingTechnique.CRADLE_SUBFRAME):
        structural = [m for m in out if m.role != "torque_strap"]
        tra = evaluate_tra(engine, rigid_body, structural, rpm=rpm)

    return MountingPlan(mounts=out, technique=technique, stability=stability,
                        rigid_body=rigid_body, tra=tra, subframe_used=subframe_used)
