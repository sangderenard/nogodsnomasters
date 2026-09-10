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

from dataclasses import dataclass
from enum import Enum

from block_dynamics import build_block_network, solve_block_modes, select_mount_points
from drivetrain_graph import build_drivetrain_graph
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


def assign_mounting(engine, install_context: str = "automotive", include_transmission: bool = False,
                    transmission_mass_kg: float = 0.0, rpm: float | None = None) -> list[MountAssignment]:
    """The real, automated policy: resolve every real mount point this
    engine's own graph declares to a technique + hardware grade.

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
    return out
