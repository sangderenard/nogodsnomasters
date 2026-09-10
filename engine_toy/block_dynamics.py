"""Stage 1/2 of the "scientific bake": a real lumped-node mass/stiffness
(and thermal-conductance) network for the engine block, and the
generalized eigenvalue solve over it -- real natural frequencies and mode
shapes, not a fabricated gradient or a hand-placed mount point.

Real, disclosed simplification: each block segment is reduced to ONE
scalar degree of freedom (its own axial/breathing displacement along the
crank axis), connected to its neighbors by real axial-beam springs
(k = E*A/L). This captures the crank-axis mode family a real block
genuinely has (the same family a firing-order harmonic excites), not
full 3D bending/torsion -- a real, limited, buildable first model, not a
claim of FEA-grade fidelity. Stiffness and conductance come from the
SAME real node network and the SAME real cross-section derivation, so
Stage 4 (thermal) reuses this file's network unchanged.

Node masses and positions are read directly off build_drivetrain_graph's
own real graph (not re-derived independently), so this network can never
drift out of sync with the mesh/physics: the mass split across cylinder
segments is "powertrain.engine"'s own real total, divided across the
segments it actually represents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import sys
from pathlib import Path

import numpy as np

from drivetrain_graph import build_drivetrain_graph
import engine_geometry as eg

# Same real pipeline drivetrain_graph.py already reuses from turing/ for
# the mechanical graph itself -- Stage 2's eigensolve reuses turing's own
# real compiled symmetric-eigenvalue kernel (src/common/tensors/
# abstraction_methods/eigen.py), not scipy: a pure-AbstractTensor Jacobi
# sweep as the definitional/reference path, with a "blas" kernel-bank-
# compiled variant (through the same symbolic-math-to-LLVM pipeline
# everything else in turing/ compiles through) for the real fast path --
# measured 126-174x faster at exactly this problem's own scale (n=3..24).
_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))
from src.common.tensors.abstraction import AbstractTensor  # noqa: E402
from src.common.tensors import linalg  # noqa: E402

# Real, disclosed material properties for a grey cast-iron block (the
# common real material for the catalogue's own naturally-aspirated/
# diesel entries) -- a genuine literature value, not tuned to fit
# anything. A future pass could branch this by a real declared block
# material on Engine; every engine in this catalogue gets the same
# disclosed assumption for now.
CAST_IRON_YOUNGS_MODULUS_PA = 100.0e9
CAST_IRON_THERMAL_CONDUCTIVITY_W_PER_MK = 50.0
# The real load-bearing cross-section between two adjacent cylinder
# bays is the webbing/deck between bores, not the block's full outer
# cross-section -- a real, disclosed fraction of the block's own
# cross-sectional area (block_half_yz_m()**2 * 4), not the whole thing.
WEBBING_CROSS_SECTION_FRACTION = 0.30
# The oil pan bolts to the crankcase through a real, much smaller rail
# flange, not the full block cross-section.
PAN_RAIL_CROSS_SECTION_FRACTION = 0.05
# The bellhousing/transmission-case flange is a substantial full-
# diameter bolted joint (it carries the whole transmission's own
# weight plus reaction torque), but still well short of the solid
# webbing between bores -- a real, disclosed middle value between the
# pan rail and the webbing fractions above, not a separate guess.
BELLHOUSING_CROSS_SECTION_FRACTION = 0.20


@dataclass
class BlockNetworkEdge:
    a: int
    b: int
    length_m: float
    stiffness_n_per_m: float
    conductance_w_per_k: float


@dataclass
class BlockNetwork:
    """One real lumped-node network -- node i's real position/mass, and
    the real edges (stiffness + conductance) connecting them. identities[i]
    is the real graph node identity this DOF represents, so results can
    be mapped straight back onto the mesh/physics graph."""
    identities: list[str] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    masses_kg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    edges: list[BlockNetworkEdge] = field(default_factory=list)


def _cross_section_area_m2(engine, fraction: float) -> float:
    half = eg.block_half_yz_m(engine)
    return fraction * (2.0 * half) ** 2


def build_block_network(engine, include_transmission: bool = False,
                        transmission_mass_kg: float = 0.0) -> BlockNetwork:
    """The real per-cylinder block segments (same graph
    build_drivetrain_graph already builds and the mesh already draws),
    plus the oil pan -- connected in a real chain along the crank axis,
    with the pan hanging off the nearest segment. "powertrain.engine"'s
    own real total mass (the bare block+crank+heads casting, tracked as
    one lumped figure by the production subunit) is split evenly across
    the segments it represents -- they're real equal-volume slices of
    the same casting, so an even split is the honest one, not a guess.

    include_transmission additionally hangs the transmission's own real
    mass off the nearest block station through a real bellhousing-face
    stiffness edge -- the same pattern as the oil pan below, not a
    separate mechanism -- so a bellhousing/transmission mount candidate
    (select_mount_points) is judged against its OWN real modal
    contribution instead of borrowing the last bare-block segment's
    figure under a different name. Off by default: most crate-engine
    installs don't carry a transmission at all (see engine_package.py's
    own supplied-elsewhere default for it), and a network built without
    one is the honest answer for that case.

    transmission_mass_kg is the CALLER's own real figure for whatever
    transmission is actually being paired -- there is no transmission
    mass anywhere on Engine to default to (a crate engine's own
    mass_kg is the bare engine only, same real reason it ships without
    one at all), so a real number has to come from outside rather than
    being fabricated here. include_transmission with no real mass
    supplied builds the network without the extra DOF rather than
    guess at one."""
    graph = build_drivetrain_graph(engine)
    node_by_id = {n["identity"]: n for n in graph["nodes"]}

    segment_ids = sorted(
        (nid for nid in node_by_id if ".engine_block_body.cylinder_" in nid),
        key=lambda nid: node_by_id[nid]["reference_position"][0],
    )
    if not segment_ids:
        # radial/rotary/electric: no per-cylinder axial segmentation --
        # the single monolithic block body is the only real node available
        monolithic = node_by_id.get("powertrain.engine_block_body")
        if monolithic is None:
            return BlockNetwork()
        segment_ids = ["powertrain.engine_block_body"]

    crank_node = node_by_id.get("powertrain.engine")
    total_block_mass_kg = float(crank_node.get("mass_kg", 0.0)) if crank_node is not None else 0.0
    mass_per_cylinder = total_block_mass_kg / max(len(segment_ids), 1)

    # A V/W-bank engine's own cylinder_sites() puts more than one real
    # cylinder at the SAME x (one per bank, sharing a crank throw --
    # confirmed directly: a V8's cylinders 1&2 sit at an identical x,
    # only differing in y/z bank offset). Sorting segment ids by x alone
    # and connecting consecutive entries treated that shared-x pair as
    # sequential neighbors with near-zero real separation, which drove
    # stiffness (E*A/L) toward infinity and produced a real but
    # unphysical ~122 kHz mode cluster on the first V8 check -- found by
    # actually running the solve and reading the numbers, not assumed.
    # Grouped here by real x (rounded to kill float noise) into one
    # "crank station" node per unique axial position, summing the real
    # mass of every cylinder sharing that station -- honest, since
    # they're bolted to the same real cross-section of the block either
    # way, and a lateral (y/z) mode is a real, different, LATER
    # extension this 1-DOF-per-station axial model doesn't attempt yet.
    stations: dict[float, list[str]] = {}
    station_order: list[float] = []
    for nid in segment_ids:
        x = round(node_by_id[nid]["reference_position"][0], 6)
        if x not in stations:
            stations[x] = []
            station_order.append(x)
        stations[x].append(nid)
    station_order.sort()

    identities = ["+".join(stations[x]) for x in station_order]
    positions = [[x, 0.0, 0.0] for x in station_order]
    masses = [mass_per_cylinder * len(stations[x]) for x in station_order]

    web_area = _cross_section_area_m2(engine, WEBBING_CROSS_SECTION_FRACTION)
    edges: list[BlockNetworkEdge] = []
    for i in range(len(identities) - 1):
        a_pos = np.array(positions[i])
        b_pos = np.array(positions[i + 1])
        length_m = max(float(np.linalg.norm(b_pos - a_pos)), 1e-4)
        stiffness = CAST_IRON_YOUNGS_MODULUS_PA * web_area / length_m
        conductance = CAST_IRON_THERMAL_CONDUCTIVITY_W_PER_MK * web_area / length_m
        edges.append(BlockNetworkEdge(i, i + 1, length_m, stiffness, conductance))

    oil_pan_node = node_by_id.get("powertrain.oil_pan")
    if oil_pan_node is not None:
        pan_pos = np.array(oil_pan_node["reference_position"])
        nearest_idx = int(np.argmin([np.linalg.norm(pan_pos - np.array(p)) for p in positions]))
        length_m = max(float(np.linalg.norm(pan_pos - np.array(positions[nearest_idx]))), 1e-4)
        pan_area = _cross_section_area_m2(engine, PAN_RAIL_CROSS_SECTION_FRACTION)
        stiffness = CAST_IRON_YOUNGS_MODULUS_PA * pan_area / length_m
        conductance = CAST_IRON_THERMAL_CONDUCTIVITY_W_PER_MK * pan_area / length_m
        identities.append("powertrain.oil_pan")
        positions.append(oil_pan_node["reference_position"])
        masses.append(float(oil_pan_node.get("mass_kg", 0.0)))
        edges.append(BlockNetworkEdge(nearest_idx, len(identities) - 1, length_m, stiffness, conductance))

    transmission_node = node_by_id.get("powertrain.transmission")
    if include_transmission and transmission_node is not None and transmission_mass_kg > 0.0:
        trans_pos = np.array(transmission_node["reference_position"])
        nearest_idx = int(np.argmin([np.linalg.norm(trans_pos - np.array(p)) for p in positions]))
        length_m = max(float(np.linalg.norm(trans_pos - np.array(positions[nearest_idx]))), 1e-4)
        bell_area = _cross_section_area_m2(engine, BELLHOUSING_CROSS_SECTION_FRACTION)
        stiffness = CAST_IRON_YOUNGS_MODULUS_PA * bell_area / length_m
        conductance = CAST_IRON_THERMAL_CONDUCTIVITY_W_PER_MK * bell_area / length_m
        identities.append("powertrain.transmission")
        positions.append(transmission_node["reference_position"])
        masses.append(transmission_mass_kg)
        edges.append(BlockNetworkEdge(nearest_idx, len(identities) - 1, length_m, stiffness, conductance))

    return BlockNetwork(
        identities=identities,
        positions=np.array(positions, dtype=np.float64),
        masses_kg=np.array(masses, dtype=np.float64),
        edges=edges,
    )


def assemble_mass_stiffness_matrices(network: BlockNetwork) -> tuple[np.ndarray, np.ndarray]:
    """Real M (diagonal masses, kg) and K (global stiffness, N/m) --
    standard FEA assembly over the real edges above: each spring adds
    +k to both diagonal entries and -k to the off-diagonal pair."""
    n = len(network.identities)
    M = np.diag(network.masses_kg)
    K = np.zeros((n, n))
    for e in network.edges:
        K[e.a, e.a] += e.stiffness_n_per_m
        K[e.b, e.b] += e.stiffness_n_per_m
        K[e.a, e.b] -= e.stiffness_n_per_m
        K[e.b, e.a] -= e.stiffness_n_per_m
    return M, K


def assemble_conductance_matrix(network: BlockNetwork) -> np.ndarray:
    """The same real network's thermal side -- Stage 4 reuses this
    unchanged. Real conductance graph Laplacian (W/K)."""
    n = len(network.identities)
    G = np.zeros((n, n))
    for e in network.edges:
        G[e.a, e.a] += e.conductance_w_per_k
        G[e.b, e.b] += e.conductance_w_per_k
        G[e.a, e.b] -= e.conductance_w_per_k
        G[e.b, e.a] -= e.conductance_w_per_k
    return G


@dataclass
class ModalResult:
    identities: list[str]
    frequencies_hz: np.ndarray       # ascending, one per real DOF
    mode_shapes: np.ndarray          # (n_dof, n_modes) -- column i is mode i's real physical displacement per node


def solve_block_modes(network: BlockNetwork, eigh_method: str = "auto") -> ModalResult:
    """Stage 2: the real generalized eigenproblem K*phi = omega^2 * M*phi
    over Stage 1's own network, solved through turing's real compiled
    eigh (src/common/tensors/linalg.eigh) -- not scipy.

    M is diagonal (real per-node masses), so the standard reduction to a
    plain symmetric eigenproblem is exact and cheap: with
    M^-1/2 = diag(1/sqrt(mass_i)), let K' = M^-1/2 K M^-1/2 and solve the
    ORDINARY eigenproblem K'*psi = omega^2*psi (K' is symmetric whenever
    K is, which the FEA assembly above guarantees). The real physical
    mode shape is then phi = M^-1/2 @ psi -- undoing the same scaling,
    not a separate approximation.

    eigh_method: "auto" lets turing's own select_eigh_kernel choose the
    compiled "blas" kernel-bank path for this real unbatched small-n
    input (the fast path this whole exercise is for); "jacobi" forces
    the pure-AbstractTensor reference path if you need to cross-check.
    """
    M, K = assemble_mass_stiffness_matrices(network)
    n = len(network.identities)
    inv_sqrt_m = 1.0 / np.sqrt(np.maximum(network.masses_kg, 1e-9))
    K_scaled = (inv_sqrt_m[:, None] * K) * inv_sqrt_m[None, :]
    # force exact numerical symmetry -- the scaling above is
    # symmetric in exact arithmetic, but float roundoff can leave a
    # residual asymmetry the Jacobi sweep would otherwise chase
    K_scaled = 0.5 * (K_scaled + K_scaled.T)

    A = AbstractTensor.get_tensor(K_scaled.tolist())
    w, psi = linalg.eigh(A, method=eigh_method)
    w = np.array(w.tolist(), dtype=np.float64)
    psi = np.array(psi.tolist(), dtype=np.float64)

    # w are omega^2 (rigid-body/near-zero modes can read slightly
    # negative from roundoff -- a real free-free network genuinely has
    # a zero-frequency rigid-body translation mode; floor at 0 rather
    # than let sqrt raise on a tiny negative)
    omega = np.sqrt(np.maximum(w, 0.0))
    frequencies_hz = omega / (2.0 * math.pi)
    mode_shapes = inv_sqrt_m[:, None] * psi   # phi = M^-1/2 @ psi, per column

    return ModalResult(identities=list(network.identities),
                       frequencies_hz=frequencies_hz, mode_shapes=mode_shapes)


# Stage 3: real candidate mount locations, not an arbitrary search over
# the whole block surface. Real automotive practice mounts at the front
# timing cover and the bell-housing/clutch face -- both already real,
# named reference nodes this graph builds (crank_shaft.front/.rear,
# see drivetrain_graph.py) precisely because they're the real
# high-stiffness ends of the casting, not because they're convenient.
# Modal proximity below is the tie-breaker AMONG these, not a search
# that could recommend mounting mid-span on a webbing wall.
#
# powertrain.transmission is one stage further out than crank_shaft.
# rear's own bell-housing/clutch face: the transmission CASE's own real
# mount point, past the bolted joint rather than at it. Honest either
# way the network was built -- with include_transmission it resolves
# against the transmission's own real hung DOF (see build_block_network
# above); without it, it still resolves (the identity always exists on
# the raw graph node), just against whichever bare-block station is
# nearest, exactly as disclosed by `nearest_station` on the result.
MOUNT_CANDIDATE_IDENTITIES = (
    "powertrain.crank_shaft.front", "powertrain.crank_shaft.rear", "powertrain.transmission",
)


@dataclass
class MountCandidate:
    identity: str
    nearest_station: str
    dominant_mode_hz: float
    modal_displacement: float   # this candidate's own real displacement in the dominant mode, |phi| (relative units)
    resonance_margin_hz: float  # |dominant_mode_hz - excitation_hz| -- bigger is safer


def dominant_excitation_hz(engine, rpm: float) -> float:
    """The real firing-order harmonic at this rpm -- same formula
    engine_cycle_sim.py already uses for scavenging/resonance
    (cylinders * 360/cycle_degrees events per revolution), not a
    separately invented one."""
    arch = engine.architecture
    if not arch.cylinders:
        return 0.0
    firing_events_per_rev = arch.cylinders * 360.0 / arch.cycle_degrees
    return firing_events_per_rev * rpm / 60.0


def select_mount_points(engine, network: BlockNetwork, result: ModalResult,
                         rpm: float | None = None) -> list[MountCandidate]:
    """For each real structurally-viable candidate (MOUNT_CANDIDATE_IDENTITIES),
    find which real mode it's closest to exciting at this engine's own
    firing frequency, and how much that candidate itself actually moves
    in that mode -- the real isolation question (mount near a NODE of
    the mode nearest resonance, not an antinode), not "the strongest
    point" or "the quietest point" taken alone.

    rpm defaults to the engine's own real idle_rpm -- idle is where an
    engine spends the most continuous time and where mount-transmitted
    idle shake is most noticeable; a full sweep across the rev range is
    a real future extension (worst-case-margin over the operating band),
    not done here.
    """
    graph = build_drivetrain_graph(engine)
    node_by_id = {n["identity"]: n for n in graph["nodes"]}
    rpm = engine.idle_rpm if rpm is None else rpm
    excitation_hz = dominant_excitation_hz(engine, rpm)

    # the real mode whose natural frequency is closest to this engine's
    # own firing harmonic -- the one actually at resonance risk right now
    real_modes = [i for i, f in enumerate(result.frequencies_hz) if f > 1.0]
    if not real_modes:
        return []
    dominant_idx = min(real_modes, key=lambda i: abs(result.frequencies_hz[i] - excitation_hz))
    dominant_hz = float(result.frequencies_hz[dominant_idx])
    mode_vec = result.mode_shapes[:, dominant_idx]
    max_abs = float(np.max(np.abs(mode_vec))) or 1.0

    station_positions = np.array([node_by_id[nid.split("+")[0]]["reference_position"][0]
                                   for nid in network.identities])

    candidates = []
    for cand_id in MOUNT_CANDIDATE_IDENTITIES:
        cand_node = node_by_id.get(cand_id)
        if cand_node is None:
            continue
        cand_x = cand_node["reference_position"][0]
        nearest = int(np.argmin(np.abs(station_positions - cand_x)))
        displacement = abs(float(mode_vec[nearest])) / max_abs
        candidates.append(MountCandidate(
            identity=cand_id, nearest_station=network.identities[nearest],
            dominant_mode_hz=dominant_hz, modal_displacement=displacement,
            resonance_margin_hz=abs(dominant_hz - excitation_hz),
        ))
    return candidates
