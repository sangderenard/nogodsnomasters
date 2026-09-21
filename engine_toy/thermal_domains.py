"""Geometry-backed thermal domains for assembled engine-toy objects.

Thin hardware uses the repository's cotangent surface Laplacian.  Filled
hardware and fluids use the metric-aware three-dimensional Laplace builder.
The resulting domains are immutable geometry; temperatures and heat sources
remain runtime state owned by the machine or coupled simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np


_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.common.tensors.abstract_convolution import laplace_nd  # noqa: E402
from src.common.tensors.abstract_convolution.laplace_nd import (  # noqa: E402
    BuildGraphLaplace,
)
from src.common.tensors.abstraction import AbstractTensor  # noqa: E402
from src.common.dt_system.dt_scaler import Metrics  # noqa: E402
from src.common.dt_system.engine_api import DtCompatibleEngine  # noqa: E402
from src.common.dt_system.error_channels import (  # noqa: E402
    DT_CHANNEL_NAMES, empty_channels,
)
from src.common.dt_system.time_contracts import BIND, HOLD  # noqa: E402
from src.common.tensors.riemann.mesh_laplace import (          # noqa: E402
    CotangentMeshGeometry,
    build_cotangent_geometry,
)
from vehicle_mesh import SolidPart, build_drivetrain_solid_parts  # noqa: E402


def _part_name(identity: str) -> str:
    return "node_" + identity.replace("/", "_").replace(".", "_")


def weld_triangle_soup(vertices, *, decimals: int = 10):
    """Weld coincident triangle-soup corners into one surface topology."""
    source = np.asarray(vertices, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) % 3:
        raise ValueError("triangle soup must have shape (3*T, 3)")
    key = np.round(source, decimals=decimals)
    unique, inverse = np.unique(key, axis=0, return_inverse=True)
    triangles = inverse.reshape(-1, 3)
    nondegenerate = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 2] != triangles[:, 0])
    )
    return unique, triangles[nondegenerate]


@dataclass(frozen=True)
class ShellThermalDomain:
    identity: str
    group: str | None
    material: str | None
    vertices: np.ndarray
    triangles: np.ndarray
    geometry: CotangentMeshGeometry
    thickness_m: float

    def laplace(self, temperature_k) -> np.ndarray:
        return self.geometry.apply(np.asarray(temperature_k, dtype=np.float64))


@dataclass(frozen=True)
class VolumeThermalDomain:
    identity: str
    group: str | None
    material: str | None
    shape: tuple[int, int, int]
    half_extent_m: tuple[float, float, float]
    laplacian: np.ndarray

    def laplace(self, temperature_k) -> np.ndarray:
        values = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        return (self.laplacian @ values).reshape(self.shape)


@dataclass(frozen=True)
class PathThermalDomain:
    """A sampled physical passage whose intrinsic coordinate follows a path.

    The path may be straight, bent, or coiled in world space.  Its Laplacian
    follows the declared connectivity, while its world positions are retained
    so interfaces can discover which samples actually face one another.
    """

    identity: str
    group: str | None
    material: str | None
    positions: np.ndarray
    nodal_lengths_m: np.ndarray
    cross_section_area_m2: float
    laplacian: np.ndarray

    def laplace(self, temperature_k) -> np.ndarray:
        values = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
        return self.laplacian @ values


@dataclass(frozen=True)
class ThermalMaterial:
    density_kg_m3: float
    specific_heat_j_kg_k: float
    conductivity_w_m_k: float

    @property
    def diffusivity_m2_s(self) -> float:
        return (self.conductivity_w_m_k
                / (self.density_kg_m3 * self.specific_heat_j_kg_k))


# Physical properties used by the dewar graph's declared construction.
# These are ordinary room-temperature handbook values.  A component may
# override any value directly on its graph node when temperature dependence
# or an effective composite property is available.
THERMAL_MATERIALS = {
    "stainless-steel": ThermalMaterial(8000.0, 500.0, 16.0),
    "steel-plate": ThermalMaterial(7850.0, 490.0, 50.0),
    "aluminium-casting": ThermalMaterial(2700.0, 900.0, 150.0),
    "copper": ThermalMaterial(8960.0, 385.0, 401.0),
    "brass": ThermalMaterial(8500.0, 380.0, 110.0),
    "polyethylene": ThermalMaterial(950.0, 1900.0, 0.40),
    # Effective pack-scale properties: cells, busbars, separators, and case.
    "battery-module": ThermalMaterial(1800.0, 1000.0, 1.5),
}


def material_for(node: dict) -> ThermalMaterial:
    base = THERMAL_MATERIALS.get(node.get("material"))
    fields = (
        node.get("thermal_density_kg_m3"),
        node.get("thermal_specific_heat_j_kg_k"),
        node.get("thermal_conductivity_w_m_k"),
    )
    if base is None and any(value is None for value in fields):
        raise ValueError(
            f"{node['identity']} has no thermal properties for {node.get('material')!r}")
    return ThermalMaterial(
        float(fields[0] if fields[0] is not None else base.density_kg_m3),
        float(fields[1] if fields[1] is not None else base.specific_heat_j_kg_k),
        float(fields[2] if fields[2] is not None else base.conductivity_w_m_k),
    )


@dataclass
class ThermalDomainState:
    domain: ShellThermalDomain | VolumeThermalDomain
    material: ThermalMaterial
    temperature_k: np.ndarray
    capacity_j_k: np.ndarray

    @property
    def total_capacity_j_k(self) -> float:
        return float(self.capacity_j_k.sum())

    @property
    def mean_temperature_k(self) -> float:
        return float(np.sum(self.temperature_k * self.capacity_j_k)
                     / self.total_capacity_j_k)

    @property
    def energy_j(self) -> float:
        return float(np.sum(self.temperature_k * self.capacity_j_k))


@dataclass(frozen=True)
class ThermalInterface:
    a: str
    b: str
    conductance_w_k: float


@dataclass(frozen=True)
class DistributedThermalInterface:
    """Geometry-matched conductance between samples in two thermal paths."""

    a: str
    b: str
    conductance_w_k: np.ndarray


class ThermalAssembly:
    """Temperatures over cached Laplace domains and conservative interfaces."""

    def __init__(self, states: dict[str, ThermalDomainState],
                 interfaces=()):
        self.states = dict(states)
        self.interfaces = list(interfaces)
        self.elapsed_s = 0.0

    @property
    def energy_j(self) -> float:
        return sum(state.energy_j for state in self.states.values())

    def temperature_k(self, identity: str) -> float:
        return self.states[identity].mean_temperature_k

    def _stable_dt(self) -> float:
        limits = []
        for state in self.states.values():
            alpha = state.material.diffusivity_m2_s
            domain = state.domain
            if isinstance(domain, (VolumeThermalDomain, PathThermalDomain)):
                rate = alpha * float(np.max(np.diag(domain.laplacian)))
            else:
                g = domain.geometry
                incident = np.zeros(len(domain.vertices), dtype=np.float64)
                np.add.at(incident, g.edges[:, 0], np.abs(g.cotangent_weights))
                np.add.at(incident, g.edges[:, 1], np.abs(g.cotangent_weights))
                rate = alpha * float(np.max(
                    incident / np.maximum(g.lumped_vertex_areas, 1e-30)))
            if rate > 0.0:
                limits.append(0.45 / rate)
        for interface in self.interfaces:
            if isinstance(interface, DistributedThermalInterface):
                conductance = np.asarray(interface.conductance_w_k, dtype=np.float64)
                a = self.states[interface.a]
                b = self.states[interface.b]
                a_rate = np.max(conductance.sum(axis=1)
                                / np.asarray(a.capacity_j_k).reshape(-1))
                b_rate = np.max(conductance.sum(axis=0)
                                / np.asarray(b.capacity_j_k).reshape(-1))
                rate = max(float(a_rate), float(b_rate))
                if rate > 0.0:
                    limits.append(0.45 / rate)
        return min(limits) if limits else math.inf

    def step(self, dt_s: float, *, source_w: dict[str, float] | None = None):
        dt = max(0.0, float(dt_s))
        if dt == 0.0:
            return self.snapshot()
        stable = self._stable_dt()
        substeps = max(1, int(math.ceil(dt / stable))) if math.isfinite(stable) else 1
        h = dt / substeps
        sources = source_w or {}
        for _ in range(substeps):
            # Each chart advances with its own repository Laplace operator.
            for state in self.states.values():
                alpha = state.material.diffusivity_m2_s
                if isinstance(state.domain, (VolumeThermalDomain, PathThermalDomain)):
                    delta = -alpha * state.domain.laplace(state.temperature_k)
                else:
                    delta = alpha * state.domain.laplace(state.temperature_k)
                state.temperature_k = state.temperature_k + h * delta

            for identity, watts in sources.items():
                state = self.states[identity]
                state.temperature_k += (
                    float(watts) * h / state.total_capacity_j_k)

            for interface in self.interfaces:
                a = self.states[interface.a]
                b = self.states[interface.b]
                if isinstance(interface, DistributedThermalInterface):
                    conductance = np.asarray(
                        interface.conductance_w_k, dtype=np.float64)
                    ta = np.asarray(a.temperature_k).reshape(-1)
                    tb = np.asarray(b.temperature_k).reshape(-1)
                    power = conductance * (ta[:, None] - tb[None, :])
                    a.temperature_k = (
                        ta - power.sum(axis=1) * h
                        / np.asarray(a.capacity_j_k).reshape(-1))
                    b.temperature_k = (
                        tb + power.sum(axis=0) * h
                        / np.asarray(b.capacity_j_k).reshape(-1))
                    continue
                delta_k = a.mean_temperature_k - b.mean_temperature_k
                energy = interface.conductance_w_k * delta_k * h
                # Do not step past the two-body equilibrium temperature.
                equilibrium_energy = delta_k / (
                    1.0 / a.total_capacity_j_k + 1.0 / b.total_capacity_j_k)
                energy = math.copysign(
                    min(abs(energy), abs(equilibrium_energy)), energy)
                a.temperature_k -= energy / a.total_capacity_j_k
                b.temperature_k += energy / b.total_capacity_j_k
        self.elapsed_s += dt
        return self.snapshot()

    def snapshot(self):
        return {
            "elapsed_s": self.elapsed_s,
            "energy_j": self.energy_j,
            "temperature_k": {
                identity: state.mean_temperature_k
                for identity, state in self.states.items()
            },
        }

    def copy_shallow(self):
        """Capture every mutable thermal column for dt-system rollback."""
        return (
            float(self.elapsed_s),
            {identity: state.temperature_k.copy()
             for identity, state in self.states.items()},
        )

    def restore(self, snapshot) -> None:
        elapsed_s, temperatures = snapshot
        self.elapsed_s = float(elapsed_s)
        if set(temperatures) != set(self.states):
            raise ValueError("thermal snapshot does not match this assembly")
        for identity, values in temperatures.items():
            state = self.states[identity]
            restored = np.asarray(values, dtype=np.float64)
            if restored.shape != state.temperature_k.shape:
                raise ValueError(
                    f"thermal snapshot shape changed for {identity}: "
                    f"{restored.shape} != {state.temperature_k.shape}")
            state.temperature_k[...] = restored


class ThermalSystem(DtCompatibleEngine):
    """One top-level thermal simulation over one or more assemblies.

    Geometry domains, interfaces, sources, and internal Laplace work remain
    owned by the thermal simulation.  The dt system sees one engine and one
    publication row, regardless of how many objects and thermal domains are
    present inside it.
    """

    def __init__(self, assemblies=(), *, source_w=None, state=None):
        if isinstance(assemblies, ThermalAssembly):
            assemblies = (assemblies,)
        self.assemblies = list(assemblies)
        if not all(isinstance(item, ThermalAssembly) for item in self.assemblies):
            raise TypeError("ThermalSystem accepts ThermalAssembly objects")
        self.source_w = source_w
        self.state = state
        self.world_time = 0.0
        self.observer_time = 0.0
        self.last_metrics = None
        self.last_conservation_error_j = 0.0

    def register_assembly(self, assembly: ThermalAssembly) -> int:
        """Add one object's declared domains to the thermal-system batch."""
        if not isinstance(assembly, ThermalAssembly):
            raise TypeError("thermal registration requires a ThermalAssembly")
        self.assemblies.append(assembly)
        return len(self.assemblies) - 1

    @property
    def active_count(self) -> int:
        return len(self.assemblies)

    def _sources(self, index: int, assembly: ThermalAssembly) -> dict[str, float]:
        source = self.source_w
        if source is None:
            return {}
        if callable(source):
            return dict(source(index, assembly) or {})
        if len(self.assemblies) == 1 and isinstance(source, dict):
            return dict(source)
        return dict(source[index] or {})

    def preferred_dt(self):
        limits = [assembly._stable_dt() for assembly in self.assemblies]
        finite = [value for value in limits if math.isfinite(value)]
        return min(finite) if finite else None

    def causal_ceiling_dt(self):
        limit = self.preferred_dt()
        return math.inf if limit is None else float(limit)

    def step(self, dt: float, state=None, state_table=None):
        dt_s = float(dt)
        if not self.assemblies:
            metrics = Metrics(
                max_vel=0.0, max_flux=0.0, div_inf=0.0, mass_err=0.0,
                advanced_dt=dt_s,
            )
            self.last_metrics = metrics
            return True, metrics, state
        energy_before = sum(assembly.energy_j for assembly in self.assemblies)
        sources = [self._sources(index, assembly)
                   for index, assembly in enumerate(self.assemblies)]
        signed_power_w = sum(sum(float(value) for value in item.values())
                             for item in sources)
        exchange_power_w = sum(sum(abs(float(value)) for value in item.values())
                               for item in sources)
        for assembly, source in zip(self.assemblies, sources):
            assembly.step(dt_s, source_w=source)
        energy_after = sum(assembly.energy_j for assembly in self.assemblies)
        conservation_error_j = abs(
            (energy_after - energy_before) - signed_power_w * dt_s)
        self.last_conservation_error_j = conservation_error_j

        channels = empty_channels()
        present = empty_channels()
        energy_slot = DT_CHANNEL_NAMES.index("energy_j")
        power_slot = DT_CHANNEL_NAMES.index("power_w")
        channels[energy_slot] = energy_after
        channels[power_slot] = exchange_power_w
        present[energy_slot] = 1.0
        present[power_slot] = 1.0
        tau_present = energy_after > 0.0 and exchange_power_w > 0.0
        limit = self.preferred_dt()
        metrics = Metrics(
            max_vel=0.0, max_flux=0.0, div_inf=0.0,
            mass_err=0.0,
            dt_limit=limit,
            error_channels=channels,
            error_present=present,
            pub_tau=AbstractTensor.tensor([
                energy_after / exchange_power_w if tau_present else 0.0]),
            pub_tau_present=AbstractTensor.tensor([float(tau_present)]),
            pub_contract=AbstractTensor.tensor([BIND if tau_present else HOLD]),
            pub_dt_limit=AbstractTensor.tensor([
                0.0 if limit is None else float(limit)]),
            pub_dt_limit_present=AbstractTensor.tensor([float(limit is not None)]),
            pub_values=channels.copy(),
            pub_present=present.copy(),
            pub_limits=AbstractTensor.zeros_like(channels),
            pub_limits_present=AbstractTensor.zeros_like(present),
            advanced_dt=dt_s,
        )
        self.last_metrics = metrics
        return True, metrics, state

    def get_state(self, state=None):
        return self.state if state is None else state

    def snapshot(self):
        return (
            tuple(assembly.copy_shallow() for assembly in self.assemblies),
            float(self.world_time), float(self.observer_time), self.last_metrics,
            float(self.last_conservation_error_j),
        )

    def restore(self, snapshot) -> None:
        (assemblies, self.world_time, self.observer_time, self.last_metrics,
         self.last_conservation_error_j) = snapshot
        if len(assemblies) != len(self.assemblies):
            raise ValueError("thermal engine snapshot batch extent changed")
        for assembly, saved in zip(self.assemblies, assemblies):
            assembly.restore(saved)


# Compatibility spelling for the first users of the graduated system.  The
# object is one thermal system; its assemblies are internal batch entries.
ThermalEngine = ThermalSystem


def build_shell_domain(node: dict, part: SolidPart) -> ShellThermalDomain:
    vertices, triangles = weld_triangle_soup(part.vertices)
    geometry = build_cotangent_geometry(vertices, triangles)
    return ShellThermalDomain(
        identity=str(node["identity"]), group=node.get("thermal_group"),
        material=node.get("material"), vertices=vertices,
        triangles=triangles, geometry=geometry,
        thickness_m=float(node["thermal_shell_thickness_m"]))


def _volume_shape(node: dict, default_resolution: int) -> tuple[int, int, int]:
    declared = node.get("voxel_shape") or node.get("thermal_grid_shape")
    if declared is not None:
        shape = tuple(int(value) for value in declared)
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError(f"{node['identity']} thermal grid must be 3D and >=2")
        return shape
    half = np.asarray(node["body_half_extent_m"], dtype=np.float64)
    longest = max(float(half.max()), 1e-12)
    return tuple(max(2, int(round(default_resolution * value / longest)))
                 for value in half)


def build_volume_domain(node: dict, *, default_resolution: int = 3) -> VolumeThermalDomain:
    shape = _volume_shape(node, default_resolution)
    half = tuple(float(value) for value in node["body_half_extent_m"])
    lengths = tuple(2.0 * value for value in half)
    transform = laplace_nd.RectangularTransform(
        Lx=lengths[0], Ly=lengths[1], Lz=lengths[2], device="cpu")
    grid_u, grid_v, grid_w = transform.create_grid_mesh(*shape)
    grid = laplace_nd.GridDomain.generate_grid_domain(
        coordinate_system="rectangular", N_u=shape[0], N_v=shape[1],
        N_w=shape[2], Lx=lengths[0], Ly=lengths[1], Lz=lengths[2],
        device="cpu")
    builder = laplace_nd.BuildLaplace3D(
        grid_domain=grid, precision=None, resolution=max(shape))
    dense, sparse, _ = builder.build_general_laplace(
        grid_u, grid_v, grid_w,
        boundary_conditions=("neumann",) * 6, device="cpu")
    operator = dense if dense is not None else sparse.to_dense()
    laplacian = np.asarray(operator.tolist(), dtype=np.float64)
    return VolumeThermalDomain(
        identity=str(node["identity"]), group=node.get("thermal_group"),
        material=node.get("material"), shape=shape, half_extent_m=half,
        laplacian=laplacian)


def build_path_domain(node: dict) -> PathThermalDomain:
    """Build a one-dimensional Laplacian along declared world-space points."""
    positions = np.asarray(node["thermal_path_points_m"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError(
            f"{node['identity']} thermal_path_points_m must have shape (N, 3), N>=2")
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if np.any(segment_lengths <= 0.0):
        raise ValueError(f"{node['identity']} thermal path has coincident samples")
    nodal_lengths = np.empty(len(positions), dtype=np.float64)
    nodal_lengths[0] = segment_lengths[0] / 2.0
    nodal_lengths[-1] = segment_lengths[-1] / 2.0
    if len(positions) > 2:
        nodal_lengths[1:-1] = (
            segment_lengths[:-1] + segment_lengths[1:]) / 2.0

    # BuildGraphLaplace is the repository's AbstractTensor graph operator.
    # Edge weights are inverse squared intrinsic distance, giving the usual
    # one-dimensional heat operator for uniformly sampled paths.
    adjacency = np.zeros((len(positions), len(positions)), dtype=np.float64)
    weights = 1.0 / np.square(segment_lengths)
    indices = np.arange(len(segment_lengths))
    adjacency[indices, indices + 1] = weights
    adjacency[indices + 1, indices] = weights
    dense, _sparse, _metadata = BuildGraphLaplace(
        AbstractTensor.tensor(adjacency), normalization="none").build()
    laplacian = np.asarray(dense.tolist(), dtype=np.float64)
    return PathThermalDomain(
        identity=str(node["identity"]), group=node.get("thermal_group"),
        material=node.get("material"), positions=positions,
        nodal_lengths_m=nodal_lengths,
        cross_section_area_m2=float(node["thermal_cross_section_area_m2"]),
        laplacian=laplacian)


def _exchange_samples(domain):
    if isinstance(domain, PathThermalDomain):
        perimeter = getattr(domain, "exchange_perimeter_m", None)
        return domain.positions, domain.nodal_lengths_m, "length"
    if isinstance(domain, ShellThermalDomain):
        return (domain.vertices,
                np.asarray(domain.geometry.lumped_vertex_areas, dtype=np.float64),
                "area")
    raise ValueError("distributed thermal exchange needs path or shell domains")


def build_distributed_interface(edge: dict,
                                domains: dict[str, object]) -> DistributedThermalInterface:
    """Match two geometries and distribute a calibrated U or UA over them."""
    a_id, b_id = str(edge["a"]), str(edge["b"])
    a, b = domains[a_id], domains[b_id]
    a_positions, a_measure, measure_kind = _exchange_samples(a)
    b_positions, _b_measure, _ = _exchange_samples(b)
    if measure_kind == "length":
        area = a_measure * float(edge["thermal_exchange_perimeter_m"])
    else:
        area = a_measure
    if "heat_exchange_ua_w_per_k" in edge:
        ua = float(edge["heat_exchange_ua_w_per_k"])
        local_g = ua * area / max(float(area.sum()), 1e-30)
    else:
        coefficient = float(edge["heat_transfer_coefficient_w_m2_k"])
        local_g = coefficient * area
    if np.any(local_g < 0.0):
        raise ValueError(f"{edge['identity']} thermal exchange values must be nonnegative")

    # Geometry, rather than array order, decides which parts of counterflow
    # passages face each other.  Each A sample gives its local exchange area
    # to its nearest B sample.  The resulting matrix is the conservative
    # conductance map used by ThermalAssembly.
    distance2 = np.sum(
        (a_positions[:, None, :] - b_positions[None, :, :]) ** 2, axis=2)
    nearest_b = np.argmin(distance2, axis=1)
    conductance = np.zeros((len(a_positions), len(b_positions)), dtype=np.float64)
    conductance[np.arange(len(a_positions)), nearest_b] = local_g
    return DistributedThermalInterface(a_id, b_id, conductance)


def electrical_conductor_thermal_nodes(graph: dict) -> list[dict]:
    """Project declared cable conductors into copper path thermal domains."""
    node_by_id = {str(node["identity"]): node for node in graph["nodes"]}
    out = []
    for edge in graph["edges"]:
        if edge.get("transport_domain") != "electrical":
            continue
        a = node_by_id.get(str(edge["a"]), {}).get("reference_position")
        b = node_by_id.get(str(edge["b"]), {}).get("reference_position")
        if a is None or b is None:
            continue
        points = [list(a), *[list(row) for row in edge.get("waypoints", ())],
                  list(b)]
        for conductor in edge.get("conductors", ()):
            role = str(conductor["role"])
            identity = f"{edge['identity']}::{role}"
            out.append({
                "identity": identity,
                "kind": "electrical-conductor-thermal-path",
                "reference_position": [
                    float((float(a[i]) + float(b[i])) / 2.0) for i in range(3)],
                "body_half_extent_m": [0.001, 0.001, 0.001],
                "mass_kg": 0.0,
                "material": "copper",
                "thermal_domain": "path",
                "thermal_group": str(edge.get("thermal_group", identity)),
                "thermal_path_points_m": points,
                "thermal_cross_section_area_m2": (
                    float(conductor["area_mm2"]) * 1.0e-6),
                "electrical_edge_identity": str(edge["identity"]),
                "electrical_conductor_role": role,
                "ampacity_a": float(conductor.get("ampacity_a", 0.0)),
            })
    return out


def _thermal_node_records(graph: dict) -> list[dict]:
    records = [dict(node) for node in graph["nodes"]]
    known = {str(node["identity"]) for node in records}
    for node in electrical_conductor_thermal_nodes(graph):
        if node["identity"] in known:
            raise ValueError(
                f"thermal identity {node['identity']!r} is authored twice")
        records.append(node)
        known.add(node["identity"])
    return records


def build_thermal_domains(graph: dict, *, default_volume_resolution: int = 3,
                          include_external: bool = True):
    """Build every declared shell/volume domain from one assembled graph."""
    thermal_nodes = _thermal_node_records(graph)
    parts = {_part_name(node["identity"]): node for node in thermal_nodes}
    needs_surface_mesh = any(
        node.get("thermal_domain") == "shell" for node in thermal_nodes)
    mesh_parts = ({part.name: part
                   for part in build_drivetrain_solid_parts(graph)}
                  if needs_surface_mesh else {})
    domains = {}
    for part_name, node in parts.items():
        if node.get("thermal_state_owner") and not include_external:
            continue
        kind = node.get("thermal_domain")
        if kind is None:
            continue
        if kind == "none":
            continue
        if kind == "shell":
            if part_name not in mesh_parts:
                raise ValueError(f"no surface mesh for thermal shell {node['identity']}")
            domain = build_shell_domain(node, mesh_parts[part_name])
        elif kind == "volume":
            domain = build_volume_domain(
                node, default_resolution=default_volume_resolution)
        elif kind == "path":
            domain = build_path_domain(node)
        else:
            raise ValueError(f"{node['identity']} has invalid thermal_domain {kind!r}")
        domains[domain.identity] = domain
    return domains


def build_thermal_assembly(graph: dict, *, initial_temperature_k=293.15,
                           temperatures_by_group: dict[str, float] | None = None,
                           interfaces=(), default_volume_resolution: int = 3):
    nodes = {str(node["identity"]): node
             for node in _thermal_node_records(graph)}
    domains = build_thermal_domains(
        graph, default_volume_resolution=default_volume_resolution,
        include_external=False)
    by_group = temperatures_by_group or {}
    states = {}
    for identity, domain in domains.items():
        node = nodes[identity]
        if node.get("thermal_state_owner"):
            continue
        material = material_for(node)
        temperature = float(by_group.get(domain.group, initial_temperature_k))
        if isinstance(domain, ShellThermalDomain):
            capacity = (domain.geometry.lumped_vertex_areas
                        * domain.thickness_m * material.density_kg_m3
                        * material.specific_heat_j_kg_k)
            values = np.full(len(domain.vertices), temperature, dtype=np.float64)
        elif isinstance(domain, VolumeThermalDomain):
            hx, hy, hz = domain.half_extent_m
            if node.get("shape") == "drum":
                volume = (math.pi * float(node["drum_radius_m"]) ** 2
                          * float(node["drum_length_m"]))
            else:
                volume = 8.0 * hx * hy * hz
            count = int(np.prod(domain.shape))
            capacity = np.full(
                domain.shape,
                volume * material.density_kg_m3
                * material.specific_heat_j_kg_k / count,
                dtype=np.float64)
            values = np.full(domain.shape, temperature, dtype=np.float64)
        else:
            capacity = (domain.nodal_lengths_m
                        * domain.cross_section_area_m2
                        * material.density_kg_m3
                        * material.specific_heat_j_kg_k)
            values = np.full(len(domain.positions), temperature, dtype=np.float64)
        states[identity] = ThermalDomainState(
            domain, material, values, capacity)
    declared = [build_distributed_interface(edge, domains)
                for edge in graph["edges"]
                if edge.get("thermal_exchange") in {
                    "distributed-path", "distributed-geometry"}]
    return ThermalAssembly(states, [*interfaces, *declared])


__all__ = [
    "ShellThermalDomain", "VolumeThermalDomain", "PathThermalDomain",
    "build_shell_domain", "build_path_domain", "build_distributed_interface",
    "ThermalAssembly", "ThermalDomainState", "ThermalInterface",
    "ThermalEngine", "ThermalSystem",
    "DistributedThermalInterface",
    "electrical_conductor_thermal_nodes",
    "ThermalMaterial", "build_thermal_assembly", "build_thermal_domains",
    "build_volume_domain", "material_for", "weld_triangle_soup",
]
