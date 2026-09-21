"""Physical AC/DC distribution declarations for machine production graphs.

This extends :mod:`dc_power` rather than replacing it.  Its ``Conductor`` and
``PowerPort`` remain the material and connector primitives.  A service says
which conductors exist; a cable instantiates one real copper conductor per
role; a breaker protects its ungrounded conductors; and an outlet exposes that
same service to another machine.

Neutral, DC return, and protective earth are deliberately distinct.  A
generator produces EMF between winding terminals.  A source distribution
panel may bond neutral to protective earth, but the generator does not
"produce ground" and downstream panels do not silently create another bond.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from dc_power import AWG_MM2, Conductor, PowerPort


# NFPA 70 Table 310.16 copper ampacities at the termination temperature.
# The small-conductor caps below remain separate because a conductor's table
# ampacity and its permitted branch-circuit overcurrent protection are not the
# same rule.
NEC_COPPER_AMPACITY_A = {
    60: {0: 125, 2: 95, 4: 70, 6: 55, 8: 40, 10: 30,
         12: 20, 14: 15, 16: 10, 18: 7, 20: 5},
    75: {0: 150, 2: 115, 4: 85, 6: 65, 8: 50, 10: 35,
         12: 25, 14: 20, 16: 10, 18: 7, 20: 5},
}
SMALL_CONDUCTOR_OCPD_CAP_A = {10: 30.0, 12: 20.0, 14: 15.0}


_ARRANGEMENT_ROLES = {
    "dc-two-wire": ("positive", "return", "protective-earth"),
    "single-phase": ("line", "neutral", "protective-earth"),
    "split-phase": ("line-1", "line-2", "neutral", "protective-earth"),
    "three-phase-wye": (
        "line-1", "line-2", "line-3", "neutral", "protective-earth"),
    "three-phase-delta": (
        "line-1", "line-2", "line-3", "protective-earth"),
}


@dataclass(frozen=True)
class ElectricalService:
    """The voltage/frequency and conductor arrangement at one interface."""

    identity: str
    arrangement: str
    line_voltage_v: float
    frequency_hz: float = 0.0
    line_neutral_voltage_v: float | None = None

    def __post_init__(self) -> None:
        if self.arrangement not in _ARRANGEMENT_ROLES:
            raise ValueError(f"unknown electrical arrangement: {self.arrangement}")
        if self.line_voltage_v <= 0.0:
            raise ValueError("electrical service voltage must be positive")
        if self.arrangement == "dc-two-wire":
            if self.frequency_hz != 0.0:
                raise ValueError("a DC service has zero frequency")
        elif self.frequency_hz <= 0.0:
            raise ValueError("an AC service needs a positive frequency")
        if (self.arrangement == "three-phase-wye"
                and self.line_neutral_voltage_v is None):
            raise ValueError("a wye service must declare line-neutral voltage")

    @property
    def conductor_roles(self) -> tuple[str, ...]:
        return _ARRANGEMENT_ROLES[self.arrangement]

    @property
    def ungrounded_roles(self) -> tuple[str, ...]:
        return tuple(role for role in self.conductor_roles
                     if role not in ("neutral", "return", "protective-earth"))

    @property
    def pole_count(self) -> int:
        return len(self.ungrounded_roles)

    def graph_attributes(self) -> dict:
        return {
            "electrical_service": self.identity,
            "phase_arrangement": self.arrangement,
            "line_voltage_v": float(self.line_voltage_v),
            "line_neutral_voltage_v": (
                None if self.line_neutral_voltage_v is None
                else float(self.line_neutral_voltage_v)),
            "frequency_hz": float(self.frequency_hz),
            "conductor_roles": list(self.conductor_roles),
        }


@dataclass(frozen=True)
class BuildingCable:
    """One installed cable or conduit fill, with each conductor explicit."""

    identity: str
    service: ElectricalService
    awg: int
    length_m: float
    solid_copper: bool = True
    in_conduit: bool = True
    termination_rating_c: int = 75
    allowable_ampacity_a: float | None = None
    junction_resistance_ohm: float = 0.0005
    junction_torque_verified: bool = True
    connector_listing: str = "UL-486A-486B"
    parallel_sets: int = 1

    def __post_init__(self) -> None:
        if self.awg not in AWG_MM2:
            raise ValueError(f"{self.identity}: unsupported AWG {self.awg}")
        if self.length_m <= 0.0:
            raise ValueError(f"{self.identity}: cable length must be positive")
        if self.termination_rating_c not in NEC_COPPER_AMPACITY_A:
            raise ValueError(
                f"{self.identity}: termination rating must be 60 or 75 C")
        code_ampacity = NEC_COPPER_AMPACITY_A[self.termination_rating_c][self.awg]
        if self.allowable_ampacity_a is not None:
            if not 0.0 < self.allowable_ampacity_a <= code_ampacity:
                raise ValueError(
                    f"{self.identity}: installed ampacity must not exceed the "
                    "termination-temperature ampacity")
        if self.junction_resistance_ohm < 0.0:
            raise ValueError(f"{self.identity}: junction resistance is negative")
        if int(self.parallel_sets) != self.parallel_sets or self.parallel_sets < 1:
            raise ValueError(f"{self.identity}: parallel_sets must be a positive integer")
        if self.parallel_sets > 1 and self.awg != 0:
            raise ValueError(
                f"{self.identity}: general parallel conductors must be 1/0 AWG")

    @property
    def ampacity_a(self) -> float:
        table = NEC_COPPER_AMPACITY_A[self.termination_rating_c][self.awg]
        return float(self.allowable_ampacity_a or table)

    @property
    def aggregate_ampacity_a(self) -> float:
        return self.ampacity_a * int(self.parallel_sets)

    @property
    def conductors(self) -> tuple[Conductor, ...]:
        return tuple(
            Conductor(
                identity=(f"{self.identity}.{role}"
                          if self.parallel_sets == 1 else
                          f"{self.identity}.{role}.{parallel_index + 1}"),
                awg=self.awg, length_m=self.length_m,
                both_directions=False,
            )
            for role in self.service.conductor_roles
            for parallel_index in range(int(self.parallel_sets))
        )

    def graph_attributes(self) -> dict:
        rows = []
        for index, conductor in enumerate(self.conductors):
            role_index = index // int(self.parallel_sets)
            parallel_index = index % int(self.parallel_sets)
            service_role = self.service.conductor_roles[role_index]
            role = (service_role if self.parallel_sets == 1
                    else f"{service_role}.{parallel_index + 1}")
            rows.append({
                "role": role,
                "service_role": service_role,
                "parallel_index": parallel_index,
                "parallel_sets": int(self.parallel_sets),
                "awg": int(self.awg),
                "area_mm2": float(conductor.area_mm2),
                "resistance_ohm": float(conductor.resistance_ohm),
                "resistance_ohm_at_20c": (
                    float(conductor.resistance_ohm)
                    / (1.0 + 0.00393 * (conductor.temperature_c - 20.0))),
                "temperature_coefficient_per_k": 0.00393,
                "ampacity_a": float(self.ampacity_a),
                "material": "solid-copper" if self.solid_copper else "stranded-copper",
            })
        return {
            **self.service.graph_attributes(),
            "transport_domain": "electrical",
            "cable_identity": self.identity,
            "cable_length_m": float(self.length_m),
            "in_conduit": bool(self.in_conduit),
            "parallel_sets": int(self.parallel_sets),
            "parallel_routing": (
                "one-set-per-raceway" if self.parallel_sets > 1
                else "single-set"),
            "conductors": rows,
            "junctions": [
                {"endpoint": endpoint,
                 "resistance_ohm_at_20c": float(self.junction_resistance_ohm),
                 "temperature_coefficient_per_k": 0.00393,
                 "temperature_rating_c": int(self.termination_rating_c),
                 "torque_verified": bool(self.junction_torque_verified),
                 "connector_listing": self.connector_listing}
                for endpoint in ("a", "b")
            ],
        }


@dataclass(frozen=True)
class CircuitBreaker:
    identity: str
    service: ElectricalService
    rating_a: float
    cable: BuildingCable

    def __post_init__(self) -> None:
        if self.cable.service != self.service:
            raise ValueError(f"{self.identity}: breaker and cable services differ")
        if not 0.0 < self.rating_a <= self.cable.aggregate_ampacity_a:
            raise ValueError(
                f"{self.identity}: {self.rating_a:g} A breaker exceeds "
                f"{self.cable.aggregate_ampacity_a:g} A installed cable ampacity")
        cap = SMALL_CONDUCTOR_OCPD_CAP_A.get(self.cable.awg)
        if cap is not None and self.rating_a > cap * self.cable.parallel_sets:
            raise ValueError(
                f"{self.identity}: {self.cable.awg} AWG general circuit "
                f"overcurrent protection is capped at "
                f"{cap * self.cable.parallel_sets:g} A")

    @property
    def poles(self) -> int:
        return self.service.pole_count

    @property
    def continuous_load_limit_a(self) -> float:
        return self.rating_a / 1.25

    def graph_attributes(self) -> dict:
        return {
            "breaker_identity": self.identity,
            "breaker_rating_a": float(self.rating_a),
            "breaker_poles": int(self.poles),
            "breaker_protects": list(self.service.ungrounded_roles),
            "continuous_load_limit_a": float(self.continuous_load_limit_a),
            "continuous_load_limit_per_conductor_a": float(
                self.continuous_load_limit_a / self.cable.parallel_sets),
        }


@dataclass(frozen=True)
class OutletBox:
    identity: str
    service: ElectricalService
    breaker: CircuitBreaker
    connector_standard: str
    exterior: bool = True
    contact_resistance_ohm: float = 0.0005

    def __post_init__(self) -> None:
        if self.breaker.service != self.service:
            raise ValueError(f"{self.identity}: outlet and breaker services differ")

    @property
    def ports(self) -> tuple[PowerPort, ...]:
        return tuple(PowerPort(
            identity=f"{self.identity}.{role}", exterior=self.exterior,
            rating_a=self.breaker.rating_a,
            contact_resistance_ohm=self.contact_resistance_ohm,
        ) for role in self.service.conductor_roles)

    def accepts(self, required: ElectricalService) -> bool:
        return (
            self.service.arrangement == required.arrangement
            and self.service.line_voltage_v == required.line_voltage_v
            and self.service.frequency_hz == required.frequency_hz
            and self.service.line_neutral_voltage_v
                == required.line_neutral_voltage_v
        )

    def graph_attributes(self) -> dict:
        return {
            **self.service.graph_attributes(),
            **self.breaker.graph_attributes(),
            "connector_standard": self.connector_standard,
            "exterior_receptacle": bool(self.exterior),
            "terminal_roles": list(self.service.conductor_roles),
        }


@dataclass(frozen=True)
class DistributionPanel:
    """A source or subpanel containing independently protected outlets."""

    identity: str
    source_service: ElectricalService
    main_rating_a: float
    outlets: tuple[OutletBox, ...] = field(default_factory=tuple)
    neutral_to_frame_bond: bool = False

    def __post_init__(self) -> None:
        if self.main_rating_a <= 0.0:
            raise ValueError(f"{self.identity}: main rating must be positive")
        names = [outlet.identity for outlet in self.outlets]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.identity}: duplicate outlet identity")

    def compatible_outlets(self, required: ElectricalService) -> tuple[OutletBox, ...]:
        return tuple(outlet for outlet in self.outlets if outlet.accepts(required))

    def graph_attributes(self) -> dict:
        return {
            **self.source_service.graph_attributes(),
            "electrical_port_model": "distribution-panel",
            "main_breaker_rating_a": float(self.main_rating_a),
            "neutral_to_frame_bond": bool(self.neutral_to_frame_bond),
            "outlet_identities": [outlet.identity for outlet in self.outlets],
        }


def standard_services(prefix: str = "site") -> dict[str, ElectricalService]:
    """Common interfaces; selecting one remains an authored machine fact."""

    return {
        "48v-dc": ElectricalService(
            f"{prefix}.48v-dc", "dc-two-wire", 48.0),
        "230v-1ph-50hz": ElectricalService(
            f"{prefix}.230v-1ph-50hz", "single-phase", 230.0, 50.0, 230.0),
        "120-240v-split-60hz": ElectricalService(
            f"{prefix}.120-240v-split-60hz", "split-phase", 240.0, 60.0, 120.0),
        "230-400v-3ph-50hz": ElectricalService(
            f"{prefix}.230-400v-3ph-50hz", "three-phase-wye",
            400.0, 50.0, 230.0),
        "277-480v-3ph-60hz": ElectricalService(
            f"{prefix}.277-480v-3ph-60hz", "three-phase-wye",
            480.0, 60.0, 277.0),
    }


__all__ = [
    "ElectricalService", "BuildingCable", "CircuitBreaker", "OutletBox",
    "DistributionPanel", "standard_services",
]
