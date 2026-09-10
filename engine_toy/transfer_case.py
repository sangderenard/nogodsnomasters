"""A transfer case (or any pure gear train) built from the same parts
library as the engine block -- proof that "engine" is just one Assembly
among many. No CombustionCrankPart, no ValvetrainPart: cylinders, crank,
and detonation are simply absent rather than turned off.

Each gear stage is a RotaryTonePart whose `order` is its own shaft's
rpm ratio relative to the case's input shaft. A tooth count folds into
that as an extra multiplier (local_shaft_rpm/60 * tooth_count), which
is how a real gear mesh's dominant tone is calculated -- with normal
tooth counts (15-40 teeth) the mesh is well approximated as a single
tone rather than resolving individual tooth impacts.
"""
from __future__ import annotations

from dataclasses import dataclass

from mech_parts import Assembly, RotaryTonePart, Vec3


@dataclass
class GearStage:
    label: str
    ratio_from_input: float     # this shaft's rpm = input_rpm * ratio_from_input
    tooth_count: int
    position: Vec3
    base_amplitude: float = 0.20


def build_gear_train_assembly(name: str, stages: list[GearStage],
                               housing_points: dict[str, Vec3]) -> Assembly:
    parts = []
    for stage in stages:
        mesh_order = stage.ratio_from_input * stage.tooth_count

        def amp_fn(rpm, thr, load, base=stage.base_amplitude):
            # more load -> more mesh tone (tooth contact force rises with torque)
            return base * (0.4 + 0.6 * load)

        parts.append(RotaryTonePart(
            name=f"{stage.label}-mesh", domain="both", position=stage.position,
            order=mesh_order, amp_fn=amp_fn,
        ))
        # gear rattle/backlash chatter, most audible near zero load (slack in the mesh)
        parts.append(RotaryTonePart(
            name=f"{stage.label}-backlash", domain="structural", position=stage.position,
            order=stage.ratio_from_input, phase_rad=1.3,
            amp_fn=lambda rpm, thr, load, base=stage.base_amplitude: base * 0.15 * max(0.0, 1.0 - load * 3.0),
        ))
    return Assembly(name=name, parts=parts, acoustic_points=housing_points, structural_points=housing_points)


def example_two_speed_transfer_case() -> Assembly:
    """A generic part-time 2-speed transfer case: input pinion straight
    off the transmission output, a chain/planetary low-range reduction,
    and a front-output chain drive -- three meshing stages, three
    positions along one housing, no engine parts involved at all."""
    stages = [
        GearStage("input-pinion", ratio_from_input=1.0, tooth_count=23, position=(-0.08, 0.0, 0.0), base_amplitude=0.22),
        GearStage("low-range-planetary", ratio_from_input=2.72, tooth_count=31, position=(0.0, 0.0, 0.02), base_amplitude=0.28),
        GearStage("front-output-chain", ratio_from_input=1.0, tooth_count=38, position=(0.06, 0.09, -0.02), base_amplitude=0.18),
    ]
    housing_points = {
        "case_front": (-0.10, 0.0, -0.03),
        "case_rear": (0.10, 0.0, -0.03),
        "case_top": (0.0, 0.0, 0.08),
    }
    return build_gear_train_assembly("transfer-case:np231-style-2-speed", stages, housing_points)
