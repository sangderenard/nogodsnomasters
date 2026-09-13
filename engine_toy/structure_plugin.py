"""THE BEAM LAW, PUBLISHED AS A GAME-ENGINE COMPONENT.

The game engine is `turing/src/compiler/abstract_ui_*` -- the one that
becomes the playable thing, with items, tools and interaction. Its
contract for a piece of physics is not "a Python loop someone wrote". It
is, exactly as `abstract_ui_physics.symbolic_world_physics_wasm_plugin`
does it:

    a law authored as simultaneous SymPy equations
        -> compile_sympy_equations   (SymPy -> ProcessGraph -> SSA)
        -> emit_ssa_function_to_wasm (SSA -> WebAssembly, directly)
        -> WorldWasmPlugin           (published through the world ABI,
                                      with a declared capability)

and the host schedules it. The equations are "the only numerical
authority"; the host supplies parameters through the published arena so
they stay live-editable without recompiling. That is what makes a
component OPTIONAL -- the world can bind it or not -- and it is why the
integrator belongs to the scheduler rather than to the law.

WHAT WAS WRONG WITH WHAT I HAD. `live_scene.LiveStructure` steps the
same oscillator in Python, on a substep count of its own, against a
stability margin I first invented and then took from the ENGINE engine's
drivetrain solver. Both are the wrong authority: the engine engine is a
separate integrator for engine items, and neither it nor I get to decide
the cadence of a law the game engine is supposed to be scheduling. A law
that owns its own stepping cannot be optional, cannot be rescheduled,
and cannot be compiled.

THE LAW ITSELF IS ALREADY SYMPY AND ALREADY CORRECT.
`beam_theory.symbolic_beam_dynamics_equations` is the first-mode
oscillator with temperature-dependent modulus, semi-implicit, and it has
been validated to 0.05% on damping. Nothing about it changes here. What
changes is who runs it.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

_TURING = Path(__file__).resolve().parent.parent / "turing"
if str(_TURING) not in sys.path:
    sys.path.insert(0, str(_TURING))


#: The parameter surface the host may edit live, in the game engine's
#: own `PhysicsParameter` shape. Every one of these is a real property
#: of a member -- none is a tuning knob -- and the damping ratio is the
#: single term that cannot be derived from geometry, which is why it is
#: the only one with a wide range.
def parameter_surface():
    from src.compiler.abstract_ui_physics import PhysicsParameter
    return (
        PhysicsParameter("length_m", 1.0, "m", "section", 0.01, 20.0),
        PhysicsParameter("outer_radius_m", 0.048, "m", "section", 0.002, 0.5),
        PhysicsParameter("wall_m", 0.004, "m", "section", 0.0005, 0.05),
        PhysicsParameter("density_kg_m3", 7850.0, "kg/m^3", "material",
                         1000.0, 20000.0),
        PhysicsParameter("youngs_cold_pa", 2.05e11, "Pa", "material",
                         1e9, 5e11),
        PhysicsParameter("temperature_k", 293.15, "K", "state", 200.0, 1400.0),
        PhysicsParameter("damping_ratio", 0.015, "-", "state", 0.0005, 0.2),
        PhysicsParameter("added_mass_per_m_kg", 0.0, "kg/m", "section",
                         0.0, 500.0),
        PhysicsParameter("force_n", 0.0, "N", "drive", -5e6, 5e6),
        PhysicsParameter("dt_s", 1.0 / 240.0, "s", "schedule", 1e-6, 0.05,
                         live_editable=False),
        PhysicsParameter("modal_q_m", 0.0, "m", "state", -1.0, 1.0),
        PhysicsParameter("modal_qdot_m_s", 0.0, "m/s", "state", -100.0, 100.0),
    )


@lru_cache(maxsize=1)
def compiled():
    """The beam law through the game engine's own compile path.

    `beam_theory.compile_beam_dynamics_ssa` uses the same underlying
    compiler but publishes under engine_toy's own names; this goes
    through `compile_sympy_equations` exactly as the world physics does,
    so the artifact is the same KIND of thing the scheduler already
    binds."""
    from src.compiler.symbolic_equation_compiler import (
        compile_sympy_equations, SymbolicPublication)
    import beam_theory
    equations, _ = beam_theory.symbolic_beam_dynamics_equations()
    return compile_sympy_equations(
        equations,
        name="abstract_ui_beam_dynamics_step",
        publications=tuple(
            SymbolicPublication(name, f"abstract-ui.structure.{name}")
            for name in beam_theory.BEAM_DYNAMICS_OUTPUTS),
    )


@lru_cache(maxsize=1)
def wasm_artifact():
    """Straight to WebAssembly, and LOUD if it does not get there.

    A law that only half lowers is not an optional component, it is a
    Python function with extra steps. The world physics plugin raises on
    an incomplete artifact for the same reason and this does not get to
    be more forgiving."""
    from src.compiler.ssa_wasm_backend import emit_ssa_function_to_wasm
    c = compiled()
    # THE CONTRACT IS PART OF THE DECLARATION. `develop` refuses the
    # sqrt-family reduction on purpose -- it is the contract that admits
    # nothing it cannot spell exactly -- and a natural frequency is a
    # square root, so under `develop` this law can never lower. `deploy`
    # permits that reduction, which the emitter's own refusal message
    # says in as many words. Naming it here is a statement about what
    # this component is allowed to be built with, not a way round a
    # complaint.
    artifact = emit_ssa_function_to_wasm(c.module, c.function.name,
                                         work_contract="deploy")
    if not artifact.complete:
        reasons = "; ".join(item.reason for item in artifact.shortfalls)
        raise RuntimeError(
            f"the beam law does not lower to WASM: {reasons}")
    return artifact


@lru_cache(maxsize=1)
def plugin():
    """The component, published through the world ABI."""
    from src.compiler.abstract_ui_world import WorldWasmPlugin
    import beam_theory
    equations, _ = beam_theory.symbolic_beam_dynamics_equations()
    c = compiled()
    artifact = wasm_artifact()
    return WorldWasmPlugin(
        "abstract-ui/plugins/structural-beam-dynamics",
        "integrate-one-member's-first-bending-mode",
        artifact.binary,
        artifact.name,
        ({"name": "io", "role": "arena-base", "dtype": "int32"},),
        "\n".join(str(eq) for eq in equations),
        source_language="sympy",
        capability="physics",
        operation_count=sum(len(b.instrs)
                            for b in c.function.blocks.values()),
        reserved_bytes=max((*artifact.input_offsets,
                            *artifact.output_offsets)) + 8,
        abi={
            "kind": "ssa-scalar-arena-v0",
            "invocation": "arena-base-pointer",
            "dtype": "float64",
            "input_names": list(artifact.input_names),
            "output_names": list(artifact.output_names),
        },
    )


def describe() -> list:
    p = plugin()
    a = wasm_artifact()
    return [
        f"identity   {p.identity}",
        f"capability {p.capability}   source {p.source_language}",
        f"entrypoint {p.entrypoint}",
        f"wasm       {len(p.binary)} bytes, {p.content_key}",
        f"operations {p.operation_count}",
        f"inputs     {', '.join(a.input_names)}",
        f"outputs    {', '.join(a.output_names)}",
        f"parameters {len(parameter_surface())} live-editable "
        f"through the published arena",
    ]


if __name__ == "__main__":
    print("\n".join(describe()))
