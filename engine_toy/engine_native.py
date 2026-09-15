"""THE ENGINE LOOP AS A COMPILABLE, LANE-BATCHED PROGRAM.

Shaped after `turing/src/compiler/vehicle_python_compilation.py`'s managed
balloon-tire program, which is the proven route: an authored Python
`*_managed_advance(material, dt)` over flat arrays, wrapped by a
`*_managed_window(...)` that calls the repository's OWN `run_superstep`,
and the whole thing -- controller loop and physics together -- lowered to
SSA and emitted as C. Not a Python loop around a compiled kernel: the
loops go inside.

WHAT IS AND IS NOT HERE

This is the CRANK CORE: the part of the engine loop that actually
integrates. Torque from the charge and the firing schedule, less pumping,
friction and load, over inertia, per lane. It is written against the flat
interior spans `engine_abi` declares, lane-major with a fixed stride, so
`state[lane * stride + slot]` is the addressing -- the same shape the
balloon tire indexes with `state + w*TIRE_STATE_STRIDE`.

It is deliberately NOT the whole of `engine_cycle_sim`. That module is
several thousand lines of Python object graph, and porting it wholesale
before anything is proven would be the wrong order. The crank core is
where the substeps actually go, so it is the piece whose compilation
changes the frame budget.

THE LANE AXIS IS THE BATCH AXIS IS THE POOL'S LANE

`turing_pool.h` dispatches `turing_lane_fn(context, lane, chunk,
chunks_per_lane)`; `HostDeploymentPool.deploy` takes a sequence of lanes;
`engine_abi` declares the engine index as the outer extent. They are the
same axis, so a compiled assembly dispatches onto the existing pool with
nothing to adapt.

ROLLBACK-COMPLETE, BECAUSE THE CONTROLLER NEEDS IT

`run_superstep` may reject a substep and retry it smaller. `copy_shallow`
and `restore` are what let it, and they have to cover every array the
advance writes or a rejected step leaves the state poisoned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


#: Interior slots this core owns, per lane. A subset of what
#: `engine_abi.engine_graph_abi` declares -- the crank core does not touch
#: the fluid-circuit or per-node spans, and declaring storage it does not
#: write would be claiming a contract it does not honour.
CRANK_SLOTS = (
    "crank_angle_deg",      # 0: total, not wrapped
    "crank_omega",          # 1: rad/s
    "charge_frac",          # 2: 0..1 cylinder filling
    "fired_deg",            # 3: crank angle of the last firing
)
CRANK_STRIDE = len(CRANK_SLOTS)

#: Per-lane parameters, in the order the flat parameter block carries them.
CRANK_PARAMS = (
    "displacement_m3",
    "bmep_pa",
    "braking_bmep_pa",
    "inertia_kg_m2",
    "cycle_degrees",
    "firing_interval_deg",   # cycle_degrees / cylinders
    "idle_omega",
    "friction_nm_per_rad_s",   # bearings, rings, valvetrain: rises with speed
    "windage_nm_per_rad_s2",   # oil churn and pumping the bay: rises with speed^2
)
CRANK_PARAM_STRIDE = len(CRANK_PARAMS)


@dataclass
class EngineCrankState:
    """Rollback-complete lane-major material for the crank core.

    `state` is (lanes, CRANK_STRIDE); `params` is (lanes,
    CRANK_PARAM_STRIDE); `inputs` is (lanes, 2) carrying throttle and load
    torque. Flat, typed, and nothing else crosses -- which is what makes
    it declarable where `EngineCycleSim` is not.
    """

    state: np.ndarray
    params: np.ndarray
    inputs: np.ndarray
    telemetry: np.ndarray
    declared_dt_s: float = 0.0

    @property
    def lanes(self) -> int:
        return int(self.state.shape[0])

    def dt_limit_hint(self) -> float:
        """The core's declared safe step, published to the controller.

        Zero means "no opinion" rather than "no limit" -- one physical
        return contract instead of a payload/None union, the same rule the
        tire's managed state follows."""
        declared = float(self.declared_dt_s)
        return declared if declared > 0.0 else 0.0

    def copy_shallow(self):
        return (self.state.copy(), self.inputs.copy(),
                float(self.declared_dt_s))

    def restore(self, snapshot) -> None:
        state, inputs, declared = snapshot
        self.state[...] = state
        self.inputs[...] = inputs
        self.declared_dt_s = float(declared)


#: The authored program. A STRING because this is what gets lowered --
#: parsed to AST, raised to SSA, emitted as C -- rather than executed as
#: Python in the shipped path. It is still ordinary, runnable Python, which
#: is what makes it checkable against the same arrays before any compiler
#: is invoked.
ENGINE_CRANK_SOURCE = r'''
def engine_crank_advance(material, dt):
    """One substep of the crank core, every lane at once.

    Semi-implicit: torque is evaluated at the CURRENT speed and the new
    speed is integrated before the angle, which is the stable ordering for
    a load that stiffens with speed. Every expression below is whole-array
    over the lane axis -- there is no per-lane branch, because a branch is
    what stops a lane loop from being a lane loop.
    """
    material.telemetry[0] = material.telemetry[0] + 1.0
    material.telemetry[1] = min(material.telemetry[1], dt)
    material.telemetry[2] = max(material.telemetry[2], dt)

    angle = material.state[:, 0]
    omega = material.state[:, 1]
    charge = material.state[:, 2]
    fired = material.state[:, 3]

    displacement = material.params[:, 0]
    bmep = material.params[:, 1]
    braking_bmep = material.params[:, 2]
    inertia = material.params[:, 3]
    cycle_degrees = material.params[:, 4]
    firing_interval = material.params[:, 5]
    idle_omega = material.params[:, 6]
    friction_coeff = material.params[:, 7]
    windage_coeff = material.params[:, 8]

    throttle = material.inputs[:, 0]
    load_torque = material.inputs[:, 1]

    # THE CHARGE. A first-order fill toward what the throttle admits, on
    # the engine's own clock. No branch: the lag is a multiply.
    charge_target = 0.06 + 0.94 * throttle
    fill = dt / (dt + 0.040)
    charge = charge + (charge_target - charge) * fill

    # THE FIRING SCHEDULE. How many firings fall inside this substep is a
    # real number, not a loop: at omega rad/s the crank turns
    # omega*dt*(180/pi) degrees, and a firing happens every
    # firing_interval of them. Fractional counts are exactly right here --
    # a substep that covers a third of an interval collects a third of the
    # impulse, which is what keeps the torque independent of substep size.
    turned_deg = omega * dt * 57.29577951308232
    firings = turned_deg / firing_interval

    # THE TORQUE. bmep over a four-stroke cycle is the standard relation,
    # T = bmep * Vd / (4*pi). Braking bmep is always present and always
    # opposes. `firings` is carried as state rather than scaling the
    # torque: the mean torque of a running engine does not depend on how
    # finely the substep samples its firing schedule, and making it do so
    # would make the result a function of K.
    gross_nm = bmep * displacement * charge / 12.566370614359172
    pumping_nm = braking_bmep * displacement / 12.566370614359172
    # SPEED-DEPENDENT LOSS, which is what actually sets a free engine's
    # running speed. Without it an unloaded lane accelerates without
    # bound -- correct for an engine with no friction and no limiter,
    # which is not an engine. Linear in omega for bearings, rings and
    # valvetrain; quadratic for oil churn and pumping the crankcase.
    friction_nm = friction_coeff * omega + windage_coeff * omega * omega
    net_nm = gross_nm - pumping_nm - friction_nm - load_torque

    # SEMI-IMPLICIT INTEGRATION, and the idle floor as a smooth max rather
    # than a branch: a stalled lane must not integrate backwards.
    omega = omega + (net_nm / inertia) * dt
    # A smooth max, not a branch: a stalled lane must not integrate
    # backwards, and a per-lane `if` is what stops a lane loop being a
    # lane loop. (The repository measured 2**-14 of residue from this
    # spelling in the SYMBOLIC path, where sympy.Max is the right answer;
    # in the AST path it lowers to plain arithmetic with no call.)
    omega = 0.5 * (omega + idle_omega + abs(omega - idle_omega))
    angle = angle + turned_deg

    fired = fired + firings

    material.state[:, 0] = angle
    material.state[:, 1] = omega
    material.state[:, 2] = charge
    material.state[:, 3] = fired
    fastest = omega.max()
    material.telemetry[3] = 0.5 * (
        material.telemetry[3] + fastest + abs(material.telemetry[3] - fastest))
    return material.state
'''


def build_crank_state(abis, *, throttle: float = 0.55) -> EngineCrankState:
    """Pack a fleet of engines into one lane-major material.

    Every lane must share a topology -- that is what makes the stride a
    compile-time constant -- so this asserts it rather than silently
    padding. `abi_negotiator.negotiate` is what produces a set that passes.
    """
    abis = list(abis)
    if not abis:
        raise ValueError("no lanes")
    topology = abis[0].topology
    for abi in abis:
        if abi.topology != topology:
            raise ValueError(
                f"lane topology {abi.topology} != {topology}; a batch is one "
                f"topology -- negotiate first")

    lanes = len(abis)
    state = np.zeros((lanes, CRANK_STRIDE), dtype=np.float64)
    params = np.zeros((lanes, CRANK_PARAM_STRIDE), dtype=np.float64)
    inputs = np.zeros((lanes, 2), dtype=np.float64)
    for lane, abi in enumerate(abis):
        p = abi.parameters
        # Friction and windage sized so the engine's own declared redline
        # is where an unloaded lane settles at full throttle -- derived
        # from the engine's own numbers, not typed in. Split 60/40 between
        # the linear and quadratic terms, a disclosed division.
        redline_omega = float(p["redline_rpm"]) * 0.10471975511965977
        peak_nm = float(p["bmep_pa"]) * float(p["displacement_m3"]) / 12.566370614359172
        drag_at_redline = peak_nm - (float(p["braking_bmep_pa"])
                                     * float(p["displacement_m3"]) / 12.566370614359172)
        params[lane] = (
            p["displacement_m3"], p["bmep_pa"], p["braking_bmep_pa"],
            p["inertia_kg_m2"], p["cycle_degrees"], _firing_interval(p),
            p["idle_rpm"] * 0.10471975511965977,
            0.60 * drag_at_redline / max(redline_omega, 1.0),
            0.40 * drag_at_redline / max(redline_omega ** 2, 1.0),
        )
        state[lane, 1] = params[lane, 6]      # start at idle, not stopped
        state[lane, 2] = 0.06
        inputs[lane, 0] = throttle
    telemetry = np.zeros(8, dtype=np.float64)
    telemetry[1] = 1e30                       # min dt seen
    return EngineCrankState(state=state, params=params, inputs=inputs,
                            telemetry=telemetry)


def _firing_interval(parameters) -> float:
    """Degrees between firings, from the declared schedule.

    Reads the angles the architecture published rather than assuming an
    even split, so an odd-fire crank carries its real mean interval."""
    angles = sorted(v for k, v in parameters.items()
                    if k.startswith("firing_angle_") and k.endswith("_deg"))
    if len(angles) < 2:
        return float(parameters.get("cycle_degrees", 720.0))
    return float(parameters["cycle_degrees"]) / float(len(angles))


def crank_advance_callable():
    """The authored source, compiled by Python itself, for checking.

    Running the SAME text that will be lowered is the point: a divergence
    between what was verified and what was compiled cannot open up if
    there is only one copy of it.
    """
    namespace: dict = {"min": min, "max": max, "abs": abs}
    exec(compile(ENGINE_CRANK_SOURCE, "<engine_crank_source>", "exec"), namespace)
    return namespace["engine_crank_advance"]


if __name__ == "__main__":
    import engines
    from engine_abi import engine_graph_abi

    # three Bussos: one topology, three lanes, different throttles
    fleet = [engine_graph_abi(engines.get("alfa-busso-v6-3000-12v"))
             for _ in range(3)]
    material = build_crank_state(fleet)
    material.inputs[:, 0] = (0.2, 0.55, 1.0)

    advance = crank_advance_callable()
    dt = 1.0 / 2000.0
    for _ in range(4000):                      # two seconds of engine time
        advance(material, dt)

    print(f"lanes={material.lanes}  stride={CRANK_STRIDE}  "
          f"substeps={material.telemetry[0]:.0f}")
    for lane in range(material.lanes):
        omega = material.state[lane, 1]
        print(f"   lane {lane}  throttle {material.inputs[lane, 0]:.2f}"
              f"  -> {omega * 9.549296585513721:7.0f} rpm"
              f"   charge {material.state[lane, 2]:.3f}"
              f"   firings {material.state[lane, 3]:8.1f}")
    print("one array pass per substep, no per-lane branch: this is the shape")
    print("that lowers, and the lane axis is the pool's lane.")
