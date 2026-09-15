"""Kinetic consequences of a time change that spreads through a region.

Two questions this answers, both by measurement on the existing field rather
than by assertion.

**Negative drift.** Nothing in the field clamps ``log_tau`` at zero, so a zone
running FASTER than reference is already representable: it is the same
difference with the opposite sign. What it costs is substeps, exactly linearly,
because the stability floor is never relaxed. Past the substep budget the
window shortens instead of the step coarsening, so the failure mode is the same
one dilation has -- correct, but behind.

**Diffusion and radius.** A time change does not arrive everywhere at once. If
``log_tau`` spreads through a coupling region, then what a joint at graph
distance ``r`` feels is not the change itself but its DERIVATIVE, delayed and
attenuated. This module measures that: peak reaction, time to peak, and the
angular impulse delivered at each radius.

The rule the measurements support:

    Kinetic energy is never transferred between time scales. It is a per-zone
    quantity on that zone's own clock. What crosses a boundary is work, torque
    times angle, and both are clock-free -- which is why the coupled ledger
    closes at every ratio. The only coupling of the field to mechanics is the
    reaction at an adaptor while the gradient is CHANGING.

Run ``python time_diffusion.py``.
"""

from __future__ import annotations

import math

import dataclasses

from time_field import ADAPTORS, TimeField, TimeFieldConfig


def with_store_inertia(kind: str, inertia_kg_m2: float) -> None:
    """State the freedom's own inertia, which the table leaves at zero.

    ``ADAPTORS`` ships every ``store_inertia_kg_m2`` at 0.0, meaning "not
    chosen yet", so a reaction computed against the bare table is identically
    zero. ``time_rig`` states its own; this module states the same figures so
    the numbers below are comparable with the rig's.
    """

    ADAPTORS[kind] = dataclasses.replace(
        ADAPTORS[kind], store_inertia_kg_m2=float(inertia_kg_m2),
    )


# ---------------------------------------------------------------------
# negative drift: a zone running faster than reference
# ---------------------------------------------------------------------

def negative_drift_ledger():
    """The coupled ledger at ratios above 1, measured on the existing rig.

    ``time_rig`` publishes 1:1, 2:1, 10:1 and both-dilated, all at or below
    reference. These are the same rig with a side running FASTER. If the
    ledger closes here too then negative drift is not a new regime; it is the
    same difference field with the opposite sign.
    """

    from time_rig import CoupledRig

    rows = []
    for label, tau_a, tau_b in (("reference", 1.0, 1.0),
                                ("b at 2x", 1.0, 2.0),
                                ("b at 4x", 1.0, 4.0),
                                ("a fast, b slow", 2.0, 0.5),
                                ("both fast", 3.0, 1.5),
                                ("b at 10x", 1.0, 10.0)):
        rig = CoupledRig.build(tau_a=tau_a, tau_b=tau_b)
        for _ in range(30):
            rig.step()
        rows.append((label, tau_a, tau_b, rig.ledger_error_j))
    return rows


def negative_drift_cost(*, frame_s: float = 1.0 / 60.0,
                        dt_limits_s=(50e-6, 1000e-6, 250e-6, 500e-6),
                        max_substeps: int = 4096):
    """What running fast costs, and what happens past the budget.

    The window a lane advances is ``tau * frame``; the step it may take is
    fixed by its own stability floor. So the substep count is linear in tau.
    Past ``max_substeps`` the negotiator caps K and SHORTENS the window, which
    means asking for more world time than the budget affords returns less
    world time rather than a coarser step.
    """

    from abi_negotiator import negotiate_substeps

    ceiling = max_substeps * dt_limits_s[0] / frame_s
    rows = []
    for tau in (0.2, 1.0, 2.0, 4.0, 12.0, 20.0, 40.0):
        windows = [frame_s * tau, *dt_limits_s[1:]] and [
            frame_s * tau, frame_s, frame_s, frame_s
        ]
        schedule = negotiate_substeps(list(windows), list(dt_limits_s),
                                      max_substeps=max_substeps)
        rows.append((tau, schedule.substeps, schedule.steps_s[0],
                     frame_s * tau, schedule.windows_s[0],
                     schedule.respects(list(dt_limits_s))))
    return ceiling, rows


# ---------------------------------------------------------------------
# gauge: what a rate change does and does not do to energy
# ---------------------------------------------------------------------

def gauge_table(inertia_kg_m2: float = 0.5, omega_local_rad_s: float = 300.0):
    """Own-clock energy is invariant; reference-frame energy is not.

    A zone's laws are written against its own clock, so its own-clock angular
    rate does not change when its rate does. Seen from the reference frame the
    same rotor turns at ``tau * omega``, so the reference-frame kinetic energy
    scales as ``tau**2``. That number is a display quantity. Ledgering it would
    invent energy on every rate change; the conserved account is per zone, plus
    work exchanged across boundaries.
    """

    rows = []
    for tau in (0.25, 0.5, 1.0, 2.0, 4.0):
        own = 0.5 * inertia_kg_m2 * omega_local_rad_s ** 2
        reference = 0.5 * inertia_kg_m2 * (tau * omega_local_rad_s) ** 2
        rows.append((tau, own, reference, reference / own))
    return rows


# ---------------------------------------------------------------------
# a time change diffusing along a coupling chain
# ---------------------------------------------------------------------

def chain_response(
    *,
    radius: int = 8,
    joint_kind: str = "differential",
    diffusivity: float = 40.0,
    dt: float = 1.0e-3,
    steps: int = 4000,
    target_log_tau: float = -0.6931471805599453,   # tau = 0.5 at the source
    omega_ref_rad_s: float = 300.0,
    source_ramp_per_s: float = 4.0,
):
    """Drive node 0 toward a new rate and let the field diffuse down a chain.

    The field is a potential, so a steady gradient anywhere along the chain is
    free. Only while the profile is still moving does any joint supply torque.
    Returns one row per joint: its radius, peak reaction, the time that peak
    arrived, and the angular impulse it delivered over the whole transient.
    """

    nodes = [f"z{index}" for index in range(radius + 1)]
    field = TimeField.flat(nodes, TimeFieldConfig())
    peak = [0.0] * radius
    peak_at = [0.0] * radius
    impulse = [0.0] * radius
    unreached = float("inf")

    for step in range(steps):
        now = step * dt
        level = list(field.log_tau)
        # Interior nodes relax toward their neighbours: a discrete Laplacian
        # on the coupling chain. The source is rate-limited by its own ramp,
        # which is the physical limit the store imposes.
        wanted = list(level)
        wanted[0] = target_log_tau
        for index in range(1, radius + 1):
            left = level[index - 1]
            right = level[index + 1] if index < radius else level[index]
            wanted[index] = level[index] + diffusivity * dt * (
                left - 2.0 * level[index] + right
            )
        field.set_target(nodes[0], wanted[0], dt, source_ramp_per_s)
        for index in range(1, radius + 1):
            # The interior is not rate limited here: the Laplacian already
            # decides how fast it may move, and the reaction it costs is what
            # this measurement is for.
            field.set_target(nodes[index], wanted[index], dt, math.inf)

        for joint in range(radius):
            reaction = field.shear_reaction_nm(
                nodes[joint], nodes[joint + 1], joint_kind, omega_ref_rad_s,
            )
            if not math.isfinite(reaction):
                continue
            impulse[joint] += reaction * dt
            if reaction > peak[joint]:
                peak[joint] = reaction
                peak_at[joint] = now

    rows = []
    for joint in range(radius):
        arrival = peak_at[joint] if peak[joint] > 0.0 else unreached
        rows.append((joint + 1, peak[joint], arrival, impulse[joint]))
    return rows, field


def rigid_joint_in_the_path(joint_kind: str = "torque-shaft") -> float:
    """A joint with no rate freedom shears instead of carrying a gradient.

    This is why a diffusing region has to be defined on the sub-graph of
    joints that HAVE freedom. A rigid coupling is not an edge with very high
    stiffness; it is an identification, and the two sides are one zone.
    """

    field = TimeField.flat(["a", "b"], TimeFieldConfig())
    field.set_target("a", -0.2, 1.0e-3, math.inf)
    return field.shear_reaction_nm("a", "b", joint_kind, 300.0)


def main() -> None:
    # The rig's own figures, so these reactions are comparable with it.
    with_store_inertia("differential", 0.02)
    with_store_inertia("torque-converter", 0.05)

    print("=" * 78)
    print("Negative drift: a zone running FASTER than reference")
    print("=" * 78)
    print(f"{'case':>16} {'tau_a':>7} {'tau_b':>7} {'ledger error (J)':>20}")
    for label, tau_a, tau_b, error in negative_drift_ledger():
        print(f"{label:>16} {tau_a:>7} {tau_b:>7} {error:>20.4e}")
    print()
    ceiling, rows = negative_drift_cost()
    print(f"Cost is linear in tau. Derived ceiling tau_max = "
          f"K_max * dt_limit / window = {ceiling:.2f}")
    print(f"{'tau':>6} {'K':>6} {'step (us)':>11} {'window asked (ms)':>19} "
          f"{'window got (ms)':>17} {'stable':>7}")
    for tau, substeps, step, asked, got, stable in rows:
        print(f"{tau:>6} {substeps:>6} {step * 1e6:>11.2f} {asked * 1e3:>19.2f} "
              f"{got * 1e3:>17.2f} {str(stable):>7}")
    print()
    print("Past the ceiling the step never coarsens; the window shortens. Asking")
    print("to run faster than the budget affords returns LESS world time, which")
    print("is the same failure dilation already has: correct, but behind.")

    print()
    print("=" * 78)
    print("Energy under a rate change: what is invariant and what is display")
    print("=" * 78)
    print(f"{'tau':>6} {'own-clock KE (J)':>18} {'reference KE (J)':>18} {'ratio':>8}")
    for tau, own, reference, ratio in gauge_table():
        print(f"{tau:>6} {own:>18.4f} {reference:>18.4f} {ratio:>8.3f}")
    print()
    print("Own-clock energy does not move when the rate does, which is why a zone")
    print("cannot measure its own time velocity. Reference-frame energy scales as")
    print("tau**2 and is a rendering of the same state, not a second account.")

    print()
    print("=" * 78)
    print("A time change diffusing along a coupling chain, felt by radius")
    print("=" * 78)
    rows, field = chain_response()
    print(f"{'radius':>7} {'peak reaction (N.m)':>21} {'time to peak (ms)':>19} "
          f"{'impulse (N.m.s)':>17}")
    for radius, peak, arrival, impulse in rows:
        arrival_text = "never" if math.isinf(arrival) else f"{arrival * 1e3:.1f}"
        print(f"{radius:>7} {peak:>21.6f} {arrival_text:>19} {impulse:>17.6f}")
    print()
    print("Settled profile (a steady gradient, which costs nothing):")
    print("  " + "  ".join(
        f"{name}={math.exp(value):.3f}"
        for name, value in zip(field.nodes, field.log_tau)
    ))
    field.settle()
    steady = max(
        field.shear_reaction_nm(field.nodes[i], field.nodes[i + 1],
                                "differential", 300.0)
        for i in range(len(field.nodes) - 1)
    )
    print(f"  reaction once settled: {steady:.6f} N.m")

    print()
    print("=" * 78)
    print("A joint with no rate freedom in the path")
    print("=" * 78)
    print(f"  torque shaft asked to carry a changing gradient: "
          f"{rigid_joint_in_the_path()}")
    print("  So a diffusing region is the sub-graph of joints that have freedom.")
    print("  A rigid coupling is an identification, not a stiff edge.")


if __name__ == "__main__":
    main()
