"""CONTRACT OPPORTUNITIES: the time store as the signal that help is worth buying.

The time force store already exists for a physical reason -- something
has to absorb the reaction when the field changes, and whatever we chose
sets the ramp limit. This module says what ELSE that quantity is good
for: it is the honest, already-measured signal that a system is being
asked for more time than it can cover, and therefore the trigger for a
contract that a server or a power user may offer to take.

WHY THE STORE IS THE RIGHT SIGNAL

A store that keeps charging means the field keeps being RAMPED, which
means the allocator keeps having to take time away and give it back.
That is chronic under-allocation stated in joules, by the machine, with
no separate "I am struggling" telemetry channel to invent, keep honest,
or let a mod lie about. Same argument as implicit profiling: the cost is
measured, not declared.

LEADING AND LAGGING, BOTH

The store is a LAGGING confirmation -- by the time it is charged the
system is already behind. The allocation shortfall (what it asked for
versus what it got) is the LEADING edge. An opportunity needs both, or
it churns: every transient raises a contract, every contract is stale
before it is accepted.

WHAT MAKES AN OFFER ACCEPTABLE

"A chance of being accepted" is the honest framing: a helper computes
PREDICTIVELY, ahead of the owner, and the owner takes the result only if
it is still valid when it arrives. Two things make that checkable rather
than a matter of trust:

  1. DETERMINISM. The sim is seeded per identity, so the same state and
     the same inputs produce bit-identical output. Acceptance is then a
     digest comparison rather than an act of faith. This is what the
     seeding work bought, and it is why offloading is possible at all --
     against a 2%-noisy sim, no prediction could ever be validated.

  2. CONSERVATION. A returned result can be rejected WITHOUT redoing the
     work, by checking the invariants the law already guarantees: the
     energy ledger closes, mass balances, nothing crossed a joint that
     the joint could not carry. A helper cannot profit by cheating the
     physics for the same reason a mod cannot -- the conservative law is
     the fraud check, exactly as it is the sandbox.

A NICE INVERSION: a lane running slow is both the one that most needs
help AND the one easiest to help, because it advances less world time
per frame, so a helper gets more runway ahead of it for the same work.
The systems in trouble are the ones prediction is cheapest for.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


# ``StoreLedger`` -- what the time force store is holding -- moved with the
# time field into ``turing/src/common/dt_system/time_field.py`` (2026-09-22).
# One definition; imported here so this module's callers are unchanged.
from time_field import StoreLedger  # noqa: E402,F401


@dataclass(frozen=True)
class ContractOpportunity:
    """A zone offering to be helped, and on what terms.

    `topology` is what a helper must be able to run -- the same
    signature the ABI negotiator buckets by, so a helper can say up front
    whether it has the assembly at all.
    """

    zone: str
    topology: str
    lanes: tuple[str, ...]
    shortfall_frac: float
    stored_j: float
    persistence: int
    runway_s: float
    state_digest: str

    @property
    def worth_offering(self) -> bool:
        """Both signals, not either. The shortfall says it is happening
        now; the persistence says it is not a transient."""
        return self.shortfall_frac > 0.05 and self.persistence >= 3

    def receipt(self) -> dict:
        return {
            "zone": self.zone, "topology": self.topology,
            "lanes": list(self.lanes),
            "shortfall": round(self.shortfall_frac, 4),
            "stored_j": round(self.stored_j, 6),
            "persistence": self.persistence,
            "runway_s": round(self.runway_s, 5),
            "digest": self.state_digest[:12],
            "offer": self.worth_offering,
        }


def state_digest(values: Sequence[float]) -> str:
    """Identify a state span exactly. Determinism is what makes this
    meaningful: the same span, stepped with the same inputs, must produce
    the same digest on any machine."""
    h = hashlib.sha256()
    for v in values:
        h.update(repr(float(v)).encode())
    return h.hexdigest()


def opportunity_for(zone: str, *, topology: str, lanes: Iterable[str],
                    ledger: StoreLedger, time_velocity: float,
                    reference_dt_s: float, predict_frames: int,
                    state: Sequence[float]) -> ContractOpportunity:
    """Raise the opportunity a zone currently represents.

    `runway_s` is how much WORLD time a helper would cover by predicting
    `predict_frames` ahead. A dilated zone advances less per frame, so a
    helper gets further ahead of it for the same work -- the inversion in
    the module docstring, computed rather than asserted.
    """
    runway = float(reference_dt_s) * float(time_velocity) * int(predict_frames)
    return ContractOpportunity(
        zone=zone, topology=topology, lanes=tuple(lanes),
        shortfall_frac=ledger.shortfall_ema,
        stored_j=ledger.stored_j,
        persistence=ledger.persistence,
        runway_s=runway,
        state_digest=state_digest(state),
    )


# ---------------------------------------------------------------------
# accepting what comes back
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class PredictedResult:
    """What a helper returns: where it started, what it did, what it got."""

    zone: str
    from_digest: str
    advanced_s: float
    state: tuple[float, ...]
    #: the helper's own claim about the energy ledger it closed
    energy_in_j: float
    energy_out_j: float
    #: For every cross-rate joint it stepped: the FRAME-CONVERTED rates it
    #: actually used, as {joint: (side_name, omega_in_joint_frame)}.
    #: Required because conservation does NOT catch a resampling mistake --
    #: measured in time_rig: two LSD rigs, one resampling correctly and one
    #: not, both closed their energy ledgers to machine epsilon while
    #: reaching materially different speeds. An energy gate accepts both.
    #: A helper running a different time zone is exactly the party most
    #: likely to make that mistake, so it must show its work.
    declared_rates: tuple[tuple[str, str, float], ...] = ()


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: str


def validate(result: PredictedResult, *, expected_from_digest: str,
             expected_advanced_s: float, energy_tolerance_j: float = 1e-9,
             advanced_tolerance_s: float = 1e-12,
             expected_rates: Mapping[tuple[str, str], float] | None = None,
             rate_tolerance: float = 1e-9) -> Verdict:
    """Accept or reject a predicted update WITHOUT redoing the work.

    Three gates, cheapest first:
      * it must have started from the state we actually had -- otherwise
        the prediction is about a world that did not happen
      * it must have advanced the time we actually needed
      * its energy ledger must close. This is the FRAUD gate: a
        dishonest or broken helper cannot pass it by construction, and
        it costs a subtraction rather than a simulation.
      * every cross-rate joint must have used the frame conversion our
        own field says it should have. This is the CORRECTNESS gate, and
        it is separate on purpose: energy conservation is blind to a
        resampling error, because the bookkeeping stays self-consistent
        for whatever rate the helper thought it saw. Two different
        questions, two different checks.
    """
    if result.from_digest != expected_from_digest:
        return Verdict(False, "diverged: predicted from a state we no longer had")
    if abs(result.advanced_s - expected_advanced_s) > advanced_tolerance_s:
        return Verdict(False,
                       f"advanced {result.advanced_s:.6g}s, needed "
                       f"{expected_advanced_s:.6g}s")
    drift = abs(result.energy_in_j - result.energy_out_j)
    if drift > energy_tolerance_j:
        return Verdict(False, f"energy ledger open by {drift:.3e} J")
    if expected_rates:
        for joint, side, used in result.declared_rates:
            want = expected_rates.get((joint, side))
            if want is None:
                return Verdict(False, f"declared a rate for unknown joint {joint}/{side}")
            if abs(used - want) > rate_tolerance:
                return Verdict(
                    False,
                    f"{joint}/{side} stepped at {used:.6g} rad/s in the joint frame, "
                    f"our field says {want:.6g} -- resampled against a different clock")
        missing = set(expected_rates) - {(j, s) for j, s, _ in result.declared_rates}
        if missing:
            joint, side = sorted(missing)[0]
            return Verdict(False, f"no rate declared for cross-rate joint {joint}/{side}")
    return Verdict(True, "accepted")


if __name__ == "__main__":
    # a zone under sustained time pressure raises an offer; a zone that
    # merely twitched once does not
    import math

    pressured = StoreLedger()
    twitchy = StoreLedger()
    for frame in range(8):
        pressured.observe(reaction_nm=0.12, slip_rad_s=40.0, dt_s=1 / 60.0,
                          asked_s=1 / 60.0, got_s=1 / 90.0)
        twitchy.observe(reaction_nm=0.12 if frame == 0 else 0.0,
                        slip_rad_s=40.0, dt_s=1 / 60.0,
                        asked_s=1 / 60.0, got_s=1 / 60.0)
        twitchy.relax(1 / 60.0)

    span = [1.0, 2.0, 3.0, 4.0]
    for name, ledger, tau in (("far-bay", pressured, 0.5), ("near-bay", twitchy, 1.0)):
        opp = opportunity_for(name, topology="cyl6.banks2.4stroke.recip.cam2.circ6.node325",
                              lanes=(name,), ledger=ledger, time_velocity=tau,
                              reference_dt_s=1 / 60.0, predict_frames=6, state=span)
        r = opp.receipt()
        print(f"{r['zone']:<10} shortfall {r['shortfall']:<7} stored {r['stored_j']:<10} "
              f"persist {r['persistence']:<3} runway {r['runway_s']}s  -> offer={r['offer']}")

    print()
    good = PredictedResult("far-bay", state_digest(span), 0.05,
                           (1.0, 2.0, 3.0, 4.5), 10.0, 10.0)
    print("honest helper :", validate(good, expected_from_digest=state_digest(span),
                                      expected_advanced_s=0.05))
    stale = PredictedResult("far-bay", state_digest([9.0]), 0.05,
                            (1.0,), 10.0, 10.0)
    print("stale helper  :", validate(stale, expected_from_digest=state_digest(span),
                                      expected_advanced_s=0.05))
    cheat = PredictedResult("far-bay", state_digest(span), 0.05,
                            (1.0, 2.0, 3.0, 99.0), 10.0, 250.0)
    print("cheating helper:", validate(cheat, expected_from_digest=state_digest(span),
                                       expected_advanced_s=0.05))
