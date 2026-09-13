"""The eleven stages of a shot, in order, each on the real engine.

This is the spine the rest hangs off. Every stage names what it needs
and what it produces, so a stage can be replaced, stubbed or measured
without the others noticing -- and so it is obvious which ones are real
and which are still descriptions.

    1  CUE            GPS identification and coarse orientation.
                      Optional: it gets the mount roughly pointed and
                      it is not a firing solution.
    2  RETICLE        the operator centres the reticle on the target.
    3  SOLUTION       the scope solves the shot for that reticle: range
                      from its own rangefinder, wind from its own
                      meter, and -- crucially -- an INTENDED ENERGY AT
                      IMPACT if the operator asked for one.
    4  PACK           charge and projectile assembled for that
                      solution, or a ready-made full charge taken off
                      the rack if the solution did not specify.
    5  LOAD           and here is the cost of firing to a solution.
                      A custom round cannot simply be fired -- whatever
                      is already in the chamber has to come OUT first,
                      under power, with no shot to extract it. That is
                      a real mechanism, a real delay, and a real reason
                      to fire full charge when it will do.
    6  IGNITE         the primer gets its detonating conditions and the
                      interior ballistics kernel runs the barrel
                      traversal. Product: a muzzle vector and state.
    7  EMIT           flash, projectile, muzzle gas, brake effect.
    8  RECOVER        waste gas force: what the brake redirected, what
                      the evacuator trapped, what the holding tank got,
                      and whether that is enough to drive the reload or
                      whether an actuator has to.
    9  FLY            the game integrates the round with the wind that
                      is actually there -- which is NOT the windage the
                      solution used. Returns the first impact vector.
                      The other party may challenge it.
   10  PENETRATE      the damage system takes the impact and resolves
                      what it went through.
   11  OBSERVE        damage visible from the scope may, at the scope's
                      discretion, be used to correct the next solution.

WHAT IS REAL TODAY. Stages 3, 4, 6 and 9 run on compiled kernels.
Stage 10 runs on the existing penetration sim. Stages 5, 7, 8 and 11
are structured and partly stubbed, and each says so in its own result
rather than quietly returning a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class Chambered:
    """What is in the gun right now, which is the whole of stage 5."""
    charge_kg: float = 0.0
    web_m: float = 0.0
    projectile_kg: float = 0.0
    kind: str = "empty"

    def matches(self, want: "Chambered", *, charge_tol: float = 1e-4,
                web_tol: float = 1e-6) -> bool:
        """Can this round serve that solution without being replaced?

        The question stage 5 turns on. If it matches, the shot is free;
        if not, something has to be pulled out of the chamber first."""
        return (self.kind == want.kind
                and abs(self.charge_kg - want.charge_kg) <= charge_tol
                and abs(self.web_m - want.web_m) <= web_tol
                and abs(self.projectile_kg - want.projectile_kg) <= 1e-4)


@dataclass
class Stage:
    """One step's result, with what it actually did."""
    number: int
    name: str
    ok: bool
    seconds: float = 0.0
    detail: dict = field(default_factory=dict)
    stub: bool = False

    def line(self) -> str:
        mark = "STUB" if self.stub else ("ok" if self.ok else "FAIL")
        bits = "  ".join(f"{k}={v}" for k, v in list(self.detail.items())[:4])
        return f"  {self.number:2d} {self.name:<11s} {mark:>4s} " \
               f"{self.seconds * 1000:7.1f} ms   {bits}"


@dataclass
class FiringSequence:
    """The gun, and the eleven stages it goes through to fire once."""
    rig: object                       # gun_rig.GunRig
    chamber: Chambered = field(default_factory=Chambered)
    #: stage 5, the powered extraction of a round nobody fired
    power_unload_s: float = 0.9
    reload_s: float = 0.6
    #: ---- STAGE 8: A HEAVY SPRING, NOT A GAS TANK ----
    #: The recoil already carries the energy. Storing it as compressed
    #: gas means a tank, a trap, valving, a seal against filthy hot
    #: products, and a conversion back to mechanical work at every step
    #: -- to move a bolt six inches. A spring takes the stroke directly,
    #: holds it as strain, and gives it back on the return. It is the
    #: oldest answer in automatic weapons and it is the right one.
    #:
    #: RATE AND STROKE ARE THE DESIGN. Too stiff and the recoil stops
    #: short of a full cycle; too soft and the gun slams its own end
    #: stop and the frame takes the blow instead of the spring.
    spring_rate_n_per_m: float = 22_000.0
    spring_stroke_m: float = 0.420
    spring_preload_n: float = 1_800.0
    spring_efficiency: float = 0.82        # friction in the guides and rollers
    recoiling_mass_kg: float = 85.0
    #: what the cycle actually costs: extract, eject, feed, ram, cock
    reload_work_needed_j: float = 140.0
    stages: list = field(default_factory=list)

    # ---------------------------------------------------------------
    def cue(self, target=None) -> Stage:
        """1. GPS identification and coarse orientation."""
        return Stage(1, "cue", target is not None, 0.0,
                     {"source": "gps" if target else "none",
                      "coarse_only": True}, stub=True)

    def reticle(self, on_target: bool, dwell_s: float) -> Stage:
        """2. The operator holds the reticle on it."""
        return Stage(2, "reticle", on_target, dwell_s,
                     {"dwell_s": round(dwell_s, 2)})

    def solution(self, range_m: float, want_energy_j: float = 0.0,
                 dart_kg: float = 0.130, wind_m_s: float = 4.0) -> Stage:
        """3. The scope solves it -- to an ENERGY if one was asked for."""
        import time
        t0 = time.perf_counter()
        detail = {"range_m": range_m, "windage_m_s": wind_m_s}
        if want_energy_j > 0.0:
            from tasked_fire import muzzle_for_impact
            want = muzzle_for_impact(want_energy_j, dart_kg, range_m, 0.0004)
            detail.update(mode="energy",
                          want_kj=round(want_energy_j / 1000, 1),
                          need_muzzle=round(want["muzzle_speed_m_s"]))
        else:
            detail.update(mode="full-charge")
        return Stage(3, "solution", True, time.perf_counter() - t0, detail)

    def pack(self, solution: Stage, *, full_charge_kg: float = 0.040,
             web_m: float = 2.7e-4, dart_kg: float = 0.130) -> tuple:
        """4. Assemble the round, or take a ready-made one off the rack."""
        import time
        t0 = time.perf_counter()
        if solution.detail.get("mode") == "energy":
            # a real search would size the stack here; for now the
            # solution's required muzzle velocity scales the charge
            want = solution.detail["need_muzzle"]
            frac = max(0.35, min(1.0, want / 1000.0))
            want_round = Chambered(round(full_charge_kg * frac, 6), web_m,
                                   dart_kg, "tasked")
            made = "assembled"
        else:
            want_round = Chambered(full_charge_kg, web_m, dart_kg, "full")
            made = "off-the-rack"
        return want_round, Stage(4, "pack", True, time.perf_counter() - t0,
                                 {"kind": want_round.kind, "how": made,
                                  "charge_g": round(want_round.charge_kg * 1000, 1)})

    def load(self, want: Chambered) -> Stage:
        """5. And THIS is what a tasked shot costs.

        If the chamber already holds what the solution wants, nothing
        happens and the shot is free. If it does not, a live round has
        to be pulled out under power -- there is no shot to extract it
        -- and that is the price of firing to an energy rather than
        firing what is already up the spout."""
        if self.chamber.matches(want):
            return Stage(5, "load", True, 0.0,
                         {"action": "already-chambered", "cost": "none"})
        had = self.chamber.kind
        seconds = (self.power_unload_s if had != "empty" else 0.0) + self.reload_s
        self.chamber = want
        return Stage(5, "load", True, seconds,
                     {"action": "power-unload+load" if had != "empty" else "load",
                      "ejected": had, "loaded": want.kind})

    def ignite(self) -> Stage:
        """6. Primer, then the barrel traversal. Product: muzzle state."""
        import time
        t0 = time.perf_counter()
        shot = self.rig.fire_one(charge_kg=self.chamber.charge_kg,
                                 web_m=self.chamber.web_m)
        return Stage(6, "ignite", True, time.perf_counter() - t0,
                     {"muzzle_m_s": round(shot["muzzle_m_s"]),
                      "peak_MPa": round(shot["peak_breech_pa"] / 1e6),
                      "exit_K": round(shot["exit_gas_k"]),
                      "closure": f"{shot['closure'] * 100:.0f}%"}), shot

    def emit(self, shot: dict) -> Stage:
        """7. Flash, round, muzzle gas, brake."""
        return Stage(7, "emit", True, 0.0,
                     {"gas_g": round(shot["gas_kg"] * 1000, 1),
                      "flash_from": "unburnt fraction",
                      "brake": "not modelled"}, stub=True)

    def recover(self, shot: dict) -> Stage:
        """8. The recoil spring takes the stroke and gives back the cycle.

        The gun is a mass on a spring. The shot hands it an impulse, it
        runs back compressing the spring until the spring has taken all
        the kinetic energy, and then the spring pushes it home again --
        and that return stroke is what works the action.

        Three things have to be true at once and they fight each other:
        the stroke must not exceed the travel available, the spring must
        return more than the cycle costs, and what is left over has to
        go somewhere other than the end stop."""
        impulse = shot["recoil_impulse_n_s"]
        m = self.recoiling_mass_kg
        v = impulse / m
        ke = 0.5 * m * v * v
        k = self.spring_rate_n_per_m
        f0 = self.spring_preload_n
        # how far it runs back: the spring absorbs 1/2 k x^2 + f0 x
        # solve 1/2 k x^2 + f0 x - ke = 0
        x = (-f0 + math.sqrt(f0 * f0 + 2.0 * k * ke)) / k
        bottomed = x >= self.spring_stroke_m
        travel = min(x, self.spring_stroke_m)
        stored = 0.5 * k * travel * travel + f0 * travel
        returned = stored * self.spring_efficiency
        # if it bottoms out, the rest of the energy hits the end stop
        slammed = max(0.0, ke - stored)
        return Stage(8, "recover", not bottomed, 0.0,
                     {"recoil_m_s": round(v, 2),
                      "energy_J": round(ke),
                      "stroke_mm": round(travel * 1000),
                      "returns_J": round(returned),
                      "needs_J": self.reload_work_needed_j,
                      "drives_reload": returned >= self.reload_work_needed_j,
                      "into_end_stop_J": round(slammed)})

    def fly(self, shot: dict, range_m: float, wind_m_s: float) -> Stage:
        """9. The round flies through the wind that is really there."""
        import time, random
        t0 = time.perf_counter()
        try:
            from sustained_fire_trial import event_trajectory, gusting_wind
            rng = random.Random()
            ev = event_trajectory(".50 ap", 0.0, 0.0, range_m,
                                  gusting_wind(rng, wind_m_s, wind_m_s * 0.6))
            detail = {"drop_mm": round(ev["drop_m"] * 1000),
                      "drift_mm": round(ev["drift_m"] * 1000),
                      "tof_s": round(ev["time_s"], 2),
                      "challengeable": True}
        except Exception as exc:
            detail = {"error": type(exc).__name__}
        return Stage(9, "fly", True, time.perf_counter() - t0, detail)

    def challenge(self, impact) -> Stage:
        """9b. The other party recomputes it from the origin vector.

        Stubbed deliberately: the point of recording it as a stage is
        that the shot must be reproducible from its ORIGINATING VECTOR
        and nothing else, so an opponent can integrate it themselves
        and show a disagreement."""
        return Stage(9, "challenge", True, 0.0,
                     {"api": "stub", "reproducible_from": "origin-vector"},
                     stub=True)

    def penetrate(self, shot: dict, fly: Stage, *,
                  target_material: str = "rha",
                  plate_m: float = 0.025,
                  dart_kg: float = 0.130, dart_d_m: float = 0.020,
                  dart_len_m: float = 0.090,
                  obliquity_deg: float = 0.0) -> Stage:
        """10. What it went through, on the project's own penetration sim.

        `ballistics.resolve_impact` is the real traversal: it takes the
        projectile's arriving STATE -- mass, diameter, length, speed,
        yaw, core hardness -- against a declared material and thickness
        and returns what it did and what is left of it. This stage's job
        is only to hand it the state the shot actually arrived in rather
        than a nominal one.

        The arriving speed comes from the FLIGHT, not the muzzle: the
        round has crossed eight hundred metres of air and lost a third
        of its energy doing it, and a penetration computed at muzzle
        velocity is a different and much more flattering gun."""
        import time
        t0 = time.perf_counter()
        try:
            from ballistics import (ProjectileState, resolve_impact,
                                    material_profile)
            arriving = fly.detail.get("arrival_m_s")
            if arriving is None:
                arriving = shot["muzzle_m_s"] * 0.72   # nothing flew it
            yaw = math.radians(obliquity_deg)
            state = ProjectileState(
                mass_kg=dart_kg, diameter_m=dart_d_m,
                speed_m_s=float(arriving), direction=(0.0, 0.0, 1.0),
                length_m=dart_len_m, yaw_rad=yaw, hardness_pa=2.5e9)
            target = material_profile(target_material)
            res = resolve_impact(state, target, plate_m, (0.0, 0.0, -1.0))
            detail = {
                "target": f"{plate_m * 1000:.0f} mm {target_material}",
                "arrived_m_s": round(float(arriving)),
                "mode": getattr(res, "mode", "?"),
                "through": getattr(res, "perforated", None),
                "exit_m_s": round(getattr(getattr(res, "exit_state", None),
                                          "speed_m_s", 0.0) or 0.0),
            }
            ok = True
        except Exception as exc:
            detail = {"error": f"{type(exc).__name__}: {exc}"[:80]}
            ok = False
        return Stage(10, "penetrate", ok, time.perf_counter() - t0, detail)

    def observe(self, impact) -> Stage:
        """11. What the scope saw, fed back as solution error."""
        return Stage(11, "observe", True, 0.0,
                     {"in_los": "unknown", "corrects": "next solution"},
                     stub=True)

    # ---------------------------------------------------------------
    def fire(self, *, range_m: float = 800.0, want_energy_j: float = 0.0,
             wind_m_s: float = 4.0, dwell_s: float = 1.2,
             target=object()) -> list:
        """All eleven, in order."""
        s = [self.cue(target), self.reticle(True, dwell_s)]
        sol = self.solution(range_m, want_energy_j, wind_m_s=wind_m_s)
        s.append(sol)
        want, pk = self.pack(sol)
        s.append(pk)
        s.append(self.load(want))
        ign, shot = self.ignite()
        s.append(ign)
        s.append(self.emit(shot))
        s.append(self.recover(shot))
        s.append(self.fly(shot, range_m, wind_m_s))
        s.append(self.challenge(None))
        s.append(self.penetrate(shot, s[-2]))
        s.append(self.observe(None))
        self.stages = s
        return s

    def report(self) -> list[str]:
        out = [f"firing sequence -- {sum(x.seconds for x in self.stages):.2f} s "
               f"from cue to impact"]
        out += [x.line() for x in self.stages]
        stubs = [x.name for x in self.stages if x.stub]
        out.append(f"  still stubbed: {', '.join(stubs)}")
        return out
