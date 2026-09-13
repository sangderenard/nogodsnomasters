"""A weapon station: engine, turret on a pole, and the logic that keeps
the whole thing alive without anyone standing next to it.

The turret does not work on its own, and that is the interesting part.
It needs electricity for its own controls -- the fire-control computer,
the sight, the valve solenoids -- and it needs hydraulic pressure to
move. Those come from an engine that is not running most of the time,
because an engine that idles all day to power a computer is an engine
that has burned its fuel by morning.

So the station is built the way a real unattended installation is:

  THE TURRET CARRIES ITS OWN BATTERY. Enough to run the electronics for
  hours and to crank the engine when it is needed. This is what lets
  the station sit dark and still be awake -- the sight watches, the
  computer solves, and nothing is running.

  THE ENGINE AUTO-STARTS WHEN SOMETHING NEEDS IT. Not on a timer: on
  the state of the stores. Battery low, hydraulic accumulator low, air
  low, or a target acquired that will need the mount to move. It runs
  until everything it was started for is satisfied, plus a cooldown, so
  it is not cycling every ninety seconds.

  IT KNOWS ITS OWN PARTS. A turret bolted to a pole beside its own
  engine can traverse across both of them, and a fire-control system
  that does not know that will eventually put a round through its own
  radiator. Friendly fire here is not a rule bolted on afterwards -- it
  is the same ray the shot uses, tested against the station's own
  structure, and it blocks the trigger rather than warning about it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# =====================================================================
#  THE TURRET'S OWN POWER
# =====================================================================
@dataclass
class EmergencyBattery:
    """The battery that lets the station be awake while it is dark.

    Sized the way these actually are: enough amp-hours to hold the
    electronics up overnight AND to turn a big diesel over several
    times, because a station that cannot start its own engine is a
    station that is finished the moment its accumulator runs down."""
    nominal_v: float = 24.0
    capacity_ah: float = 55.0
    state_of_charge: float = 1.0
    internal_ohms: float = 0.012
    # a lead-acid bank that goes below about a quarter is being damaged,
    # and below a tenth will not crank
    reserve_floor: float = 0.25
    crank_floor: float = 0.10

    @property
    def energy_wh(self) -> float:
        return self.nominal_v * self.capacity_ah * self.state_of_charge

    @property
    def can_crank(self) -> bool:
        return self.state_of_charge > self.crank_floor

    @property
    def terminal_v(self) -> float:
        """Voltage sags as it empties, and sags further under load."""
        return self.nominal_v * (0.86 + 0.14 * self.state_of_charge)

    def draw(self, watts: float, dt: float) -> float:
        """Take power out, and say how much was actually available."""
        if watts <= 0.0 or dt <= 0.0:
            return 0.0
        available_wh = self.nominal_v * self.capacity_ah * self.state_of_charge
        wanted_wh = watts * dt / 3600.0
        taken = min(wanted_wh, available_wh)
        self.state_of_charge = max(0.0, self.state_of_charge
                                   - taken / max(self.nominal_v * self.capacity_ah, 1e-9))
        return taken * 3600.0 / max(dt, 1e-9)

    def charge(self, watts: float, dt: float) -> None:
        if watts <= 0.0 or dt <= 0.0:
            return
        # charge acceptance falls off as it fills, which is why the last
        # tenth takes as long as the first half
        acceptance = max(0.15, 1.0 - self.state_of_charge ** 3)
        gained_wh = watts * acceptance * dt / 3600.0
        self.state_of_charge = min(1.0, self.state_of_charge
                                   + gained_wh / max(self.nominal_v * self.capacity_ah, 1e-9))

    def describe(self) -> list[str]:
        return [f"  battery {self.nominal_v:.0f} V {self.capacity_ah:.0f} Ah  "
                f"{self.state_of_charge * 100:5.1f} %  {self.terminal_v:5.2f} V  "
                f"{self.energy_wh / 1000:5.2f} kWh"
                + ("" if self.can_crank else "   TOO FLAT TO CRANK")]


# What the turret's own electronics cost, continuously and in bursts.
LOAD_FIRE_CONTROL_W = 45.0        # the computer, awake and solving
LOAD_SIGHT_W = 14.0               # the sight and its sensor
LOAD_STANDBY_W = 6.0              # everything else, dark
LOAD_VALVES_W = 38.0              # solenoids, only while the mount moves
STARTER_W = 6_400.0               # cranking a big diesel
STARTER_SECONDS = 2.4


# =====================================================================
#  WHEN TO START THE ENGINE
# =====================================================================
@dataclass
class AutoStart:
    """Start on the state of the stores, not on a clock.

    Each reason is a real one with its own threshold, and the engine
    runs until every reason that started it is satisfied. The cooldown
    is what stops a station short-cycling its own diesel, which is hard
    on it and audible for miles."""
    battery_start_below: float = 0.45
    battery_stop_above: float = 0.85
    accumulator_start_below: float = 0.35
    accumulator_stop_above: float = 0.90
    air_start_below: float = 0.55
    air_stop_above: float = 0.92
    cooldown_s: float = 45.0
    warmup_s: float = 8.0
    # live
    running: bool = False
    reasons: list = field(default_factory=list)
    since_satisfied_s: float = 0.0
    cranking_s: float = 0.0

    def assess(self, *, battery: float, accumulator: float, air: float,
               engagement: bool) -> list[str]:
        """Why the engine should be running right now."""
        reasons = []
        if battery < self.battery_start_below:
            reasons.append(f"battery {battery * 100:.0f}%")
        if accumulator < self.accumulator_start_below:
            reasons.append(f"accumulator {accumulator * 100:.0f}%")
        if air < self.air_start_below:
            reasons.append(f"air {air * 100:.0f}%")
        if engagement:
            # the one that is not about a store: the mount is about to
            # need flow, and waiting for the accumulator to sag first
            # means slewing at a crawl through the part that matters
            reasons.append("engagement")
        return reasons

    def satisfied(self, *, battery: float, accumulator: float, air: float,
                  engagement: bool) -> bool:
        return (battery >= self.battery_stop_above
                and accumulator >= self.accumulator_stop_above
                and air >= self.air_stop_above
                and not engagement)

    def step(self, dt: float, *, battery: float, accumulator: float, air: float,
             engagement: bool) -> dict:
        self.reasons = self.assess(battery=battery, accumulator=accumulator,
                                   air=air, engagement=engagement)
        want = bool(self.reasons)
        event = None
        if not self.running:
            if want:
                self.running = True
                self.cranking_s = STARTER_SECONDS
                self.since_satisfied_s = 0.0
                event = "start"
        else:
            self.cranking_s = max(0.0, self.cranking_s - dt)
            if self.satisfied(battery=battery, accumulator=accumulator,
                              air=air, engagement=engagement):
                self.since_satisfied_s += dt
                if self.since_satisfied_s >= self.cooldown_s:
                    self.running = False
                    event = "stop"
            else:
                self.since_satisfied_s = 0.0
        return {"running": self.running, "reasons": list(self.reasons),
                "cranking": self.cranking_s > 0.0, "event": event}


# =====================================================================
#  KNOWING YOUR OWN PARTS
# =====================================================================
@dataclass
class FriendlyMask:
    """What the station must not shoot.

    The check is the same ray the shot takes, against the station's own
    structure. That matters: a rule written as "do not fire within
    thirty degrees of the engine" is a guess about geometry, and this
    is the geometry."""
    own_mesh: object = None            # RayMesh over the station's own parts
    muzzle_clear_m: float = 1.2        # past this, a hit is the mount's own body
    blocked_by: str | None = None

    def clear_to_fire(self, muzzle, direction, reach_m: float = 60.0) -> bool:
        """Would this shot hit us on the way out?"""
        self.blocked_by = None
        if self.own_mesh is None:
            return True
        from engine_rays import Ray
        muzzle = np.asarray(muzzle, dtype=float)
        direction = np.asarray(direction, dtype=float)
        direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
        hit = self.own_mesh.pick(Ray.from_points(muzzle, muzzle + direction * reach_m))
        if hit is None:
            return True
        distance = float(np.linalg.norm(np.asarray(hit.point, dtype=float) - muzzle))
        if distance <= self.muzzle_clear_m:
            return True                # the barrel's own shroud, not an obstacle
        self.blocked_by = str(hit.part)
        return False


# =====================================================================
#  THE STATION
# =====================================================================
@dataclass
class WeaponStation:
    """Engine, turret, power and the rules that run them unattended."""
    identity: str = "station"
    engine_sim: object = None
    fire_control: object = None
    turret_sim: object = None       # machines.MachineSim, for the recoil slide
    battery: EmergencyBattery = field(default_factory=EmergencyBattery)
    autostart: AutoStart = field(default_factory=AutoStart)
    friendly: FriendlyMask = field(default_factory=FriendlyMask)
    pole_height_m: float = 2.6
    # stores, as fractions
    accumulator_charge: float = 0.85
    air_charge: float = 1.0
    # live
    engine_running: bool = False
    electrical_load_w: float = 0.0
    log: list = field(default_factory=list)
    rounds_fired: int = 0
    #: rounds currently in the air (projectiles.ProjectileField)
    projectiles: object = None
    #: () -> (muzzle position, bore direction), read off the POSED gun.
    #: Supplied by whatever owns the geometry; without it the
    #: fire-control angles are used, which is close but is not the tube.
    muzzle_pose: object = None
    engagements: list = field(default_factory=list)
    #: ---- THE TUBE'S OWN BENDING, live ----
    #: A barrel is a beam: every shot rings it, and at any real rate of
    #: fire it never settles between rounds. So the muzzle is pointing
    #: somewhere slightly different at the instant each round leaves,
    #: and that is not an aiming error anyone can correct -- nothing
    #: commanded it. Stepped every tick by `beam_theory`'s own baked
    #: kernel and handed to the pose, so the drawn tube, the shot tube
    #: and the targeting tube are the same bent tube.
    whip_q_m: float = 0.0
    whip_qdot_m_s: float = 0.0
    whip_rad: float = 0.0
    barrel_temperature_k: float = 293.15
    _beam: object = None

    def note(self, line: str) -> None:
        self.log.append(line)
        del self.log[:-12]

    # ---------------- the tube ----------------
    def soak_barrel(self, joules: float, coolant_k: float = None,
                    ua_w_per_k: float = 680.0, dt: float = 0.0) -> float:
        """Heat in from the shot, heat out to the jacket.

        The barrel's temperature is not a setting -- it is the running
        balance of what the rounds put in and what the coolant takes
        out, and it feeds straight back into the tube's stiffness. A hot
        barrel rings slower, so the phase each round leaves at changes
        as the gun heats. That coupling is the reason this is on the
        station and not in a table."""
        nodes = {n["identity"]: n for n in self.graph["nodes"]}
        tube = nodes.get("turret.outer_barrel") or nodes.get("turret.weapon")
        mass = float(tube.get("mass_kg", 60.0)) if tube else 60.0
        self.barrel_temperature_k += joules / max(mass * 470.0, 1.0)
        if coolant_k is not None and dt > 0.0:
            w = ua_w_per_k * (self.barrel_temperature_k - coolant_k)
            self.barrel_temperature_k -= w * dt / max(mass * 470.0, 1.0)
        return self.barrel_temperature_k

    def _beam_kw(self) -> dict:
        """The tube's own section, read off the graph rather than typed.

        The barrel node already declares what it is made of and how big
        it is; asking it is the difference between one description of
        the tube and two that can disagree."""
        nodes = {n["identity"]: n for n in self.graph["nodes"]}
        tube = nodes.get("turret.outer_barrel") or nodes.get("turret.weapon")
        if tube is None:
            return None
        ro = float(tube.get("tube_outer_radius_m",
                            tube.get("drum_radius_m", 0.082)))
        ri = float(tube.get("tube_inner_radius_m", ro - 0.022))
        # whatever the jacket is holding rides with the tube
        jacket = sum(float(n.get("fluid_volume_l", 0.0))
                     for n in self.graph["nodes"]
                     if n.get("part_role") == "barrel-jacket-chamber")
        length = float(tube.get("drum_length_m", 1.70))
        return dict(length_m=length, outer_radius_m=ro, wall_m=max(ro - ri, 0.004),
                    density_kg_m3=7850.0, youngs_cold_pa=2.05e11,
                    damping_ratio=0.005,
                    added_mass_per_m_kg=(jacket / 1000.0 * 1000.0) / max(length, 1e-6),
                    temperature_k=self.barrel_temperature_k)

    def step_tube(self, dt: float, impulse_n_s: float = 0.0,
                  in_tube_s: float = 0.0027) -> float:
        """Ring the tube for this tick, and hit it if a round went off."""
        if self._beam is None:
            from beam_theory import beam_dynamics_native
            self._beam = beam_dynamics_native()[0]
        kw = self._beam_kw()
        if kw is None:
            return self.whip_rad
        sub = 2.0e-4
        r = None
        for _ in range(max(1, int(dt / sub))):
            r = self._beam(modal_q_m=self.whip_q_m,
                           modal_qdot_m_s=self.whip_qdot_m_s,
                           force_n=0.0, dt_s=sub, **kw)
            self.whip_q_m = r["modal_q_next_m"]
            self.whip_qdot_m_s = r["modal_qdot_next_m_s"]
        if impulse_n_s > 0.0:
            # RECOIL IS AXIAL. It runs back along the bore and on a
            # perfectly symmetric gun it bends the tube not at all.
            # Applying the whole 1340 N.s impulse as a transverse force
            # -- which the first version did -- drove the bending mode
            # with half a meganewton and reported fourteen milliradians
            # of whip, eleven metres of miss at 800 m.
            #
            # What actually bends it is that the bore axis does not pass
            # through the trunnion. The recoil force acts on a lever
            # equal to that offset, and the moment it makes is what the
            # tube feels sideways. A gun trunnioned exactly on its bore
            # line has no whip from recoil at all, which is precisely
            # why mounts are built that way where they can be.
            nodes = {n["identity"]: n for n in self.graph["nodes"]}
            bore_y = nodes.get("turret.weapon", {}).get(
                "reference_position", (0.0, 0.44, 0.0))[1]
            pivot_y = nodes.get("turret.pitch", {}).get(
                "reference_position", (0.0, 0.42, 0.0))[1]
            offset = abs(float(bore_y) - float(pivot_y))
            axial_f = impulse_n_s / in_tube_s
            length = kw["length_m"]
            f = axial_f * offset / max(length, 1e-6)
            self.recoil_offset_m = offset
            sub2 = 1.0e-5
            for _ in range(int(in_tube_s / sub2)):
                r = self._beam(modal_q_m=self.whip_q_m,
                               modal_qdot_m_s=self.whip_qdot_m_s,
                               force_n=f, dt_s=sub2, **kw)
                self.whip_q_m = r["modal_q_next_m"]
                self.whip_qdot_m_s = r["modal_qdot_next_m_s"]
        if r is not None:
            self.whip_rad = r["tip_slope_rad"]
            self.first_mode_hz = r["natural_frequency_hz"]
        return self.whip_rad

    # ---------------- the tick ----------------
    def step(self, dt: float, hostiles) -> dict:
        self.step_tube(dt)
        fc = self.fire_control
        target = fc.acquire(hostiles) if fc else None
        engaging = target is not None

        # ---- the turret's own electrics, always ----
        load = LOAD_STANDBY_W + (LOAD_FIRE_CONTROL_W + LOAD_SIGHT_W if engaging else 0.0)
        if fc and fc.tracking_error_deg > 0.05:
            load += LOAD_VALVES_W
        self.electrical_load_w = load

        # ---- should the engine be running? ----
        decision = self.autostart.step(
            dt, battery=self.battery.state_of_charge,
            accumulator=self.accumulator_charge, air=self.air_charge,
            engagement=engaging)
        if decision["event"] == "start":
            if self.battery.can_crank:
                self.note(f"auto-start: {', '.join(decision['reasons'])}")
            else:
                self.autostart.running = False
                decision["running"] = False
                self.note("auto-start REFUSED: battery too flat to crank")
        elif decision["event"] == "stop":
            self.note("engine stopped: stores full")
        self.engine_running = bool(decision["running"])

        # cranking is a big draw, and it comes out of the same battery
        if decision["cranking"]:
            load += STARTER_W
        self.battery.draw(load, dt)

        # ---- the engine, and what it replenishes ----
        sim = self.engine_sim
        flow = 0.0
        if sim is not None:
            sim.throttle = 0.55 if self.engine_running else 0.0
            if self.engine_running and sim.rpm < 200.0 and self.battery.can_crank:
                sim.start()
            sim.step(dt)
            plant = getattr(sim, "plant", None)
            hyd = getattr(plant, "hydraulics", None) if plant else None
            flow = float(getattr(hyd, "pump_flow_lpm", 0.0) or 0.0)
            if self.engine_running and sim.rpm > 400.0:
                self.battery.charge(1_800.0, dt)                 # the alternator
                self.accumulator_charge = min(1.0, self.accumulator_charge + 0.10 * dt)
                self.air_charge = min(1.0, self.air_charge + 0.05 * dt)

        # ---- the mount draws the accumulator down when it moves ----
        if fc:
            moving = fc.tracking_error_deg > 0.05
            if moving:
                self.accumulator_charge = max(0.0, self.accumulator_charge - 0.06 * dt)
            # what the mount can do is what the stores allow
            supply = 0.15 + 0.85 * max(self.accumulator_charge,
                                       min(1.0, flow / 130.0))
            fc.traverse_rate_deg_s = 60.0 * supply
            fc.elevation_rate_deg_s = 15.6 * supply
            # electronics only solve when they have power
            if self.battery.state_of_charge <= 0.02:
                fc.target = None
                fc.solver = None
            state = fc.step(dt)
        else:
            state = {}

        # ---- and whether it may fire ----
        may_fire = False
        if state.get("on_target") and target is not None:
            direction = np.asarray(target.position, dtype=float) - fc.muzzle_position
            may_fire = self.friendly.clear_to_fire(fc.muzzle_position, direction)
            if not may_fire:
                self.note(f"HOLD FIRE: own {self.friendly.blocked_by} in the line")
        state["may_fire"] = may_fire
        state["engine_running"] = self.engine_running
        state["battery"] = self.battery.state_of_charge
        state["accumulator"] = self.accumulator_charge
        state["air"] = self.air_charge
        state["load_w"] = self.electrical_load_w
        state["reasons"] = decision["reasons"]
        return state

    # ---------------- the shot ----------------
    def fire(self, target, gun) -> dict:
        """Send an anti-material round at what the mount is pointed at.

        Anti-material is the right description rather than a flourish:
        the projectile is chosen to defeat STRUCTURE -- an engine block,
        a wheel station, a radiator -- not to wound. What comes back is
        a traversal report saying which parts it reached and where it
        stopped, which is the only honest way to say what a hit did."""
        self.rounds_fired += 1
        if not hasattr(self, "_rng"):
            self._rng = np.random.default_rng(20260911)
        muzzle = np.asarray(self.fire_control.muzzle_position, dtype=float)
        # THE RECOIL LANDS ON THE SLIDE, as an impulse added to whatever
        # velocity the slide already has -- not reset to a fresh shot
        # every time. This is what makes rapid fire show real partial
        # recovery: fire faster than the slide returns to battery and
        # the next round's impulse stacks on top of the last one's.
        if self.turret_sim is not None:
            from calibres import get_calibre, recoil_of
            # the machine's OWN declared cartridge (machines.py's
            # sight_cartridge), not a re-derived guess from a display
            # name -- the two must never be able to disagree
            cal = get_calibre(getattr(self.turret_sim.machine, "sight_cartridge", "40mm l70"))
            wrench = recoil_of(cal, direction=(0.0, 0.0, 1.0))
            self.turret_sim.fire_recoil(wrench.net_impulse_n_s)
            # AND THE SAME IMPULSE RINGS THE TUBE. It is one shot: the
            # slide takes it as travel and the barrel takes it as a
            # bending mode, from the one impulse, so the two can never
            # be told different stories about how hard the gun was hit.
            self.step_tube(0.0, impulse_n_s=wrench.net_impulse_n_s)
        # USE THE BAKED TABLE, not a fresh integration. Re-tracing the
        # trajectory for every round was costing more than the rest of
        # the simulation put together, and it was answering a question
        # the bake already answered -- what the round is doing when it
        # arrives. The table gives speed and energy at range; the ray
        # does the rest.
        # THE ROUND IS EMITTED, NOT RESOLVED. Firing used to mean
        # "work out where this lands, now" -- the trajectory kernel ran
        # the whole flight inside this call and the impact was decided
        # before the tick ended. The physics in that flight was real;
        # the TIME was not. A round that takes eight tenths of a second
        # to reach eight hundred metres arrived in the same instant it
        # left, so nothing could move while it travelled and a lead
        # error could never actually miss.
        #
        # So this puts a projectile in the air and returns. It leaves
        # the gun's real muzzle, flies under the same compiled drag law
        # at its own ballistic timestep, and `ProjectileField.step`
        # resolves it against whatever its path actually crosses,
        # whenever that happens -- which may be several ticks and
        # several metres of target movement later.
        if self.projectiles is None:
            from projectiles import ProjectileField
            self.projectiles = ProjectileField(
                wind=(float(getattr(self.fire_control, "crosswind_m_s", 0.0)), 0.0, 0.0))
        fired = (getattr(self.turret_sim.machine, "sight_cartridge", None)
                 if self.turret_sim is not None else None)
        fired = fired or getattr(gun, "calibre", "40mm l70")
        spread = float(getattr(self.turret_sim.machine, "dispersion_mrad", 0.6)
                       if self.turret_sim is not None else 0.6)
        # THE MUZZLE AND THE BORE COME FROM THE GUN ITSELF when the
        # caller has a posed document to read them from; otherwise the
        # fire-control angles stand in. The first is the truth, the
        # second is the best available.
        if self.muzzle_pose is not None:
            origin, bore = self.muzzle_pose()
        else:
            origin = muzzle
            b = math.radians(self.fire_control.bearing_deg)
            el = math.radians(self.fire_control.elevation_deg)
            bore = np.array([math.sin(b) * math.cos(el), math.sin(el),
                             math.cos(b) * math.cos(el)])
        shot = self.projectiles.fire(muzzle=origin, direction=bore, calibre=fired,
                                     target=target.identity, dispersion_mrad=spread,
                                     rng=self._rng)
        self.note(f"round {self.rounds_fired} away at {target.identity} "
                  f"({fired}, {spread:.1f} mrad)")
        return {"target": target.identity, "round": shot.identity, "in_flight": True}

    def step_projectiles(self, dt: float, hostiles) -> list:
        """Fly what is in the air, and book whatever it reaches.

        Kept separate from `step` because a round outlives the tick that
        fired it: the station may be tracking something else entirely by
        the time its last shot arrives, and the bookkeeping belongs to
        the round rather than to whatever the mount is doing now."""
        if self.projectiles is None:
            return []
        bodies = {h.identity: (h.position, getattr(h, "body", None))
                  for h in hostiles if getattr(h, "body", None) is not None}
        events = self.projectiles.step(dt, bodies)
        for ev in events:
            report = ev.get("report")
            if report is not None:
                self.engagements.append((ev.get("target", "?"), report))
            effect = ev.get("effect") or {}
            what = ev["outcome"]
            if effect.get("destroyed"):
                what = f"DESTROYED -- {effect['killed_by']}"
            self.note(f"{ev['round']} -> {ev.get('target', 'ground')}: {what}"
                      + (f"  ({ev['flight_s'] * 1000:.0f} ms, {ev['range_m']:.0f} m, "
                         f"{ev['impact_speed_m_s']:.0f} m/s)" if "flight_s" in ev else ""))
        return events

    def summary(self) -> list[str]:
        out = [f"  {self.identity}: engine {'RUNNING' if self.engine_running else 'stopped'}"
               f"   load {self.electrical_load_w:5.0f} W   rounds {self.rounds_fired}"]
        out.extend(self.battery.describe())
        out.append(f"  stores: accumulator {self.accumulator_charge * 100:5.1f} %   "
                   f"air {self.air_charge * 100:5.1f} %")
        if self.autostart.reasons:
            out.append(f"  running because: {', '.join(self.autostart.reasons)}")
        out.extend(f"    {line}" for line in self.log[-5:])
        return out
