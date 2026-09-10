"""Real-time additive synthesis at the two acoustic emission points.

`render_stereo` is the live (streaming, phase-continuous) counterpart to
`engine_baker.bake_acoustic` -- same idea, two spatial listening points,
but built from persistent phase accumulators block-to-block instead of
a re-triggered loop, so it can run under a live audio callback without
clicking.

  header      -- inside the header/primary-drive housing: the sharp,
                 harmonically rich firing-frequency note (or motor
                 whine for electric/servo units).
  engine_bay  -- standing next to the block: softer, more articulated
                 valvetrain tick, oil/water pump and belt/fan/alternator
                 tones, plus a heavily muffled bleed of the header note
                 (what actually carries through the block and firewall).
"""
from __future__ import annotations

import math

import numpy as np

from engines import Engine
import native_audio

SAMPLE_RATE = 44100
BLOCK_SIZE = 512


class EngineSoundSynth:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sr = sample_rate
        self.harm_phase = np.zeros(8)
        self.half_phase = 0.0
        self.motor_phase = np.zeros(3)
        self.noise_state = 0.0
        self.turbine_whine_phase = np.zeros(3)
        self.turbine_noise_state = 0.0
        self.rng = np.random.default_rng(1234)

        self.valve_phase = 0.0
        self.pump_phase = np.zeros(4)   # oil pump, water pump, belt flutter, alternator
        self.fan_phase = 0.0
        self.bleed_state = 0.0

        self.knock_env = 0.0
        self.knock_phase = np.zeros(2)
        self.misfire_env = 0.0

        self.turbo_phase = 0.0
        self.sc_phase = 0.0
        self.wastegate_phase = 0.0
        self.intake_noise_state = 0.0
        self.intake_gate_phase = 0.0
        self._prev_throttle_for_intake = 0.0
        self.backfire_boom_env = 0.0
        self.backfire_crack_env = 0.0
        self.backfire_noise_lp_state = 0.0
        self.backfire_phase = 0.0

    def render_stereo(self, engine: Engine, rpm: float, throttle: float,
                       load_frac: float, n_frames: int,
                       knock_active: bool = False, knock_intensity: float = 0.0,
                       misfire_active: bool = False,
                       boost_frac: float = 0.0, turbo_spool_frac: float = 0.0,
                       wastegate_flutter: bool = False, surge_active: bool = False,
                       backfire_active: bool = False, backfire_kind: str | None = None,
                       backfire_strength: float = 0.0,
                       real_fire_hz: float = 0.0,
                       intake_flow_demand_frac: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        cylinders = engine.architecture.cylinders
        # A discrete-combustion header (sharp, harmonically rich pulses)
        # applies to any engine with real individual firing events --
        # a crank engine (cylinders > 0) AND a free-piston one
        # (kind=="atmospheric", cylinders==0 but still real discrete
        # combustion, just not tied to a rotating crank at all). Both
        # are driven by real_fire_hz, the actually-measured firing rate
        # (EngineCycleState.real_fire_hz -- every_engine_cycle_sim's own
        # general per-kind ignition tracking, see _record_ignition), not
        # a formula assuming firing rate is a fixed ratio of shaft rpm --
        # true for a crank engine, but not the reason this argument
        # exists: a free-piston engine's cycle time is set by its own
        # gas-law physics regardless of flywheel speed, so a formula
        # would be flatly wrong there, not just less precise. Only a
        # genuinely continuous-combustion or non-combustion engine
        # (electric/servo/turbine) falls back to the motor-whine header.
        if cylinders > 0 or engine.kind == "atmospheric":
            header = self._render_header_combustion(engine, rpm, throttle, load_frac, n_frames, real_fire_hz)
        elif engine.kind == "turbine":
            # a real gas turbine's dominant tone is the gas-generator
            # spool's own compressor/turbine blade-pass whine -- pitched
            # off N1 (turbo_spool_frac, the real shaft-speed fraction
            # gas_turbine.py already computes), NOT the output shaft rpm
            # this method's other callers use: the output shaft sits
            # behind a real reduction gearbox (engines.TurbineSpec.
            # reduction_ratio) and can be spinning at a small fraction of
            # the gas generator's real speed -- pitching off it would
            # radically understate how fast this engine's own spinning
            # mass actually is.
            header = self._render_header_turbine(engine, turbo_spool_frac, throttle, load_frac, n_frames)
        else:
            header = self._render_header_electric(engine, rpm, throttle, load_frac, n_frames)

        if knock_active:
            self.knock_env = max(self.knock_env, 0.5 + 0.5 * knock_intensity)
        if misfire_active:
            self.misfire_env = 1.0

        if self.knock_env > 1e-4:
            steps = np.arange(1, n_frames + 1)
            ping = np.zeros(n_frames)
            for i, freq in enumerate((3400.0, 5200.0)):
                phase_inc = 2 * np.pi * freq / self.sr
                phases = self.knock_phase[i] + phase_inc * steps
                ping += np.sin(phases)
                self.knock_phase[i] = phases[-1] % (2 * np.pi)
            decay = np.exp(-np.arange(n_frames) / (self.sr * 0.006))
            env_trace = self.knock_env * decay
            header = header + 0.5 * env_trace * ping
            self.knock_env *= math.exp(-n_frames / (self.sr * 0.006))

        if self.misfire_env > 1e-3:
            decay = np.exp(-np.arange(n_frames) / (self.sr * 0.02))
            dip = 1.0 - 0.85 * self.misfire_env * decay
            header = header * dip
            self.misfire_env *= math.exp(-n_frames / (self.sr * 0.02))

        header = header + self._render_backfire(n_frames, backfire_active, backfire_strength)

        bay = self._render_engine_bay(engine, rpm, throttle, load_frac, n_frames, header)
        bay = bay + self._render_induction(engine, rpm, throttle, load_frac, n_frames,
                                            boost_frac, turbo_spool_frac, wastegate_flutter, surge_active,
                                            real_fire_hz, intake_flow_demand_frac)

        # A stopped crank has no physical excitation to make any of this
        # noise from -- no combustion, no valvetrain motion, no spinning
        # pump/fan/belt, no exhaust flow to pop a backfire through. Every
        # component above is built from oscillators/filters that have a
        # numerical floor (a tiny nonzero firing-frequency clamp, a
        # constant noise-floor coefficient, a phase accumulator that holds
        # its last nonzero value when its own frequency hits exactly
        # zero) to stay well-behaved as rpm approaches zero, not to keep
        # sounding at rpm == 0. This is the one real, physically-grounded
        # gate that actually silences the output there -- a smooth ramp
        # over the first 40 rpm (genuine starter-cranking speed) rather
        # than a hard step, so cranking still fades in instead of
        # clicking, but a genuinely stopped/stalled engine is silent.
        # for kind=="atmospheric", rpm is the FLYWHEEL's speed, which
        # can genuinely lag real piston activity for seconds (a cold-
        # start bootstrap, a freewheeling ratchet) -- gating solely on
        # it would silence a real, actively-firing engine. real_fire_hz
        # is the piston's own real activity signal; either one being
        # up is enough to be audible.
        activity = max(rpm, 0.0) if engine.kind != "atmospheric" else max(rpm, real_fire_hz * 40.0)
        run_gate = np.clip(activity / 40.0, 0.0, 1.0)
        header = header * run_gate
        bay = bay * run_gate
        return np.tanh(header * 1.3).astype(np.float32), np.tanh(bay * 1.3).astype(np.float32)

    # Backward-compatible single-channel accessor (header only).
    def render(self, engine: Engine, rpm: float, throttle: float,
               load_frac: float, n_frames: int) -> np.ndarray:
        header, _bay = self.render_stereo(engine, rpm, throttle, load_frac, n_frames)
        return header

    def _render_header_combustion(self, engine, rpm, throttle, load_frac, n_frames,
                                   real_fire_hz: float = 0.0) -> np.ndarray:
        sr = self.sr
        firing_events_per_rev = engine.firing_events_per_rev()
        # the REAL, measured firing rate (EngineCycleState.real_fire_hz)
        # -- not rpm*firing_events_per_rev, which assumes firing rate is
        # a fixed ratio of shaft rpm. True for a crank engine in normal
        # running (so this and that formula mostly agree there), false
        # for a free-piston engine (firing_events_per_rev is even
        # structurally 0 for one -- cylinders==0 -- so that formula
        # would be silent here regardless). The rpm-based formula is
        # kept only as a fallback for a caller that hasn't wired
        # real_fire_hz through yet.
        fire_freq = max(real_fire_hz, 0.01) if real_fire_hz > 0.0 else max(
            max(rpm, 0.0) / 60.0 * firing_events_per_rev, 0.01)
        exhaust = engine.exhaust_system
        # the exhaust hardware itself shapes the tone, not just rpm/throttle:
        # a boxed-in stock manifold (cat + muffler) rolls off high harmonics,
        # an open header lets them all through -- same brightness_frac that
        # also derives the pipe's static backpressure
        brightness = exhaust.brightness_frac
        n_harm = 8
        idx = np.arange(1, n_harm + 1)
        base_amps = 1.0 / (idx ** 1.15)
        rasp = (0.15 + 0.85 * throttle) * (0.55 + 0.45 * brightness)
        weight = rasp ** np.arange(0, n_harm)
        amps = base_amps * weight
        amps /= amps.sum()

        sig = np.zeros(n_frames)
        steps = np.arange(1, n_frames + 1)
        for i in range(n_harm):
            freq = fire_freq * (i + 1)
            phase_inc = 2 * np.pi * freq / sr
            phases = self.harm_phase[i] + phase_inc * steps
            sig += amps[i] * np.sin(phases)
            self.harm_phase[i] = phases[-1] % (2 * np.pi)

        wobble_amt = engine.architecture.wobble_amt
        half_inc = 2 * np.pi * (fire_freq / 2.0) / sr
        half_phases = self.half_phase + half_inc * steps
        sig *= 1.0 + wobble_amt * np.sin(half_phases)
        self.half_phase = half_phases[-1] % (2 * np.pi)

        noise = self.rng.standard_normal(n_frames)
        alpha = 0.15
        filtered, self.noise_state = native_audio.leaky_integrator(noise, alpha, self.noise_state)
        noise_level = (0.05 + 0.25 * throttle + 0.15 * load_frac) * (0.5 + 0.5 * brightness)
        sig += filtered * noise_level

        if engine.kind == "atmospheric":
            # this engine's own "rpm" is the FLYWHEEL's speed, not the
            # piston's -- the two are only loosely coupled through a
            # one-way ratchet (see otto_langen.py's own module
            # docstring), and the flywheel can genuinely lag far behind
            # real piston activity (a whole cold-start bootstrap, or
            # any tick the ratchet is freewheeling). rpm/redline_rpm
            # and a crank-rpm-tuned exhaust pipe resonance both assume
            # a rigid crank relationship this mechanism doesn't have,
            # so neither applies here -- loudness instead follows the
            # piston's own real, already-measured activity (real_fire_
            # hz), and there's no throttle plate to modulate charge
            # size with either (real, fixed-charge ignition every real
            # cycle), so no size_factor from a control input that has
            # no real effect on this engine.
            size_factor = 1.0
            # ramps up to full over the first ~1 Hz of real firing rate
            # (this engine's own typical governed cycle rate, verified
            # running in the ~0.3-2 Hz band) -- zero only when it's
            # genuinely not firing at all, not when the flywheel is slow
            rate_factor = np.clip(real_fire_hz / 1.0, 0.0, 1.0)
            resonance_boost = 1.0
        else:
            rpm_frac = np.clip(rpm / engine.redline_rpm, 0.0, 1.2)
            # it's always firing while running -- rpm (how many explosions per
            # second) is what actually drives whether you hear it at all; idle
            # lope and closed-throttle engine braking are both real, recognizable
            # sounds, not silence. throttle governs how BIG each explosion is
            # (that's torque), so it shapes size, not whether there's sound.
            size_factor = 0.55 + 0.45 * throttle
            rate_factor = 0.45 + 0.55 * rpm_frac

            # the pipe's own quarter-wave tuned rpm band -- a real header
            # "coming alive"/singing at one specific rpm as the reflected
            # exhaust pulse lines up with valve closing
            tuned_rpm = exhaust.tuned_rpm(firing_events_per_rev)
            detune = (rpm - tuned_rpm) / max(tuned_rpm * 0.25, 1.0)
            resonance_boost = 1.0 + 0.20 * max(0.0, 1.0 - detune * detune)

        loudness = size_factor * rate_factor * resonance_boost
        return sig * loudness

    def _render_header_electric(self, engine, rpm, throttle, load_frac, n_frames) -> np.ndarray:
        sr = self.sr
        base = max(rpm, 0.0) / 60.0
        mults = (6, 12, 18)
        sig = np.zeros(n_frames)
        steps = np.arange(1, n_frames + 1)
        for i, mult in enumerate(mults):
            freq = base * mult
            phase_inc = 2 * np.pi * freq / sr
            phases = self.motor_phase[i] + phase_inc * steps
            amp = 0.6 if i == 0 else 0.25 / i
            sig += amp * np.sin(phases)
            self.motor_phase[i] = phases[-1] % (2 * np.pi)

        whine = 0.05 + 0.3 * throttle + 0.2 * load_frac
        rpm_frac = np.clip(rpm / engine.redline_rpm, 0.0, 1.2)
        loudness = 0.2 + 0.8 * (0.3 + 0.7 * rpm_frac)
        return sig * whine * loudness * 1.5

    def _render_header_turbine(self, engine, n1_frac, throttle, load_frac, n_frames) -> np.ndarray:
        """A real gas turbine's own compressor/turbine spool whine --
        pitched off N1 (turbo_spool_frac, the gas-generator's own real
        shaft-speed fraction), not the output shaft rpm every other
        header here uses: the dominant real tone is the compressor/
        turbine blade-pass note, set by how fast that spool actually
        turns, independent of whatever real reduction gearbox
        (engines.TurbineSpec.reduction_ratio) sits between it and the
        output shaft."""
        sr = self.sr
        n1 = max(n1_frac, 0.0)
        # real, disclosed pitch mapping -- same style as the existing
        # turbocharger whistle in _render_induction (600 + 7000*spool_
        # frac): not a literal blade-count*shaft-Hz computation, a
        # plausible "pitch rises with spool speed" curve for a small
        # centrifugal stage
        base_freq = 220.0 + 3200.0 * n1
        mults = (1.0, 2.0, 3.0)
        sig = np.zeros(n_frames)
        steps = np.arange(1, n_frames + 1)
        for i, mult in enumerate(mults):
            freq = base_freq * mult
            phase_inc = 2 * np.pi * freq / sr
            phases = self.turbine_whine_phase[i] + phase_inc * steps
            amp = 0.6 if i == 0 else 0.3 / i
            sig += amp * np.sin(phases)
            self.turbine_whine_phase[i] = phases[-1] % (2 * np.pi)

        noise = self.rng.standard_normal(n_frames)
        filtered, self.turbine_noise_state = native_audio.leaky_integrator(noise, 0.2, self.turbine_noise_state)
        # real combustor roar -- rises with N1 and throttle (more fuel,
        # more mass flow, more real broadband noise), same physical
        # reasoning the induction roar layer below uses for a piston
        # engine's own intake
        roar_level = 0.05 + 0.30 * n1 + 0.15 * throttle
        sig += filtered * roar_level

        loudness = 0.25 + 0.75 * n1
        return sig * loudness

    def _render_engine_bay(self, engine, rpm, throttle, load_frac, n_frames, header) -> np.ndarray:
        sr = self.sr
        steps = np.arange(1, n_frames + 1)
        crank_freq = max(rpm, 0.0) / 60.0
        rpm_frac = np.clip(rpm / max(engine.redline_rpm, 1.0), 0.0, 1.2)
        sig = np.zeros(n_frames)
        cylinders = engine.architecture.cylinders

        if cylinders > 0:
            acc = engine.accessories
            if not engine.architecture.rotary:
                # valvetrain: a soft double-tick per cylinder pair, cam-order
                # rate -- every poppet-valve engine has this, a port-timed
                # rotary doesn't. A stiffer lifter spring makes it louder and
                # sharper (higher click exponent = narrower pulse).
                spring_ratio = engine.lifter_spring.spring_rate_n_per_mm / 45.0
                valve_freq = crank_freq / 2.0 * max(cylinders, 1)
                phase_inc = 2 * np.pi * valve_freq / sr
                phases = self.valve_phase + phase_inc * steps
                click_sharpness = np.clip(24.0 * spring_ratio ** 0.5, 10.0, 60.0)
                click = np.abs(np.sin(phases)) ** click_sharpness
                sig += 0.5 * (0.3 + 0.7 * rpm_frac) * min(2.0, spring_ratio ** 0.5) * click
                self.valve_phase = phases[-1] % (2 * np.pi)

            # oil pump is essential and always fitted; water pump/belt
            # accessories are per-loadout and get silently skipped (phase
            # state just idles) when the engine doesn't carry them
            pump_specs = [
                (True, crank_freq, 0.10 + 0.14 * rpm_frac),
                (acc.water_pump, crank_freq * 1.2, 0.08 + 0.10 * rpm_frac),
                (acc.mechanical_fan or acc.alternator, crank_freq * 2.6, 0.05 + 0.04 * load_frac),
                (acc.alternator, crank_freq * 2.9, 0.06 + 0.12 * load_frac),
            ]
            for i, (fitted, freq, amp) in enumerate(pump_specs):
                phase_inc = 2 * np.pi * freq / sr
                phases = self.pump_phase[i] + phase_inc * steps
                if fitted:
                    sig += amp * np.sin(phases)
                self.pump_phase[i] = phases[-1] % (2 * np.pi)

            if acc.mechanical_fan:
                fan_engaged = min(1.0, max(0.0, rpm_frac * 1.4 - 0.15))
                fan_freq = crank_freq * 1.05 * 6.0
                phase_inc = 2 * np.pi * fan_freq / sr
                phases = self.fan_phase + phase_inc * steps
                sig += 0.08 * fan_engaged * fan_engaged * np.sin(phases)
                self.fan_phase = phases[-1] % (2 * np.pi)
        else:
            phase_inc = 2 * np.pi * (crank_freq * 1.3) / sr
            phases = self.pump_phase[0] + phase_inc * steps
            if engine.accessories.coolant_pump:
                sig += (0.06 + 0.06 * rpm_frac) * np.sin(phases)
            self.pump_phase[0] = phases[-1] % (2 * np.pi)

        # muffled bleed of the header note: heavy one-pole lowpass + attenuation
        alpha = 0.06
        bleed, self.bleed_state = native_audio.leaky_integrator(header, alpha, self.bleed_state)
        sig += bleed * 0.35

        loudness = 0.3 + 0.5 * throttle + 0.2 * load_frac
        return sig * loudness

    def _render_induction(self, engine, rpm, throttle, load_frac, n_frames,
                           boost_frac, turbo_spool_frac, wastegate_flutter, surge_active,
                           real_fire_hz: float = 0.0, intake_flow_demand_frac: float = 0.0) -> np.ndarray:
        """Turbo compressor whistle (pitch rises with spool speed) or
        supercharger blower whine (order-locked to crank), plus a real
        intake roar. A crank engine's intake is a genuinely PULSED
        event -- the port/valve opens once per real cycle (real_fire_hz,
        the same actually-measured firing rate every cylinder kind now
        tracks -- see engine_cycle_sim.EngineCycleSim._record_ignition),
        not a continuous hiss -- so it's gated open for a real fraction
        of that cycle (an intake stroke's own real duration) instead of
        playing all the time. Loudness comes from intake_flow_demand_
        frac, the REAL restriction (how hard the port is actually being
        asked to flow relative to its own real flow_capacity_kg_s
        ceiling, drivetrain_graph.py), not a throttle/rpm proxy -- a
        genuinely choked port reads as louder real turbulent-flow roar,
        an idling one as quiet, independent of where the pedal sits. A
        turbine's real intake is continuous instead (a compressor draws
        air constantly, no valve events at all), so it gets no gate,
        just the same real flow-proportional loudness (turbo_spool_frac
        is exactly proportional to this machine's own real compressor
        mass flow -- gas_turbine.py's mdot = mdot_design*omega_frac
        affinity law)."""
        fi = engine.forced_induction
        sr = self.sr
        steps = np.arange(1, n_frames + 1)
        sig = np.zeros(n_frames)

        if fi.kind == "turbo":
            whine_freq = 600.0 + 7000.0 * turbo_spool_frac
            phase_inc = 2 * np.pi * whine_freq / sr
            phases = self.turbo_phase + phase_inc * steps
            sig += 0.35 * turbo_spool_frac * np.sin(phases)
            self.turbo_phase = phases[-1] % (2 * np.pi)
            if wastegate_flutter:
                noise = self.rng.standard_normal(n_frames)
                gate_phase = self.wastegate_phase + 2 * np.pi * 40.0 * steps / sr
                gate = (np.sin(gate_phase) > 0.3).astype(np.float64)
                sig += 0.15 * noise * gate
                self.wastegate_phase = gate_phase[-1] % (2 * np.pi)
            if surge_active:
                lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 9.0 * steps / sr)
                sig += 0.3 * lfo * self.rng.standard_normal(n_frames) * 0.2
        elif fi.kind == "supercharger":
            order = fi.belt_ratio * fi.lobe_count
            whine_freq = max(rpm, 0.0) / 60.0 * order
            phase_inc = 2 * np.pi * whine_freq / sr
            phases = self.sc_phase + phase_inc * steps
            sig += 0.30 * (0.3 + 0.7 * boost_frac) * np.sin(phases)
            self.sc_phase = phases[-1] % (2 * np.pi)
        elif engine.kind == "turbine":
            # this engine's OWN real compressor is its whole induction
            # system (there's no separate turbo bolted to a different
            # engine) -- the same real physical phenomenon (a
            # centrifugal stage's whistle rising with spool speed) as
            # the turbo branch above, reused rather than re-derived, at
            # this bay/induction listening point rather than the header
            # (see _render_header_turbine for that primary-tone layer)
            whine_freq = 500.0 + 6000.0 * turbo_spool_frac
            phase_inc = 2 * np.pi * whine_freq / sr
            phases = self.turbo_phase + phase_inc * steps
            sig += 0.30 * turbo_spool_frac * np.sin(phases)
            self.turbo_phase = phases[-1] % (2 * np.pi)

        dthrottle = throttle - self._prev_throttle_for_intake
        self._prev_throttle_for_intake = throttle
        noise = self.rng.standard_normal(n_frames)
        alpha = 0.25
        filtered, self.intake_noise_state = native_audio.leaky_integrator(noise, alpha, self.intake_noise_state)
        flow = max(0.0, intake_flow_demand_frac)

        if engine.architecture.cylinders:
            # real port timing: pulse once per real cycle, open for a
            # real fraction of it -- a raised-cosine window, not a hard
            # click at the edges
            intake_hz = max(real_fire_hz, 0.01)
            phase_inc = 2 * np.pi * intake_hz / sr
            gate_phase = (self.intake_gate_phase + phase_inc * steps) % (2 * np.pi)
            self.intake_gate_phase = gate_phase[-1]
            duty = 0.35   # real, disclosed: an intake stroke's own fraction of a full 720-degree cycle
            frac = gate_phase / (2 * np.pi)
            gate = np.where(frac < duty, 0.5 - 0.5 * np.cos(2 * np.pi * frac / duty), 0.0)
            roar_level = 0.05 + 0.55 * min(1.3, flow) + 2.0 * min(0.3, abs(dthrottle))
            sig += filtered * roar_level * gate
        elif engine.kind == "turbine":
            # continuous real flow, no port events to gate against
            roar_level = 0.05 + 0.35 * turbo_spool_frac
            sig += filtered * roar_level
        return sig

    def _render_backfire(self, n_frames, active, strength) -> np.ndarray:
        """A gunshot-like bang, not a soft whoosh -- three components on
        three different decay times, not one blended envelope:
          - a single-sample hard click right at the trigger instant (the
            actual "pop" transient; a resonant sine/noise blend alone
            starts at zero slope and can never sound this sharp)
          - a fast, highpassed broadband crack (~8ms) riding right after it
          - a lower 130Hz boom that rings on (~50ms) after the crack has
            already died out
        Triggered by engine_cycle_sim's planned (anti-lag) or unplanned
        (misfire igniting in a hot/boosted exhaust) events."""
        sr = self.sr
        was_idle = self.backfire_boom_env <= 1e-4 and self.backfire_crack_env <= 1e-4
        if active:
            level = 0.6 + 0.4 * min(1.0, strength)
            self.backfire_boom_env = max(self.backfire_boom_env, level)
            self.backfire_crack_env = max(self.backfire_crack_env, level)
        is_new_trigger = active and was_idle
        if self.backfire_boom_env <= 1e-4 and self.backfire_crack_env <= 1e-4:
            return np.zeros(n_frames)

        steps = np.arange(1, n_frames + 1)
        idx = np.arange(n_frames)

        thump_phase_inc = 2 * np.pi * 130.0 / sr
        phases = self.backfire_phase + thump_phase_inc * steps
        thump = np.sin(phases)
        self.backfire_phase = phases[-1] % (2 * np.pi)

        raw_noise = self.rng.standard_normal(n_frames)
        lowpassed, self.backfire_noise_lp_state = native_audio.leaky_integrator(
            raw_noise, 0.5, self.backfire_noise_lp_state)
        crack_noise = raw_noise - lowpassed  # highpassed: a sharp crack, not a dull hiss

        boom_tau_s = 0.05
        crack_tau_s = 0.008
        boom_decay = np.exp(-idx / (sr * boom_tau_s))
        crack_decay = np.exp(-idx / (sr * crack_tau_s))

        sig = thump * 0.45 * self.backfire_boom_env * boom_decay
        sig = sig + crack_noise * 0.8 * self.backfire_crack_env * crack_decay
        if is_new_trigger:
            sig[0] += 1.5 * (0.6 + 0.4 * min(1.0, strength))

        self.backfire_boom_env *= math.exp(-n_frames / (sr * boom_tau_s))
        self.backfire_crack_env *= math.exp(-n_frames / (sr * crack_tau_s))
        return sig
