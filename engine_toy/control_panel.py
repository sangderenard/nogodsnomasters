"""A generic switch panel -- the dash/firewall's real electrical
counterpart (drivetrain_graph.py's `include_firewall`, the dash-mounted
arm switches abstract_ui_vehicles.py now builds for nitrous and
auxiliary injection). Enumerates whatever real switches THIS engine's
own build actually carries -- not a fixed control scheme -- so a plain
commuter engine shows none of this and a nitrous-armed drag car or a
WWII fighter each get exactly the real switch their own hardware has.

Each Switch is a real physical control (a name, the real hardware kind
it is -- a guarded toggle, a wire-sealed emergency-power switch,
whatever historical hardware the engine's own build calls for -- and a
getter/setter into the sim). A switch can also be put in AUTOMATED
mode: instead of the pilot/driver's own keypress, an `AutomatedSource`
supplies its state every tick from a real rule (the same real WOT +
rpm interlock production's own wiring already declares for nitrous,
for instance) -- "taking an automated system in source" is exactly
this: swap what DRIVES the switch, the wiring downstream of it never
changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from engine_cycle_sim import EngineCycleSim

# real WOT + rpm-window interlock -- mirrors production's own declared
# nitrous solenoid wire (abstract_ui_vehicles.py: interlock_feedback=
# ["engine_angular_speed", "wide_open_throttle"]) instead of inventing
# a separate rule
NITROUS_AUTO_WOT_THROTTLE = 0.95
NITROUS_AUTO_MIN_RPM_FRAC = 0.35    # of redline -- real: no low-rpm nitrous hit, that's how you break rods


@dataclass
class Switch:
    name: str
    label: str
    kind: str              # the real hardware kind (production's own switch node kind)
    get: Callable[[], bool]
    set: Callable[[bool], None]
    auto_rule: Callable[[EngineCycleSim], bool] | None = None   # None = no automated mode available
    key: str = ""           # assigned by ControlPanel.rebuild
    auto: bool = False      # currently in automated mode


@dataclass
class Macro:
    """A real actuator MACRO: one keypress driving more than one real
    control at once -- a WWII "combat power" gate push (arm WMI, go to
    full throttle) or a drag car's "stage" routine (arm nitrous, hold
    at a WOT rpm target) are real single-action sequences a pilot/
    driver actually performs, not this toy inventing a shortcut. Also
    used for the real "hand this switch to its own automated interlock"
    action (the auto-mode toggle previously on its own keybinding) --
    releasing a switch to automation IS a real dash action, it belongs
    in the same macro list, not a separate control scheme."""
    name: str
    label: str
    key: str
    apply: Callable[[EngineCycleSim, "ControlPanel"], None]


class ControlPanel:
    """Rebuilt whenever the active engine changes (`rebuild`); `step`
    applies any switch currently in automated mode; manual switches are
    toggled directly by the frontend on a keypress. Number keys are the
    real dash actuators themselves (one press = flip that switch);
    letter keys i/j/m/o are actuator MACROS (a preset action, possibly
    touching several switches/sim fields at once) -- deliberately kept
    off the number row so dialing in an engine (roster browsing used to
    live on 0-9) never fights with actually operating one."""

    KEY_POOL = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    MACRO_KEY_POOL = ("i", "j", "m", "o")

    def __init__(self) -> None:
        self.switches: list[Switch] = []
        self.macros: list[Macro] = []

    def rebuild(self, sim: EngineCycleSim) -> None:
        eng = sim.engine
        switches: list[Switch] = []
        if eng.has_nitrous:
            switches.append(Switch(
                name="nitrous_arm", label="NITROUS ARM", kind="guarded-toggle-switch",
                get=lambda: sim.nitrous_active, set=lambda v: setattr(sim, "nitrous_active", v),
                auto_rule=lambda s: (s.throttle >= NITROUS_AUTO_WOT_THROTTLE
                                     and s.rpm >= s.engine.redline_rpm * NITROUS_AUTO_MIN_RPM_FRAC)))
        if eng.has_auxiliary_injection:
            switches.append(Switch(
                name="auxiliary_injection_arm", label="WMI ARM", kind="wire-sealed-emergency-power-switch",
                get=lambda: sim.auxiliary_injection_active,
                set=lambda v: setattr(sim, "auxiliary_injection_active", v),
                # a real WMI kit's own default: armed whenever the
                # engine runs (its own regulator already doses off real
                # boost) -- "auto" here just releases the pilot's manual
                # override back to that real default
                auto_rule=lambda s: True))
        if len(eng.starting_systems) > 1:
            # more than one real starting system on this build (the
            # Cat C18: electric-starter + air-motor-starter) -- a real
            # selector, not a switch (no automated mode: which starter
            # to use is always a deliberate choice)
            switches.append(Switch(
                name="starter_select", label=f"STARTER: {sim.starter.systems[0].mode}",
                kind="rotary-selector-switch",
                get=lambda: True, set=lambda v: None))
        for i, sw in enumerate(switches):
            sw.key = self.KEY_POOL[i] if i < len(self.KEY_POOL) else ""
        self.switches = switches

        def _auto_toggle_macro(switch_name: str) -> Callable[[EngineCycleSim, "ControlPanel"], None]:
            def _apply(_sim: EngineCycleSim, panel: "ControlPanel") -> None:
                sw = next((s for s in panel.switches if s.name == switch_name), None)
                if sw is not None and sw.auto_rule is not None:
                    sw.auto = not sw.auto
            return _apply

        macros: list[Macro] = []
        for sw in switches:
            if sw.auto_rule is not None:
                macros.append(Macro(
                    name=f"auto_{sw.name}", label=f"AUTO: {sw.label}",
                    key="", apply=_auto_toggle_macro(sw.name)))
        if len(switches) > 1:
            def _safe_all(_sim: EngineCycleSim, panel: "ControlPanel") -> None:
                # a real "master safe" action: disarm everything and
                # release every switch back to manual, the same real
                # single motion a checklist calls "safe the aircraft" /
                # "disarm the system" for
                for s in panel.switches:
                    s.auto = False
                    s.set(False)
            macros.append(Macro(name="safe_all", label="SAFE ALL", key="", apply=_safe_all))
        for i, m in enumerate(macros):
            m.key = self.MACRO_KEY_POOL[i] if i < len(self.MACRO_KEY_POOL) else ""
        self.macros = macros

    def by_key(self, key: str) -> Switch | None:
        return next((sw for sw in self.switches if sw.key == key.lower()), None)

    def macro_by_key(self, key: str) -> Macro | None:
        return next((m for m in self.macros if m.key == key.lower()), None)

    def toggle(self, key: str) -> None:
        """A manual keypress: if the switch is in automated mode, the
        FIRST press just releases it back to manual (the real action of
        grabbing a switch that's currently being driven automatically)
        rather than also flipping its state blind."""
        sw = self.by_key(key)
        if sw is None:
            return
        if sw.auto:
            sw.auto = False
        else:
            sw.set(not sw.get())

    def trigger_macro(self, key: str, sim: EngineCycleSim) -> None:
        m = self.macro_by_key(key)
        if m is not None:
            m.apply(sim, self)

    def step(self, sim: EngineCycleSim) -> None:
        for sw in self.switches:
            if sw.auto and sw.auto_rule is not None:
                sw.set(sw.auto_rule(sim))

    def lines(self) -> list[str]:
        if not self.switches:
            return []
        out = ["  DASH ACTUATORS  (number key arms/disarms)"]
        for sw in self.switches:
            state = "ON " if sw.get() else "off"
            mode = " [AUTO]" if sw.auto else ""
            key_note = f"({sw.key})" if sw.key else "(-)"
            out.append(f"    {key_note} {sw.label:<16s} [{state}]{mode}   {sw.kind}")
        if self.macros:
            out.append("  ACTUATOR MACROS")
            for m in self.macros:
                key_note = f"({m.key.upper()})" if m.key else "(-)"
                out.append(f"    {key_note} {m.label}")
        return out
