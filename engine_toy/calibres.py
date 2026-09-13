"""Real cartridges, and a firing squad's itinerary against the engine.

Each calibre is a real projectile for ballistics.ProjectileState: mass,
diameter, muzzle speed, length, and the core's hardness (a soft lead
handgun bullet mushrooms and stops in a casting; a jacketed rifle
bullet with a steel or tungsten core keeps going). Disclosed typical
service loads; a shooter can override the range (speed falls with
distance -- a flat drag law, conservative) and the yaw.

A `Shot` is one aim: a calibre, a target (a graph node or edge
identity, or a point), where it is fired from (a direction the shooter
stands in, at a range), how many rounds and how wide the group. A
`FiringSquad` is an ordered itinerary of shots with dwell time between
volleys (the engine keeps running, leaking, bursting between them);
`execute(sim, ray_mesh_factory)` fires every round through the same
path the app's right-click uses (engine_rays.RayMesh.penetrate ->
EngineCycleSim.apply_penetration: holes, emitters, damage sounds,
cascades) and returns a log of what each round did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ballistics import ProjectileState
from engine_rays import Ray


@dataclass(frozen=True)
class Calibre:
    name: str
    diameter_m: float
    mass_kg: float
    muzzle_m_s: float
    length_m: float
    hardness_pa: float          # the core: lead ~0.25 GPa, jacketed lead ~0.6, mild-steel core ~1.2, AP tungsten ~2.5
    drag_k_per_m: float = 0.0012   # v(r) = v0 exp(-k r): a pistol bullet loses ~10 % over 100 m, a rifle less
    yaw_rad: float = 0.0
    note: str = ""

    def projectile(self, direction, range_m: float = 5.0, yaw_rad: float | None = None) -> ProjectileState:
        v = self.muzzle_m_s * math.exp(-self.drag_k_per_m * max(range_m, 0.0))
        d = np.asarray(direction, dtype=float); d = d / max(np.linalg.norm(d), 1e-9)
        return ProjectileState(mass_kg=self.mass_kg, diameter_m=self.diameter_m, speed_m_s=v,
                               direction=tuple(float(x) for x in d), length_m=self.length_m,
                               yaw_rad=self.yaw_rad if yaw_rad is None else yaw_rad, hardness_pa=self.hardness_pa)

    @property
    def muzzle_energy_j(self) -> float:
        return 0.5 * self.mass_kg * self.muzzle_m_s ** 2


CALIBRES: dict[str, Calibre] = {c.name: c for c in (
    Calibre(".22 lr", 0.00570, 0.0026, 330.0, 0.0155, 0.25e9, 0.0020, note="rimfire, bare lead"),
    Calibre("9mm", 0.00902, 0.0080, 360.0, 0.0155, 0.60e9, 0.0015, note="9x19 FMJ 124 gr"),
    Calibre(".45 acp", 0.01143, 0.0149, 260.0, 0.0170, 0.55e9, 0.0018, note="230 gr FMJ"),
    Calibre(".357 mag", 0.00907, 0.0102, 440.0, 0.0165, 0.60e9, 0.0014, note="158 gr JSP"),
    Calibre(".500 s&w", 0.01270, 0.0227, 560.0, 0.0250, 0.65e9, 0.0012, note="350 gr, the big revolver"),
    Calibre("12 ga slug", 0.01840, 0.0280, 480.0, 0.0220, 0.20e9, 0.0030, note="1 oz soft lead Foster slug"),
    Calibre("5.56", 0.00570, 0.0040, 940.0, 0.0230, 1.00e9, 0.0006, note="5.56x45 M855, steel penetrator tip"),
    Calibre("5.8x42", 0.00600, 0.0042, 930.0, 0.0240, 1.05e9, 0.0006, note="DBP87, hardened steel core"),
    Calibre("7.62 nato", 0.00782, 0.0095, 840.0, 0.0290, 0.80e9, 0.0005, note="7.62x51 M80 ball"),
    Calibre("7.62x39", 0.00792, 0.0079, 715.0, 0.0260, 1.10e9, 0.0007, note="M43, mild-steel core"),
    Calibre(".308 ap", 0.00782, 0.0098, 830.0, 0.0290, 2.30e9, 0.0005, note="M61 AP, hardened-steel core"),
    Calibre(".338 lapua", 0.00858, 0.0162, 900.0, 0.0380, 0.85e9, 0.0004, note="250 gr match"),
    Calibre(".50 bmg", 0.01295, 0.0430, 890.0, 0.0580, 1.30e9, 0.0003, note="M33 ball"),
    Calibre(".50 ap", 0.01295, 0.0450, 880.0, 0.0580, 2.50e9, 0.0003, note="M2 AP, tungsten-carbide-class core"),
    # MOUNT-SIZED ORDNANCE. A turret's gun belongs in the same table as
    # everything else -- the ray, the material budget and the trajectory
    # law do not care how big the projectile is, and keeping the 40 mm
    # out of here would have meant a second, parallel ballistic path for
    # no reason. Bofors 40 mm L/70: 0.87 kg at 1010 m/s.
    Calibre("40mm l70", 0.040, 0.870, 1010.0, 0.180, 1.10e9, 0.00018,
            note="Bofors 40 mm L/70, mount-sized autocannon"),
    Calibre("30mm chain", 0.030, 0.360, 1080.0, 0.145, 1.60e9, 0.00020,
            note="30 mm chain gun"),
)}


@dataclass(frozen=True)
class RecoilWrench:
    """What firing puts back into the mount: a force and a moment.

    A shot is not one impulse, it is at least two and often three, and
    they do not all point the same way:

      THE PROJECTILE. Straight back along the bore, mass times muzzle
      velocity. This is the part everyone accounts for.

      THE PROPELLANT GAS. Several hundred grams of it leave the muzzle
      at roughly one and a half times the shell's speed, and on a
      full-charge gun that is a THIRD of the total recoil. Leaving it
      out under-predicts recoil by thirty per cent, which is the
      difference between a mount that holds and one that does not.

      THE VENT, if there is one. A muzzle brake turns some of that gas
      sideways and backwards, and the reaction pushes the gun FORWARD --
      which is why a braked gun recoils less while being far louder and
      far worse to stand beside. A recoilless gun takes this to its
      conclusion and vents enough rearward to cancel recoil entirely,
      paying for it with a back-blast that is lethal behind the weapon.

    And because the bore is not through the mount's centre of mass, the
    whole thing arrives as a MOMENT as well as a force -- the r x F the
    production graph's `point-impulse-wrench-coupling` already names.
    """
    projectile_impulse_n_s: float
    gas_impulse_n_s: float
    vent_impulse_n_s: float
    direction: tuple                      # unit vector the shot goes out along
    bore_offset_m: tuple = (0.0, 0.0, 0.0)   # bore axis from the mount's pivot

    @property
    def net_impulse_n_s(self) -> float:
        """Backwards is positive. A brake subtracts; a recoilless vent
        can take it through zero and push the gun forward."""
        return self.projectile_impulse_n_s + self.gas_impulse_n_s - self.vent_impulse_n_s

    @property
    def force_vector_n_s(self):
        """The impulse as a vector: opposite the way the round went."""
        import numpy as _np
        d = _np.asarray(self.direction, dtype=float)
        d = d / max(float(_np.linalg.norm(d)), 1e-12)
        return -d * self.net_impulse_n_s

    @property
    def moment_n_m_s(self):
        """r cross F about the pivot: the couple that tries to rotate
        the mount rather than shove it."""
        import numpy as _np
        return _np.cross(_np.asarray(self.bore_offset_m, dtype=float),
                         self.force_vector_n_s)

    def peak_force_n(self, duration_s: float = 0.002) -> float:
        """Unbraked, the whole impulse arrives in about two
        milliseconds. This is what the structure would see with no
        recoil system at all, and it is the number that makes the case
        for having one."""
        return abs(self.net_impulse_n_s) / max(duration_s, 1e-6)

    def describe(self) -> list[str]:
        total = self.projectile_impulse_n_s + self.gas_impulse_n_s
        gas_share = self.gas_impulse_n_s / max(total, 1e-9) * 100.0
        out = [f"  recoil: projectile {self.projectile_impulse_n_s:8.0f} N.s"
               f"  + gas {self.gas_impulse_n_s:7.0f} N.s ({gas_share:.0f} % of it)"]
        if self.vent_impulse_n_s:
            out.append(f"          vent returns {self.vent_impulse_n_s:8.0f} N.s forward "
                       f"-> net {self.net_impulse_n_s:8.0f} N.s "
                       f"({(1 - self.net_impulse_n_s / max(total, 1e-9)) * 100:.0f} % reduction)")
        out.append(f"          net {self.net_impulse_n_s:8.0f} N.s, "
                   f"{self.peak_force_n() / 1000:7.0f} kN if nothing absorbs it")
        m = self.moment_n_m_s
        if float(abs(m).sum()) > 1e-6:
            out.append(f"          and a moment of {float(__import__('numpy').linalg.norm(m)):7.0f} N.m.s "
                       f"about the pivot, because the bore is off the centre")
        return out


def recoil_of(calibre, *, direction=(0.0, 0.0, 1.0), charge_kg: float = 0.0,
              gas_velocity_ratio: float = 1.5, muzzle_brake_efficiency: float = 0.0,
              vent_fraction: float = 0.0, bore_offset_m=(0.0, 0.0, 0.0)) -> RecoilWrench:
    """The full recoil contribution of one shot.

    `muzzle_brake_efficiency` is the fraction of the GAS impulse the
    brake turns around -- 0.3 to 0.5 on a real brake, and it only ever
    acts on the gas, never on the projectile, which is why a brake can
    never remove more than about a third of total recoil.

    `vent_fraction` is for a genuinely vented system: the share of the
    whole propellant charge blown rearward on purpose. At around 0.8 a
    weapon becomes recoilless, and everything behind it becomes a very
    bad place to stand."""
    charge = charge_kg if charge_kg > 0.0 else calibre.mass_kg * 0.35
    projectile = calibre.mass_kg * calibre.muzzle_m_s
    gas = charge * gas_velocity_ratio * calibre.muzzle_m_s
    vent = gas * (muzzle_brake_efficiency + vent_fraction * 2.0)
    return RecoilWrench(projectile_impulse_n_s=projectile, gas_impulse_n_s=gas,
                        vent_impulse_n_s=vent, direction=tuple(direction),
                        bore_offset_m=tuple(bore_offset_m))


def cannon(bore_mm: float, *, muzzle_m_s: float = 0.0, name: str = "") -> Calibre:
    """A main gun of any bore, built from the bore itself.

    Real artillery scales predictably enough to derive rather than
    tabulate: shell mass goes roughly with the cube of the bore, a
    full-charge gun leaves the muzzle in the 700-1000 m/s band with the
    bigger bores slower, and the projectile is about five calibres long.
    So a bore is very nearly a complete specification.

    This exists because a station should be able to mount whatever size
    of gun the design calls for without somebody hand-entering a
    cartridge for each one. Ask for 30 mm and get an autocannon round;
    ask for 300 mm and get a shell weighing a third of a tonne, with the
    flight time and the mount loads that go with it.
    """
    bore_m = float(bore_mm) / 1000.0
    # 0.5 * (bore/0.04)^3 kg: a 40 mm comes out at 0.87 kg, which is the
    # real Bofors round, and the cube law carries it from there
    mass_kg = 0.87 * (bore_m / 0.040) ** 3
    if muzzle_m_s <= 0.0:
        # big guns are slower: a 40 mm leaves at about 1010 m/s and a
        # 300 mm at about 800, falling roughly with the sixth root
        muzzle_m_s = 1010.0 * (0.040 / max(bore_m, 1e-6)) ** (1.0 / 6.0)
        muzzle_m_s = max(600.0, min(1100.0, muzzle_m_s))
    return Calibre(
        name or f"{bore_mm:.0f}mm cannon", bore_m, mass_kg, muzzle_m_s,
        bore_m * 5.0, 1.10e9,
        # a heavier, longer shell carries better: drag falls with
        # sectional density, which rises with the bore
        drag_k_per_m=max(4e-5, 0.00018 * (0.040 / max(bore_m, 1e-6)) ** 0.5),
        note=f"parametric main gun, {bore_mm:.0f} mm bore")


def get_calibre(name: str) -> Calibre:
    key = name.strip().lower()
    if key in CALIBRES:
        return CALIBRES[key]
    aliases = {"9x19": "9mm", "40mm": "40mm l70", "bofors": "40mm l70", "30mm": "30mm chain", "45": ".45 acp", ".45": ".45 acp", "colt 45": ".45 acp", "500": ".500 s&w", "s&w 500": ".500 s&w",
               "50": ".50 bmg", ".50": ".50 bmg", "browning 50": ".50 bmg", "7.62": "7.62 nato", "7.6": "7.62 nato",
               "308": "7.62 nato", ".308": "7.62 nato", "556": "5.56", "5.56x45": "5.56", "582": "5.8x42", "5.8": "5.8x42",
               "338": ".338 lapua", "22": ".22 lr", "12 gauge": "12 ga slug", "slug": "12 ga slug", "357": ".357 mag"}
    if key in aliases:
        return CALIBRES[aliases[key]]
    raise KeyError(f"unknown calibre {name!r}; known: {', '.join(CALIBRES)}")


@dataclass
class Shot:
    calibre: str
    target: str | tuple            # graph node/edge identity, or an (x, y, z) point
    from_direction: tuple = (0.0, 0.3, 1.0)   # where the shooter stands, relative to the target
    range_m: float = 5.0
    rounds: int = 1
    spread_deg: float = 1.5        # the group's 1-sigma cone
    yaw_rad: float | None = None
    dwell_s: float = 0.0           # engine time before this shot


@dataclass
class ShotResult:
    shot: Shot
    round_index: int
    holes: list
    impacts: int
    log: list[str]
    energy_j: float


@dataclass
class FiringSquad:
    """An ordered itinerary. `execute` steps the sim `dwell_s` before
    each shot (so the engine keeps running between volleys), fires each
    round through a fresh ray from the shooter's position with the
    group's spread, and records what it did."""
    shots: list[Shot] = field(default_factory=list)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(1911))

    @classmethod
    def swiss_cheese(cls, target: str, calibre: str = "9mm", rounds: int = 12, spread_deg: float = 6.0,
                     from_direction=(0.0, 0.3, 1.0), range_m: float = 5.0) -> "FiringSquad":
        return cls([Shot(calibre, target, from_direction, range_m, rounds, spread_deg)])

    def execute(self, sim, ray_mesh_factory, graph: dict, dt: float = 1 / 60) -> list[ShotResult]:
        positions = {n["identity"]: np.asarray(n["reference_position"], dtype=float) for n in graph["nodes"]
                     if n.get("reference_position") is not None}
        for e in graph["edges"]:
            a, b = positions.get(e["a"]), positions.get(e["b"])
            if a is not None and b is not None and e["identity"] not in positions:
                positions[e["identity"]] = (a + b) / 2.0
        results: list[ShotResult] = []
        rm = None
        for shot in self.shots:
            if shot.dwell_s > 0.0:
                for _ in range(int(shot.dwell_s / dt)):
                    sim.step(dt)
                rm = None                       # the crank moved: fresh geometry
            cal = get_calibre(shot.calibre)
            tgt = positions[shot.target] if isinstance(shot.target, str) else np.asarray(shot.target, dtype=float)
            if isinstance(shot.target, str) and shot.target not in positions:
                results.append(ShotResult(shot, 0, [], 0, [f"no such target {shot.target}"], 0.0))
                continue
            stand = np.asarray(shot.from_direction, dtype=float); stand /= max(np.linalg.norm(stand), 1e-9)
            origin = tgt + stand * shot.range_m
            if rm is None:
                rm = ray_mesh_factory()
            for k in range(max(1, int(shot.rounds))):
                aim = tgt + self.rng.normal(0.0, math.radians(shot.spread_deg) * shot.range_m, 3)
                direction = aim - origin
                proj = cal.projectile(direction, shot.range_m, shot.yaw_rad)
                pen = rm.penetrate(Ray.from_points(origin, aim), proj.energy_j, cal.diameter_m, projectile=proj,
                                   fluid_by_part=sim.ballistic_fluid_for_part)
                rec = sim.apply_penetration(pen)
                results.append(ShotResult(shot, k, rec, len(pen.impacts),
                                          [f"{cal.name} #{k + 1}: {'; '.join(f'{i.split(chr(46))[-1]} {q.damage_mode}' for i, q in rec) or 'missed'}"]
                                          + list(pen.log), proj.energy_j))
        return results


def describe_catalogue() -> list[str]:
    return [f"{c.name:11s} {c.diameter_m * 1000:5.2f} mm {c.mass_kg * 1000:5.1f} g {c.muzzle_m_s:4.0f} m/s "
            f"{c.muzzle_energy_j / 1000:5.2f} kJ core {c.hardness_pa / 1e9:.2f} GPa  {c.note}" for c in CALIBRES.values()]
