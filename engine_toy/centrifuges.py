"""Centrifugal separators, sized from their own geometry.

A filter catches what is too big to pass it. A centrifuge throws out
what is too DENSE to stay, and that is a different question with a
different answer: it will take sub-micron soot and emulsified water
that no full-flow element can touch, and it will ignore a large, light
particle that a filter stops without trying. The two are complements,
not competitors, which is why a large marine engine carries both and
why one that carries a purifier is not condemned by an oil analysis
that would finish a truck.

WHY THIS EXISTS. The wear/soot ledger reported a Wartsila RTA96C at
nearly ten percent soot by mass over a drain interval -- past any
condemning limit -- because nothing in the model did what every large
marine installation really does: run its oil continuously through a
separator. That was a missing machine, not a wrong number.

PARAMETRIC, ON SIGMA THEORY. A separator is not rated by "efficiency".
It is rated by its equivalent clarification area, capital sigma, which
is the area of a gravity settling tank that would do the same job:

    Sigma = 2 pi omega^2 n (r2^3 - r1^3) / (3 g tan(theta))

for a disc stack of `n` discs between radii r1 and r2 at half-angle
theta. Throughput divided by sigma is an equivalent settling velocity,
and that velocity gives the CUT SIZE through Stokes' law:

    d50 = sqrt(18 mu Q / (delta_rho g 2 Sigma))

Everything follows from bowl geometry, speed, how hot and thus how thin
the oil is, and how fast it is being pushed through -- so a bigger bowl,
a faster bowl, hotter oil or a slower feed all sharpen the cut for the
real reason, and a machine can be specified the way a real one is.

CONTINUOUS AND SINGLE-ACTION are genuinely different machines, not a
setting:

  continuous      a self-cleaning disc stack ejects its sludge on a
                  timer, or a nozzle bowl discharges it constantly. It
                  runs unattended for as long as the fuel lasts, which
                  is why a ship has one.
  single-action   a solid bowl fills with what it has taken out and
                  then does NOTHING MORE. Its capacity is the whole
                  story: a bypass spinner on a truck engine is a real,
                  effective machine that is also a brick after its bowl
                  packs, and the only cure is stopping and scraping it.

The difference shows up as a hard edge in behaviour, not a coefficient:
a single-action unit's efficiency goes to zero when it is full.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

G = 9.80665
ATMOSPHERE_PA = 101_325.0

# SAE 40 at 40 C: 150 cSt x 870 kg/m3. The exponent is the two-point fit
# through 15 cSt at 100 C.
MU_40C_PA_S = 0.1305
VISCOSITY_EXPONENT = -13.14

# Densities that decide what a centrifuge can throw out of engine oil.
# Soot is the interesting one: individual particles are close to carbon
# black, and it is denser than the oil by enough to be thrown even when
# it is far too fine to filter.
OIL_DENSITY_KG_M3 = 870.0
SOOT_DENSITY_KG_M3 = 1800.0
WATER_DENSITY_KG_M3 = 1000.0
WEAR_METAL_DENSITY_KG_M3 = 7500.0


def oil_viscosity_pa_s(temp_k: float) -> float:
    """A real SAE 40's dynamic viscosity at temperature.

    Separation is inversely proportional to viscosity, so this is not a
    detail: the same machine on cold oil cuts far coarser than on hot,
    which is exactly why a purifier is fed from a heater and why running
    one on a cold engine is close to pointless. Two-point fit through
    the real values for a monograde at 40 C and 100 C."""
    # Two real points for a monograde SAE 40 and a power law through
    # them: 150 cSt at 40 C and 15 cSt at 100 C, times the oil's own
    # density to get dynamic viscosity. The exponent is steep because
    # oil viscosity really is -- it falls by a factor of ten over sixty
    # degrees, which is why a purifier is fed through a heater.
    t = max(250.0, float(temp_k))
    return MU_40C_PA_S * (t / 313.15) ** VISCOSITY_EXPONENT


@dataclass(frozen=True)
class CentrifugeClass:
    key: str
    label: str
    continuous: bool          # False = single-action: it fills and stops
    disc_count: int
    half_angle_deg: float
    sludge_capacity_frac: float   # of bowl volume it can hold before it is full
    self_driven: bool             # True = spun by the oil it is cleaning
    why: str
    # WHAT THE BOWL IS MADE OF, because it decides the one thing that
    # matters about starting it. A marine disc stack is a steel bowl
    # around a hundred-odd steel cones and it is genuinely heavy; a
    # bypass spinner's rotor is a light alloy or moulded shell with
    # nothing inside it. Sizing both as solid steel made a spinner's
    # rotor five times too heavy and gave it a six-minute run-up on a
    # drive that in reality brings it up promptly.
    bowl_density_kg_m3: float = 7850.0     # steel
    solidity: float = 0.42                 # how much of the bowl volume is metal


_CLASSES: tuple[CentrifugeClass, ...] = (
    CentrifugeClass(
        "disc-stack-self-cleaning", "self-cleaning disc stack", continuous=True,
        disc_count=120, half_angle_deg=45.0, sludge_capacity_frac=0.18, self_driven=False,
        why="the marine standard: a stack of conical discs cutting the settling distance "
            "to millimetres, in a bowl that opens on a timer and fires its sludge out "
            "without stopping. It is what lets a ship run one oil charge for months"),
    CentrifugeClass(
        "disc-stack-nozzle", "nozzle-discharge disc stack", continuous=True,
        disc_count=100, half_angle_deg=45.0, sludge_capacity_frac=0.05, self_driven=False,
        why="discharges continuously through fixed nozzles instead of opening the bowl: "
            "simpler and never interrupts, at the cost of throwing good oil out with the "
            "sludge the whole time"),
    CentrifugeClass(
        "solid-bowl-batch", "solid-bowl batch centrifuge", continuous=False,
        disc_count=60, half_angle_deg=45.0, sludge_capacity_frac=0.45, self_driven=False,
        why="no discharge mechanism at all: it packs its bowl with what it removes and "
            "must be stopped, opened and scraped. Cheap, effective, and entirely "
            "dependent on somebody actually doing that"),
    CentrifugeClass(
        "bypass-spinner", "oil-driven bypass spinner", continuous=False,
        disc_count=0, half_angle_deg=90.0, sludge_capacity_frac=0.55, self_driven=True,
        bowl_density_kg_m3=2700.0, solidity=0.16,
        why="spun by a pair of jets of the very oil it is cleaning, so it needs no drive "
            "and no controls -- the truck-engine answer. A solid bowl, so it is also a "
            "brick once packed, and it only ever sees the bypass flow"),
)

BY_KEY: dict[str, CentrifugeClass] = {c.key: c for c in _CLASSES}


def centrifuge_class(key: str) -> CentrifugeClass:
    c = BY_KEY.get(str(key))
    if c is None:
        raise KeyError(
            f"unknown centrifuge class {key!r}; declare one of: {', '.join(sorted(BY_KEY))}")
    return c


@dataclass
class Centrifuge:
    """One real machine, specified the way a real one is."""
    kind: str
    bowl_radius_m: float = 0.150
    inner_radius_m: float = 0.060
    bowl_height_m: float = 0.180
    speed_rpm: float = 8000.0
    # what turns it, and how big that is
    drive: str = "electric-belt"
    motor_kw: float = 7.5
    # oil-jet drives only: the nozzles that actually turn it
    nozzle_count: int = 2
    nozzle_diameter_m: float = 0.0010
    nozzle_arm_m: float = 0.045
    bowl_mass_kg: float = 0.0      # 0 = derive from the bowl's own volume
    # live state
    current_rpm: float = 0.0
    sludge_kg: float = 0.0
    running: bool = True
    discharged_kg: float = 0.0    # lifetime sludge ejected (continuous machines)
    discharges: int = 0

    @property
    def spec(self) -> CentrifugeClass:
        return centrifuge_class(self.kind)

    @property
    def omega_rad_s(self) -> float:
        return self.speed_rpm * 2.0 * math.pi / 60.0

    @property
    def g_force(self) -> float:
        """How many times gravity at the bowl wall -- the number these
        machines are advertised by, and a real one runs five to fifteen
        thousand."""
        return self.omega_rad_s ** 2 * self.bowl_radius_m / G

    @property
    def bowl_volume_m3(self) -> float:
        return math.pi * (self.bowl_radius_m ** 2 - self.inner_radius_m ** 2) * self.bowl_height_m

    @property
    def sludge_capacity_kg(self) -> float:
        return self.bowl_volume_m3 * self.spec.sludge_capacity_frac * SOOT_DENSITY_KG_M3

    @property
    def sigma_m2(self) -> float:
        """Equivalent clarification area: the gravity settling tank this
        machine stands in for. All of the geometry lands here."""
        s = self.spec
        w = self.omega_rad_s
        r2, r1 = self.bowl_radius_m, self.inner_radius_m
        if s.disc_count > 0:
            theta = math.radians(max(5.0, min(85.0, s.half_angle_deg)))
            return (2.0 * math.pi * w * w * s.disc_count
                    * (r2 ** 3 - r1 ** 3) / (3.0 * G * math.tan(theta)))
        # a plain bowl with no stack: the tubular-bowl form
        return (w * w * self.bowl_volume_m3) / (G * math.log(max(1.0001, r2 / r1)))

    @property
    def full_frac(self) -> float:
        cap = self.sludge_capacity_kg
        return 0.0 if cap <= 0.0 else min(1.0, self.sludge_kg / cap)

    @property
    def is_full(self) -> bool:
        return (not self.spec.continuous) and self.full_frac >= 1.0

    def cut_size_m(self, flow_m3_s: float, oil_temp_k: float = 363.15,
                   particle_density_kg_m3: float = SOOT_DENSITY_KG_M3) -> float:
        """The particle size this machine removes half of, right now.

        Anything well above it goes out with the sludge; anything well
        below stays in the oil. It is the one number that says what a
        separator is actually doing, and it moves with feed rate and oil
        temperature exactly as a real one does."""
        sigma = self.sigma_m2
        q = max(1e-12, float(flow_m3_s))
        if sigma <= 0.0:
            return 1.0
        d_rho = max(1.0, float(particle_density_kg_m3) - OIL_DENSITY_KG_M3)
        mu = oil_viscosity_pa_s(oil_temp_k)
        v_g = q / (2.0 * sigma)
        return math.sqrt(18.0 * mu * v_g / (d_rho * G))

    def grade_efficiency(self, particle_m: float, flow_m3_s: float,
                         oil_temp_k: float = 363.15,
                         particle_density_kg_m3: float = SOOT_DENSITY_KG_M3) -> float:
        """Fraction of particles of this size removed in one pass.

        The standard grade-efficiency form: exactly half at the cut size
        and rising with the square of size, because Stokes settling does.
        A single-action bowl that is full removes nothing at all."""
        if not self.running or self.is_full:
            return 0.0
        d50 = self.cut_size_m(flow_m3_s, oil_temp_k, particle_density_kg_m3)
        d = max(1e-12, float(particle_m))
        r = d / d50
        return (r * r) / (1.0 + r * r)

    def discharge(self) -> float:
        """Eject the bowl. A continuous machine does this on its own
        timer; a single-action one only does it when somebody opens it.
        Returns the sludge removed."""
        out = self.sludge_kg
        self.sludge_kg = 0.0
        return out


#: Representative particle size of each thing a centrifuge is asked to
#: pull out of engine oil. Soot is the hard case and the important one:
#: it agglomerates to around a micrometre, far below any full-flow
#: element, which is exactly why a filter cannot help and a centrifuge
#: can.
PARTICLE_SIZE_M = {
    "soot": 1.0e-6,
    "wear-fine": 2.0e-6,
    "wear-coarse": 2.0e-5,
    "water": 1.0e-5,
}


def clean_oil(unit: Centrifuge, ledger, flow_m3_s: float, dt_s: float,
              oil_mass_kg: float, oil_temp_k: float = 363.15,
              receiver: "SludgeReceiver | None" = None) -> dict:
    """Run a ledger's oil through this machine for `dt_s`.

    Takes from the FINE pools -- which is the whole point, because those
    are what the filter cannot reach and what therefore decides whether
    an oil is condemned. Whatever is removed goes into the bowl, and on
    a single-action machine it stays there and eventually stops it."""
    if not unit.running or unit.is_full or dt_s <= 0.0:
        return {"soot_g": 0.0, "metal_g": 0.0, "full_frac": unit.full_frac,
                "discharges": unit.discharges, "spilled_kg": 0.0}
    # how much of the charge passed through in this interval
    swept_kg = max(0.0, float(flow_m3_s)) * OIL_DENSITY_KG_M3 * float(dt_s)
    pass_frac = min(1.0, swept_kg / max(1e-6, float(oil_mass_kg)))

    eta_soot = unit.grade_efficiency(PARTICLE_SIZE_M["soot"], flow_m3_s, oil_temp_k,
                                     SOOT_DENSITY_KG_M3)
    eta_metal = unit.grade_efficiency(PARTICLE_SIZE_M["wear-fine"], flow_m3_s, oil_temp_k,
                                      WEAR_METAL_DENSITY_KG_M3)

    soot_taken = ledger.soot_fine_g * pass_frac * eta_soot
    ledger.soot_fine_g -= soot_taken
    metal_taken = float(ledger.fine.sum()) * pass_frac * eta_metal
    if metal_taken > 0.0:
        ledger.fine *= (1.0 - pass_frac * eta_metal)
    removed = soot_taken + metal_taken
    # it is not lost from the machine, it is in the bowl -- the ledger
    # counts it as captured, exactly like metal on a filter element
    ledger.captured_g += removed
    unit.sludge_kg += removed / 1000.0
    # A SELF-CLEANING BOWL CLEANS ITSELF. That is the whole difference
    # between the two families and it has to be in the behaviour, not in
    # a comment: a continuous machine reaches its capacity, fires the
    # sludge out and carries straight on, while a single-action bowl
    # reaches the same point and stops separating until somebody opens
    # it. Without this the "continuous" machines silently packed three
    # hundred kilograms into a twenty-seven kilogram bowl and kept
    # working, which is neither of the two real behaviours.
    spilled = 0.0
    if unit.spec.continuous and unit.sludge_kg >= unit.sludge_capacity_kg:
        spilled = discharge_to(unit, receiver)
    return {"soot_g": soot_taken, "metal_g": metal_taken,
            "full_frac": unit.full_frac, "discharges": unit.discharges,
            "spilled_kg": spilled}


# ---------------------------------------------------------------------
# THE EJECTION PORT, AND WHAT HAPPENS WHEN NOBODY EMPTIES THE TANK
# ---------------------------------------------------------------------
#
# A self-cleaning bowl does not make its sludge disappear -- it fires it
# out of a discharge port, hard, into a sludge tank. That tank is a real
# vessel with a real capacity, and the interesting case is the one where
# it is full: the ejection still happens, on its timer, because the bowl
# does not know or care, and what comes out goes onto whatever is in
# front of the port.
#
# So an overflow is not an error condition to be clamped away. It is an
# EMITTER at the ejection port, with real mass and a real direction, and
# it wets whatever the spray lands on -- which is the same thing a
# punctured line or an uncapped filler does, resolved by the same rays
# and deposited into the same tally. This module produces that emitter
# and stops there: what the wetted surfaces then LOOK like is a later
# wetting engine's business, and it reads `HoleEmitterField.wetted_kg`
# like any other source.


def ejection_emitter_spec(unit: "Centrifuge", part: str,
                          position=None, direction=None) -> dict:
    """The emitter a centrifuge's discharge port becomes when it spills.

    Returned as a plain specification so `HoleEmitterField` builds the
    emitter its own way -- the same shape `fouling.autogenous_port`
    returns for a corrosion hole, for the same reason: there is one kind
    of hole in this machine and everything that makes one describes it
    identically."""
    return dict(
        identity=f"{part}.ejection_port",
        part=part,
        circuit="oil",
        fluid="sludge",
        position=tuple(float(v) for v in (position or (0.0, 0.0, 0.0))),
        # a discharge port fires outward and slightly down, which is
        # where the sludge ends up on a real engine room floor
        direction=tuple(float(v) for v in (direction or (0.0, -0.4, 1.0))),
        radius_m=max(0.004, unit.bowl_radius_m * 0.06),
        through=True,
        kind="ejection-overflow",
    )


@dataclass
class SludgeReceiver:
    """The tank a discharge is fired into.

    Capacity is the whole point: a bowl that ejects on a timer keeps
    ejecting whether or not there is anywhere for it to go."""
    capacity_kg: float = 200.0
    held_kg: float = 0.0
    overflowed_kg: float = 0.0

    @property
    def full_frac(self) -> float:
        return 0.0 if self.capacity_kg <= 0.0 else min(1.0, self.held_kg / self.capacity_kg)

    @property
    def is_full(self) -> bool:
        return self.held_kg >= self.capacity_kg

    def accept(self, kg: float) -> float:
        """Take what the bowl just fired. Returns what would not fit --
        which is what comes out of the port and onto the floor."""
        g = max(0.0, float(kg))
        room = max(0.0, self.capacity_kg - self.held_kg)
        took = min(g, room)
        self.held_kg += took
        spill = g - took
        self.overflowed_kg += spill
        return spill

    def empty(self) -> float:
        out = self.held_kg
        self.held_kg = 0.0
        return out


def discharge_to(unit: "Centrifuge", receiver: "SludgeReceiver | None") -> float:
    """Fire the bowl into its receiver. Returns the mass that overflowed.

    With no receiver declared at all, everything the bowl ejects
    overflows -- which is the honest answer for an installation that
    never plumbed the discharge anywhere, and a real way to make a mess."""
    ejected = unit.discharge()
    unit.discharged_kg += ejected
    unit.discharges += 1
    if receiver is None:
        return ejected
    return receiver.accept(ejected)


# Ejected sludge is not oil: it is oil plus the soot, water and metal
# that were taken out of it, and it is denser and thicker than either.
# Declared as its own fluid in fluids.py so the density is the SAME
# number on both sides of the kilograms/litres seam below.
from fluids import DENSITY_KG_M3 as _FLUID_DENSITY
SLUDGE_DENSITY_KG_M3 = _FLUID_DENSITY.get("sludge", 1100.0)


def sludge_litres(kg: float) -> float:
    """Kilograms of ejected sludge as the litres an emitter expects.

    THE EMITTER SIDE WORKS IN LITRES AND THIS MODULE WORKS IN KILOGRAMS,
    which is exactly the kind of seam that quietly loses ten percent of
    a spill and is never noticed. Convert here, once, rather than at
    every call site."""
    return max(0.0, float(kg)) / SLUDGE_DENSITY_KG_M3 * 1000.0


def spill_onto(field, unit: "Centrifuge", part: str, kg: float,
               position=None, direction=None):
    """Hand an ejection-port overflow to the emitter field as wetting.

    Creates the port's emitter if it is not there yet, then deposits the
    spill through it, so the spray resolves against real geometry and
    lands on real surfaces like every other source. Returns the emitter,
    for a caller that wants to resolve its spray immediately."""
    from hole_emitters import HoleEmitter
    spec = ejection_emitter_spec(unit, part, position, direction)
    em = next((e for e in field.emitters if e.identity == spec["identity"]), None)
    if em is None:
        em = HoleEmitter(**spec)
        field.emitters.append(em)
    field.deposit_spray(em, sludge_litres(kg))
    return em


# ---------------------------------------------------------------------
# WHAT SPINS IT
# ---------------------------------------------------------------------
#
# Almost every industrial separator is turned by an ELECTRIC MOTOR, and
# the exception is worth naming: a bypass spinner on a truck engine is
# driven by jets of the very oil it is cleaning, which is why it has no
# motor, no starter and no controls. Everything on a station has a motor.
#
# The drive is not a detail, because a separator bowl is a serious
# rotating mass and starting one is the hard part of owning it. A 300 mm
# bowl assembly is well over a hundred kilograms at a radius where that
# counts, so its stored energy at speed runs into megajoules and a motor
# sized for the steady load cannot possibly accelerate it quickly. Real
# machines take three to eight minutes to come up, through a friction
# clutch that slips the whole way so the motor is never stalled against
# a bowl that will not move. That run-up is a real operational fact --
# you do not "just turn one on" -- and it falls straight out of the
# bowl's own polar inertia rather than being a number anyone picks.
#
# Once at speed the motor is only holding it there against bearing
# friction and windage, which is a fraction of what starting took.

DRIVE_KINDS = {
    "electric-belt": "electric motor through a flat belt and friction clutch",
    "electric-direct": "close-coupled electric motor with an integral clutch",
    "oil-jet": "spun by jets of the oil it is cleaning -- no motor at all",
    "air-turbine": "shop-air turbine, for a unit in a place with no power",
}

#: Real bowl assembly mass as a fraction of what a solid bowl of that
#: size would weigh -- a disc stack is mostly hollow, but the stack
#: itself is a hundred-odd steel cones.
BOWL_SOLIDITY = 0.42
BOWL_STEEL_DENSITY = 7850.0
#: Windage and bearing drag at speed, as a fraction of rated motor power.
#: A separator at speed is a light load; it is the START that is hard.
STEADY_LOAD_FRAC = 0.30
#: A friction clutch cannot pass more than this share of motor torque
#: while it slips, which is what stops the motor stalling on run-up.
CLUTCH_TORQUE_FRAC = 0.85


def _bowl_inertia(unit: "Centrifuge") -> float:
    """The bowl's real polar inertia, from its own geometry.

    A rim-weighted rotating assembly, sized through the same shape
    vocabulary every other rotating mass in this machine uses -- so a
    centrifuge cannot be a massless drive component any more than a
    flywheel can."""
    from rotating_inertia import polar_inertia
    spec = unit.spec
    mass = unit.bowl_mass_kg or (unit.bowl_volume_m3 * spec.solidity * spec.bowl_density_kg_m3)
    return polar_inertia("rim-weighted-flywheel", max(0.1, mass), unit.bowl_radius_m)


def stored_energy_j(unit: "Centrifuge") -> float:
    """Kinetic energy in the bowl at its running speed.

    Worth knowing for its own sake: this is what has to be put in to
    start it, what has to come out to stop it, and what is released if
    an out-of-balance bowl lets go."""
    return 0.5 * _bowl_inertia(unit) * unit.omega_rad_s ** 2


# ---- the oil-jet drive, which is a real hydraulic machine ----------
#
# A bypass spinner has NO MOTOR AND NO RATED SPEED. Oil at gallery
# pressure is fed up the middle of the rotor and leaves through two
# small tangential nozzles at the bottom; the reaction of those jets is
# the only thing turning it. So its speed is a RESULT -- of supply
# pressure, nozzle size, arm radius and how thick the oil is -- and not
# something declared. Take the supply away and it stops, and stops
# cleaning; run it on a worn engine with low hot-oil pressure and it
# turns slower and cuts coarser, which is exactly the case where you
# wanted it most.
#
# Jet velocity is Bernoulli through the orifice, the thrust of each jet
# is its own momentum flux, and the torque is that thrust on the arm:
#
#     v = Cd sqrt(2 dp / rho),  F = mdot v,  T = n F r_arm
#
# It settles where that balances windage and bearing drag, which rise
# with the square of speed.
NOZZLE_DISCHARGE_COEFF = 0.80
#: Drag torque coefficient, T_drag = k w^2, calibrated so a real two-
#: nozzle 1 mm spinner on a 45 mm arm reaches about 6000 rpm at 4 bar --
#: the figure these are sold on.
SPINNER_DRAG_K = 9.2e-8
#: Bearing drag does not vanish with speed the way windage does -- it is
#: roughly constant Coulomb friction, and it is the only thing that
#: actually brings a rotor to a STOP. With windage alone a coasting
#: rotor asymptotes towards zero and never reaches it, which is not what
#: a spinner does when you shut the oil off.
SPINNER_BEARING_TORQUE_NM = 2.0e-4


def jet_drive_torque_nm(unit: "Centrifuge", supply_pa: float,
                        oil_density_kg_m3: float = OIL_DENSITY_KG_M3) -> float:
    """Torque the nozzles make at this supply pressure."""
    dp = max(0.0, float(supply_pa) - ATMOSPHERE_PA)
    if dp <= 0.0 or unit.nozzle_count <= 0:
        return 0.0
    area = math.pi * (unit.nozzle_diameter_m * 0.5) ** 2
    v = NOZZLE_DISCHARGE_COEFF * math.sqrt(2.0 * dp / oil_density_kg_m3)
    mdot = oil_density_kg_m3 * area * v * NOZZLE_DISCHARGE_COEFF
    return unit.nozzle_count * mdot * v * unit.nozzle_arm_m


def jet_flow_m3_s(unit: "Centrifuge", supply_pa: float,
                  oil_density_kg_m3: float = OIL_DENSITY_KG_M3) -> float:
    """Oil the nozzles pass -- which is also the flow being cleaned, and
    the flow the engine's gallery has to give up to run it."""
    dp = max(0.0, float(supply_pa) - ATMOSPHERE_PA)
    if dp <= 0.0 or unit.nozzle_count <= 0:
        return 0.0
    area = math.pi * (unit.nozzle_diameter_m * 0.5) ** 2
    v = NOZZLE_DISCHARGE_COEFF * math.sqrt(2.0 * dp / oil_density_kg_m3)
    return unit.nozzle_count * area * v * NOZZLE_DISCHARGE_COEFF


def jet_equilibrium_rpm(unit: "Centrifuge", supply_pa: float) -> float:
    """Where it settles: drive torque equal to drag."""
    t = jet_drive_torque_nm(unit, supply_pa)
    if t <= 0.0:
        return 0.0
    w = math.sqrt(t / SPINNER_DRAG_K)
    return w * 60.0 / (2.0 * math.pi)


def spin_up_time_s(unit: "Centrifuge", supply_pa: float = 0.0) -> float:
    """How long from stopped to running speed.

    Energy in the bowl divided by what the drive can actually deliver. A
    real marine purifier is several minutes on its motor; an oil-jet
    spinner is seconds, but only because its rotor is tiny -- and only
    if it is given pressure."""
    if unit.drive == "oil-jet":
        t = jet_drive_torque_nm(unit, supply_pa)
        if t <= 0.0:
            return float("inf")          # no supply, no spin, ever
        w = jet_equilibrium_rpm(unit, supply_pa) * 2.0 * math.pi / 60.0
        inertia = _bowl_inertia(unit)
        # mean accelerating torque is roughly a third of stall once drag
        # is taken off through the run-up
        return inertia * w / max(1e-9, t * 0.33)
    power_w = max(1.0, unit.motor_kw * 1000.0) * CLUTCH_TORQUE_FRAC
    return stored_energy_j(unit) / power_w


def steady_power_w(unit: "Centrifuge") -> float:
    """Draw at speed: bearing drag and windage, not the starting load."""
    if not unit.running or unit.drive == "oil-jet":
        return 0.0
    return unit.motor_kw * 1000.0 * STEADY_LOAD_FRAC


def step_drive(unit: "Centrifuge", dt: float, powered: bool = True,
               supply_pa: float = 0.0) -> dict:
    """Advance the bowl's speed for `dt`, and report what it drew.

    Below running speed it is accelerating and pulling its full clutch-
    limited load; at speed it settles back to the steady draw. Cut the
    power and it coasts down on its own inertia against the same drag,
    which takes far longer than anyone expects and is why a separator is
    not something you stop and restart casually."""
    target = unit.speed_rpm
    if unit.drive == "oil-jet":
        # NO SUPPLY, NO SPIN. Its speed is whatever its own jets can hold
        # against drag, and it accelerates there on the net of the two.
        inertia = _bowl_inertia(unit)
        w = unit.current_rpm * 2.0 * math.pi / 60.0
        drive_t = jet_drive_torque_nm(unit, supply_pa) if powered else 0.0
        drag = SPINNER_DRAG_K * w * w + (SPINNER_BEARING_TORQUE_NM if w > 0.0 else 0.0)
        w = max(0.0, w + (drive_t - drag) / max(inertia, 1e-9) * dt)
        unit.current_rpm = w * 60.0 / (2.0 * math.pi)
        eq = jet_equilibrium_rpm(unit, supply_pa) if powered else 0.0
        unit.running = unit.current_rpm > 1.0
        return {"rpm": unit.current_rpm, "power_w": 0.0,
                "at_speed": eq > 0.0 and unit.current_rpm >= eq * 0.95,
                "equilibrium_rpm": eq,
                "flow_m3_s": jet_flow_m3_s(unit, supply_pa) if powered else 0.0}
    inertia = _bowl_inertia(unit)
    w = unit.current_rpm * 2.0 * math.pi / 60.0
    w_target = target * 2.0 * math.pi / 60.0
    if powered and w < w_target:
        p = max(1.0, unit.motor_kw * 1000.0) * CLUTCH_TORQUE_FRAC
        # accelerate on energy, which behaves properly from a standstill
        # where a constant-torque form would divide by zero speed
        ke = 0.5 * inertia * w * w + p * dt
        w = min(w_target, math.sqrt(max(0.0, 2.0 * ke / max(inertia, 1e-9))))
        draw = unit.motor_kw * 1000.0
    elif powered:
        w = w_target
        draw = steady_power_w(unit)
    else:
        # coasting down against the same drag that the steady load held
        # it against
        drag_w = unit.motor_kw * 1000.0 * STEADY_LOAD_FRAC
        ke = max(0.0, 0.5 * inertia * w * w - drag_w * dt)
        w = math.sqrt(2.0 * ke / max(inertia, 1e-9))
        draw = 0.0
    unit.current_rpm = w * 60.0 / (2.0 * math.pi)
    at_speed = unit.current_rpm >= target * 0.98
    unit.running = at_speed
    return {"rpm": unit.current_rpm, "power_w": draw, "at_speed": at_speed}


# ---------------------------------------------------------------------
# HYDROCYCLONES: MULTI-TRACK, AND NOTHING MOVES
# ---------------------------------------------------------------------
#
# A hydrocyclone separates by making the fluid spin instead of spinning
# the fluid: tangential inlet, conical body, heavy phase down the wall
# to the underflow, light phase up the middle. There is no bowl, no
# motor, no bearing and no run-up -- only a pressure drop. That makes it
# the complement to everything above: it cannot be stopped by cutting
# power, cannot wear out a bearing, and starts working the instant it
# has flow.
#
# MULTI-TRACK IS NOT AN ODDITY, IT IS HOW THEY ARE USED. A cyclone's cut
# size falls with its DIAMETER, and its throughput falls with the square
# of it, so one big cyclone is a coarse cyclone. The way to get a fine
# cut at a useful flow is to run many small ones in parallel off a
# common manifold -- a desilter bank, a multiclone, a cyclone bundle --
# and that is standard practice in drilling mud, produced water, pulp
# and boiler ash. The disc stack above is the same idea in a different
# geometry: a hundred and twenty parallel channels, each a short
# settling path, which is exactly why it cuts where it does.
#
# WHAT A CYCLONE CANNOT DO, and the reason a plant carries both. Cut
# size goes as the square root of viscosity, so the same bank that cuts
# a few micrometres on water cuts four times coarser on hot oil and
# twenty times coarser on cold. A hydrocyclone is excellent at water and
# grit and hopeless at soot; a disc stack takes the soot and is an
# expensive, motor-driven, wearing thing. Neither replaces the other.

#: Rietema-geometry Stokes number at the cut point, the standard design
#: correlation for a hydrocyclone. Real banks are specified from this.
CYCLONE_STK50 = 5.0e-4
#: Euler number for the same geometry: pressure drop as a multiple of
#: inlet velocity head. It is what makes a cyclone bank a pressure
#: consumer rather than a power consumer.
CYCLONE_EULER = 1200.0


@dataclass
class CycloneBank:
    """N small cyclones on one manifold. No moving parts anywhere."""
    cyclone_diameter_m: float = 0.010
    count: int = 24
    label: str = ""

    @property
    def area_m2(self) -> float:
        return math.pi * (self.cyclone_diameter_m * 0.5) ** 2

    def per_cyclone_flow_m3_s(self, total_flow_m3_s: float) -> float:
        return max(0.0, float(total_flow_m3_s)) / max(1, self.count)

    def inlet_velocity_m_s(self, total_flow_m3_s: float) -> float:
        return self.per_cyclone_flow_m3_s(total_flow_m3_s) / max(1e-9, self.area_m2)

    def pressure_drop_pa(self, total_flow_m3_s: float,
                         density_kg_m3: float = OIL_DENSITY_KG_M3) -> float:
        """What it costs to run. Rises with the SQUARE of flow, which is
        why a bank is sized by adding cyclones rather than by pushing
        harder through the ones it has."""
        v = self.inlet_velocity_m_s(total_flow_m3_s)
        return 0.5 * CYCLONE_EULER * density_kg_m3 * v * v

    def cut_size_m(self, total_flow_m3_s: float, oil_temp_k: float = 363.15,
                   particle_density_kg_m3: float = SOOT_DENSITY_KG_M3,
                   fluid_density_kg_m3: float = OIL_DENSITY_KG_M3,
                   viscosity_pa_s: float | None = None) -> float:
        """The cut size this bank achieves at this flow.

        Smaller cyclones cut finer; more of them share the flow so each
        runs slower, which cuts finer again. Both levers are real and
        both are why banks look the way they do."""
        v = self.inlet_velocity_m_s(total_flow_m3_s)
        if v <= 0.0:
            return float("inf")
        mu = viscosity_pa_s if viscosity_pa_s is not None else oil_viscosity_pa_s(oil_temp_k)
        d_rho = max(1.0, float(particle_density_kg_m3) - float(fluid_density_kg_m3))
        return math.sqrt(CYCLONE_STK50 * 18.0 * mu * self.cyclone_diameter_m / (d_rho * v))

    def grade_efficiency(self, particle_m: float, total_flow_m3_s: float,
                         oil_temp_k: float = 363.15,
                         particle_density_kg_m3: float = SOOT_DENSITY_KG_M3,
                         fluid_density_kg_m3: float = OIL_DENSITY_KG_M3,
                         viscosity_pa_s: float | None = None) -> float:
        d50 = self.cut_size_m(total_flow_m3_s, oil_temp_k, particle_density_kg_m3,
                              fluid_density_kg_m3, viscosity_pa_s)
        if not math.isfinite(d50) or d50 <= 0.0:
            return 0.0
        r = max(1e-12, float(particle_m)) / d50
        return (r * r) / (1.0 + r * r)


# ---------------------------------------------------------------------
# DRYING OIL: VACUUM DEHYDRATION
# ---------------------------------------------------------------------
#
# A vacuum pump on its own does not dry oil. What dries oil is a vacuum
# DEHYDRATOR, which is three things at once and needs all three:
#
#   heat      to about 50-70 C, which raises water's vapour pressure by
#             an order of magnitude and does almost nothing to the oil's
#   vacuum    down to around 20-30 mbar absolute, well below water's
#             saturation pressure at that temperature, so the water
#             boils out of the oil at a temperature that does not harm it
#   surface   the oil is dispersed over packing or media, because this
#             is mass transfer and mass transfer needs area. A vacuum
#             over a still pool of oil dries the top of the pool.
#
# WHY IT EARNS ITS PLACE ALONGSIDE THE SEPARATORS. Water in oil comes in
# three forms and they are not interchangeable: FREE water settles or
# throws out in a centrifuge; EMULSIFIED water mostly does too, given a
# good enough machine; DISSOLVED water does neither, because it is in
# solution and there is nothing to throw. A disc stack cannot take it, a
# cyclone certainly cannot, a filter never could -- and dissolved water
# is what rusts bearings and strips additives. Vacuum dehydration is the
# only one of these machines that removes it, which is exactly why a
# plant that has centrifuges still buys one.

#: Oil holds this much water in true solution at room temperature before
#: any of it appears as free water -- a real figure for a mineral oil,
#: and the reason "no visible water" means very little.
OIL_SATURATION_PPM_20C = 300.0


def water_vapour_pressure_pa(temp_k: float) -> float:
    """Saturation pressure of water, by Antoine.

    This is the whole driving force: heat the oil and this climbs
    steeply while the oil's own vapour pressure stays negligible, which
    is what lets the water leave and the oil stay."""
    t_c = max(0.0, float(temp_k) - 273.15)
    mmhg = 10.0 ** (8.07131 - 1730.63 / (233.426 + t_c))
    return mmhg * 133.322


def oil_saturation_ppm(temp_k: float) -> float:
    """How much water this oil can hold dissolved at temperature.

    Solubility rises with heat, which is the trap in a hot sample: oil
    that looks clear at ninety degrees drops free water in the sump
    overnight."""
    return OIL_SATURATION_PPM_20C * (water_vapour_pressure_pa(temp_k)
                                     / water_vapour_pressure_pa(293.15)) ** 0.5


@dataclass
class VacuumDehydrator:
    """A real vacuum dehydration unit.

    Sized the way one is: how much oil it can pass, how hard a vacuum
    its pump holds, how hot it runs the oil, and how much dispersal area
    the tower gives it."""
    flow_m3_s: float = 2.0 / 3600.0        # a small industrial unit
    vacuum_pa: float = 2_500.0             # 25 mbar absolute
    oil_temp_k: float = 338.15             # 65 C
    tower_area_m2: float = 6.0             # dispersed surface in the packing
    heater_kw: float = 9.0
    pump_kw: float = 1.5
    running: bool = True

    @property
    def driving_pressure_pa(self) -> float:
        """How far below water's saturation pressure the chamber is.

        At or above it nothing happens: a vacuum that is not deep enough
        for the temperature is a pump making noise."""
        return max(0.0, water_vapour_pressure_pa(self.oil_temp_k) - self.vacuum_pa)

    @property
    def power_w(self) -> float:
        if not self.running:
            return 0.0
        return (self.heater_kw + self.pump_kw) * 1000.0

    def removal_fraction_per_pass(self) -> float:
        """Share of the water in the oil taken out in one pass.

        Mass-transfer limited: proportional to dispersed area and to the
        driving pressure, and inversely to how fast the oil is pushed
        through. A real unit takes a good fraction per pass and is run
        on recirculation for hours rather than once through."""
        if not self.running or self.driving_pressure_pa <= 0.0:
            return 0.0
        # dimensionless transfer group; the coefficient is set so a real
        # tower at 65 C and 25 mbar clears roughly half its water per
        # pass, which is what these are quoted at
        ntu = (MASS_TRANSFER_COEFF * self.tower_area_m2
               * self.driving_pressure_pa / max(1e-9, self.flow_m3_s))
        return 1.0 - math.exp(-min(20.0, ntu))

    def step(self, dt: float, charge_l: float, water_ppm: float) -> dict:
        """Run the charge through for `dt`. Returns the new water content.

        It cannot dry below what the oil holds in solution AT THE VACUUM
        it is pulling -- there is a floor, and a unit that cannot reach
        it is not broken, it is simply not pulling hard enough."""
        if not self.running or dt <= 0.0 or charge_l <= 0.0:
            return {"water_ppm": water_ppm, "removed_g": 0.0, "power_w": 0.0}
        passes = self.flow_m3_s * 1000.0 * dt / charge_l
        eff = self.removal_fraction_per_pass()
        floor = OIL_SATURATION_PPM_20C * (self.vacuum_pa
                                          / max(1.0, water_vapour_pressure_pa(self.oil_temp_k)))
        above = max(0.0, water_ppm - floor)
        remaining = above * math.exp(-eff * passes)
        new_ppm = floor + remaining
        removed_g = (water_ppm - new_ppm) * 1e-6 * charge_l * OIL_DENSITY_KG_M3
        return {"water_ppm": new_ppm, "removed_g": max(0.0, removed_g),
                "power_w": self.power_w, "per_pass": eff}


#: Mass-transfer coefficient for a packed dehydration tower, in the
#: dimensionless group above. Calibrated so a real 6 m2 tower at 65 C
#: and 25 mbar removes roughly half the water per pass -- which is what
#: these are quoted at, and why they are run on recirculation for hours
#: rather than passed through once. An earlier value cleared the whole
#: charge in a single pass, which would make the recirculation loop
#: every real unit has pointless.
MASS_TRANSFER_COEFF = 2.9e-9


# ---------------------------------------------------------------------
# WHERE THE WATER GOES
# ---------------------------------------------------------------------
#
# A dehydrator that made water vanish would be missing its second half.
# The vapour leaving the tower has to be condensed BEFORE the vacuum
# pump, not after and not never: water vapour through a vacuum pump
# condenses in the pump oil and destroys it, which is why every real
# unit is tower -> condenser -> condensate receiver -> pump, and why
# the receiver is a vessel somebody empties.
#
# THE RECLAIMED WATER IS NOT CLEAN. It carries oil mist the demister did
# not catch, the acids that oxidised oil makes, and -- very often -- the
# glycol that was the reason the oil needed drying in the first place.
# So it is an oily-water waste stream: worth collecting because it must
# be collected, not because it is useful. A plant with a separator can
# recover the oil out of it; a plant without one tanks it and disposes
# of it, and either way somebody deals with it.

#: Oil carried over into the condensate as mist the demister missed.
#: A real tower with a demister runs a few hundred ppm; without one it
#: is percent-level and the "water" is visibly an emulsion.
CONDENSATE_OIL_PPM_DEMISTED = 400.0
CONDENSATE_OIL_PPM_BARE = 15_000.0
#: Latent heat of water at the low pressure a dehydrator runs at -- what
#: the condenser actually has to take out.
WATER_LATENT_HEAT_J_KG = 2.40e6


@dataclass
class CondensateReceiver:
    """The pot the reclaimed water collects in, before the pump."""
    capacity_kg: float = 25.0
    held_kg: float = 0.0
    oil_kg: float = 0.0
    overflowed_kg: float = 0.0

    @property
    def full_frac(self) -> float:
        return 0.0 if self.capacity_kg <= 0.0 else min(1.0, self.held_kg / self.capacity_kg)

    @property
    def is_full(self) -> bool:
        return self.held_kg >= self.capacity_kg

    @property
    def oil_frac(self) -> float:
        return 0.0 if self.held_kg <= 0.0 else self.oil_kg / self.held_kg

    def accept(self, water_kg: float, oil_kg: float) -> float:
        """Take the condensate. Returns what would not fit -- and a full
        receiver is not a stopped process, it is one carrying water into
        the vacuum pump, which is how these get wrecked."""
        total = max(0.0, water_kg) + max(0.0, oil_kg)
        room = max(0.0, self.capacity_kg - self.held_kg)
        took = min(total, room)
        if total > 0.0:
            self.oil_kg += oil_kg * (took / total)
        self.held_kg += took
        spill = total - took
        self.overflowed_kg += spill
        return spill

    def empty(self) -> tuple:
        out = (self.held_kg, self.oil_kg)
        self.held_kg = 0.0
        self.oil_kg = 0.0
        return out

    def need(self, identity: str, position=(0.0, 0.0, 0.0)):
        """A full pot wants emptying, and says so like everything else."""
        import servicing as sv
        if self.full_frac < 0.80:
            return None
        return sv.Need(
            identity=identity, want="drain", position=tuple(position),
            quantity=self.held_kg, unit="kg", matches="oily-water-waste",
            urgency=self.full_frac / 0.80, minutes=15.0, skill="operator",
            label=f"{identity} (oily water, {self.oil_frac * 100:.1f}% oil)",
            why="a full condensate pot carries water on into the vacuum pump, which "
                "is how a dehydrator destroys its own pump while looking like it is "
                "working")


def reclaim_condensate(dehydrator: "VacuumDehydrator", removed_water_g: float,
                       receiver: "CondensateReceiver | None" = None,
                       demisted: bool = True) -> dict:
    """Condense what the tower drove off, and account for it.

    Returns the water and the oil that came over with it, the condenser
    duty it took, and whatever overflowed. Nothing disappears: the water
    that left the oil is now in the pot, dirty."""
    water_kg = max(0.0, float(removed_water_g)) / 1000.0
    ppm = CONDENSATE_OIL_PPM_DEMISTED if demisted else CONDENSATE_OIL_PPM_BARE
    oil_kg = water_kg * ppm * 1e-6
    duty_j = water_kg * WATER_LATENT_HEAT_J_KG
    spill = receiver.accept(water_kg, oil_kg) if receiver is not None else water_kg + oil_kg
    return {"water_kg": water_kg, "oil_kg": oil_kg, "condenser_duty_j": duty_j,
            "overflow_kg": spill,
            "oil_frac": (oil_kg / (water_kg + oil_kg)) if (water_kg + oil_kg) > 0 else 0.0}


# ---------------------------------------------------------------------
# GAS CYCLONES: the pre-cleaner on an engine intake
# ---------------------------------------------------------------------
#
# A cyclone works on gas as happily as on liquid and for the same reason,
# but the ARITHMETIC ABOVE DOES NOT TRANSFER, and quietly reusing it
# would be wrong by a long way. The hydrocyclone correlations here --
# Rietema's Stokes number, the Euler number for pressure drop -- are
# fitted to liquid-continuous systems where the particle and the fluid
# have densities within a factor of three of each other and the device is
# fed by a pump. In a gas cyclone the density ratio is two thousand to
# one, the flow is fed by the engine's own suction, and the empirical
# constants are different numbers from different experiments.
#
# So gas cyclones get Lapple's equation, which is what the industry
# actually sizes intake pre-cleaners with:
#
#     d50 = sqrt( 9 mu W / (2 pi Ne v (rho_p - rho_g)) )
#
# W is the inlet width, Ne the number of effective turns the gas makes
# before it reaches the outlet, and v the inlet velocity. Every one of
# those is a thing you can point at on the part.
#
# WHY A PRE-CLEANER IS THE HIGHEST-VALUE PART ON A DESERT INTAKE. Paper
# elements do not fail gradually in dust; they fill, and the time to fill
# is the dust concentration times the airflow divided by the element's
# holding capacity. A big engine moves half a cubic metre of air a second
# and desert air off-road carries tens of milligrams a cubic metre, so a
# six-hundred-gram element has a life measured in HOURS. A cyclone with
# no moving parts that throws away nine tenths of that dust before it
# reaches the paper multiplies the interval by ten, and the only thing it
# costs is a kilopascal of restriction.
#
# AND IT CUTS COARSE, WHICH IS EXACTLY RIGHT. A cyclone is poor at fine
# particles and excellent at big ones -- efficiency goes as the square of
# size relative to the cut. That is a bad filter and a perfect
# PRE-filter, because the coarse fraction is almost all of the dust BY
# MASS and it is what fills an element, while the fine fraction is what
# actually reaches the bore and is the paper's job.
#
# THE EJECTOR IS THE MAINTENANCE ITEM NOBODY LOGS. Separated dust
# collects in a cup and leaves through a rubber duckbill -- a vacuator
# valve -- held shut by intake depression and flapped open by pulsation.
# It hardens with heat and age, stops ejecting, the cup fills, and then
# the cyclone re-entrains everything it caught and delivers it to the
# element in a lump. A pre-cleaner that has stopped ejecting is worse
# than no pre-cleaner at all.

#: Silica dust. Not soot: an order of magnitude denser and far larger.
MINERAL_DUST_DENSITY_KG_M3 = 2650.0
AIR_DENSITY_KG_M3 = 1.204
AIR_VISCOSITY_PA_S = 1.825e-5

#: Dust loadings that actually occur, in mg per cubic metre of air.
DUST_LOADING_MG_M3: dict[str, float] = {
    "clean-road": 1.0,
    "dirt-road": 15.0,
    "desert-offroad": 50.0,
    "convoy-trail": 250.0,
    "rotor-wash": 1500.0,
}


#: WHICH KIND OF PRE-CLEANER, because the two common ones are not the
#: same device and quoting one figure for both is how "85% efficient"
#: and "99% efficient" both end up describing a cyclone.
#:
#: An AXIAL-VANE unit turns the air with a fixed swirl vane and lets it
#: make barely two turns. It is cheap, it is nearly free in restriction,
#: and it is the thing bolted on top of a stack. It catches the big
#: stuff and that is all it promises.
#:
#: A TANGENTIAL MULTI-TUBE unit feeds each small tube through a real
#: tangential slot and gets five effective turns out of it. It is the
#: proper article, it cuts an order of magnitude finer, and it costs
#: several times the restriction.
@dataclass(frozen=True)
class CycloneConstruction:
    key: str
    label: str
    effective_turns: float
    inlet_width_frac: float
    inlet_height_frac: float
    loss_coefficient: float      # velocity heads of pressure drop
    #: Inlet velocity it is designed around. Cyclones have a working
    #: band: too slow and they stop separating, too fast and the
    #: restriction runs away as the square and re-entrainment starts.
    design_velocity_m_s: float
    why: str = ""


CYCLONE_CONSTRUCTIONS: dict[str, CycloneConstruction] = {
    "axial-vane": CycloneConstruction(
        "axial-vane", "axial vane pre-cleaner", 1.8, 0.35, 0.80, 4.0, 16.0,
        why="a swirl vane in a can. Two turns is not much and the cut is coarse, but "
            "it costs almost nothing in restriction and it removes most of the MASS, "
            "which is all a pre-filter is asked to do"),
    "tangential-multitube": CycloneConstruction(
        "tangential-multitube", "tangential multi-tube", 5.0, 0.25, 0.50, 8.0, 18.0,
        why="dozens of small tubes each fed through a real tangential slot. This is "
            "the one that gets quoted at ninety-something per cent, and the reason it "
            "can be is that it makes five turns instead of two"),
    "single-large": CycloneConstruction(
        "single-large", "single large cyclone", 5.0, 0.25, 0.50, 8.0, 18.0,
        why="the industrial shape, used where there is room for it. Cut size goes as "
            "the square root of diameter, so one big cyclone is always worse than the "
            "same area split into many small ones -- which is the entire argument for "
            "multi-tube"),
}


def cyclone_construction(key: str) -> CycloneConstruction:
    c = CYCLONE_CONSTRUCTIONS.get(str(key))
    if c is None:
        raise KeyError(f"unknown cyclone construction {key!r}; declared: "
                       f"{', '.join(sorted(CYCLONE_CONSTRUCTIONS))}")
    return c


@dataclass
class GasCyclone:
    """One cyclone tube, or a bank of them, sized by Lapple.

    Deliberately a separate class from CycloneBank rather than a mode of
    it: the correlations are different experiments and mixing them would
    let a liquid constant silently size an air part."""
    identity: str = "intake.precleaner"
    tube_diameter_m: float = 0.050
    count: int = 12
    construction: str = "tangential-multitube"
    #: Cup and duckbill. When this stops working the cyclone re-entrains.
    ejector_working: bool = True
    cup_capacity_g: float = 400.0
    cup_g: float = 0.0
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def spec(self) -> CycloneConstruction:
        return cyclone_construction(self.construction)

    @property
    def effective_turns(self) -> float:
        return self.spec.effective_turns

    @property
    def inlet_width_m(self) -> float:
        return self.tube_diameter_m * self.spec.inlet_width_frac

    @property
    def inlet_area_m2(self) -> float:
        return (self.tube_diameter_m * self.spec.inlet_width_frac
                * self.tube_diameter_m * self.spec.inlet_height_frac)

    def velocity_verdict(self, flow_m3_s: float) -> str:
        """Whether it is being run inside its working band.

        A cyclone sized by wishful thinking is not a slightly worse
        cyclone, it is a restriction. This says which."""
        v = self.inlet_velocity_m_s(flow_m3_s)
        d = self.spec.design_velocity_m_s
        if v < 0.4 * d:
            return f"too slow ({v:.0f} m/s): not enough swirl to separate"
        if v > 2.0 * d:
            return (f"OVERSPEED ({v:.0f} m/s vs {d:.0f} design): restriction goes as "
                    "the square and the catch re-entrains")
        return f"in band ({v:.0f} m/s)"

    def inlet_velocity_m_s(self, flow_m3_s: float) -> float:
        per = max(0.0, float(flow_m3_s)) / max(1, self.count)
        return per / max(1e-9, self.inlet_area_m2)

    def cut_size_m(self, flow_m3_s: float,
                   particle_density_kg_m3: float = MINERAL_DUST_DENSITY_KG_M3,
                   gas_density_kg_m3: float = AIR_DENSITY_KG_M3,
                   viscosity_pa_s: float = AIR_VISCOSITY_PA_S) -> float:
        """Lapple. The size this thing catches half of."""
        v = self.inlet_velocity_m_s(flow_m3_s)
        if v <= 0.0:
            return float("inf")
        d_rho = max(1.0, float(particle_density_kg_m3) - float(gas_density_kg_m3))
        return math.sqrt(9.0 * viscosity_pa_s * self.inlet_width_m
                         / (2.0 * math.pi * self.effective_turns * v * d_rho))

    def grade_efficiency(self, particle_m: float, flow_m3_s: float, **kw) -> float:
        """Lapple's grade curve: goes as the SQUARE of relative size.

        This is the shape that makes a cyclone a pre-filter and not a
        filter -- double the particle and you quadruple the term."""
        d50 = self.cut_size_m(flow_m3_s, **kw)
        if not math.isfinite(d50) or d50 <= 0.0:
            return 0.0
        r = float(particle_m) / d50
        return (r * r) / (1.0 + r * r)

    def mass_efficiency(self, flow_m3_s: float, distribution=None, **kw) -> float:
        """Efficiency BY MASS against a real dust size distribution.

        The single number people quote. It is high -- eighty to ninety
        per cent -- not because the cyclone is a good filter but because
        coarse dust is most of the mass."""
        dist = distribution if distribution is not None else COARSE_TEST_DUST
        num = sum(frac * self.grade_efficiency(size, flow_m3_s, **kw)
                  for size, frac in dist)
        return num / max(1e-9, sum(f for _, f in dist))

    def pressure_drop_pa(self, flow_m3_s: float,
                         gas_density_kg_m3: float = AIR_DENSITY_KG_M3) -> float:
        """Restriction, which the engine pays for in pumping work.

        Velocity heads times a loss coefficient set by the construction:
        an axial vane is four, a proper tangential tube is eight. Twice
        the cleaning for twice the restriction, and that is the trade."""
        v = self.inlet_velocity_m_s(flow_m3_s)
        return self.spec.loss_coefficient * 0.5 * gas_density_kg_m3 * v * v

    @property
    def cup_full_frac(self) -> float:
        return min(1.0, self.cup_g / max(1e-6, self.cup_capacity_g))

    @property
    def reentraining(self) -> bool:
        """A full cup does not merely stop catching -- it gives back."""
        return self.cup_full_frac >= 1.0

    def step(self, dt_s: float, flow_m3_s: float, dust_mg_m3: float, **kw) -> dict:
        """Run it. Returns what got past, which is the element's problem."""
        dt = max(0.0, float(dt_s))
        presented_g = float(dust_mg_m3) * 1e-3 * max(0.0, flow_m3_s) * dt
        eff = self.mass_efficiency(flow_m3_s, **kw)
        if self.reentraining:
            # the cup is full: it catches nothing net, and shedding what
            # it holds is the lump that kills the element
            shed = min(self.cup_g, presented_g * 0.5)
            self.cup_g -= shed
            return {"presented_g": presented_g, "captured_g": 0.0,
                    "passed_g": presented_g + shed, "efficiency": 0.0,
                    "cup_full_frac": self.cup_full_frac, "reentraining": True,
                    "why": "the cup is full and the duckbill is not ejecting, so the "
                           "cyclone is handing the element everything it ever caught"}
        captured = presented_g * eff
        if self.ejector_working:
            # a working vacuator keeps the cup nearly empty; it is the
            # ejector, not the cup, that gives a pre-cleaner its life
            self.cup_g = max(0.0, self.cup_g + captured * 0.05)
        else:
            self.cup_g += captured
        return {"presented_g": presented_g, "captured_g": captured,
                "passed_g": presented_g - captured, "efficiency": eff,
                "cup_full_frac": self.cup_full_frac, "reentraining": False,
                "ejecting": self.ejector_working}

    def service_need(self):
        import servicing as sv
        if self.cup_full_frac < 0.6 and self.ejector_working:
            return None
        return sv.Need(
            identity=self.identity, want="empty-precleaner",
            position=tuple(self.position), quantity=self.cup_g, unit="g",
            matches="dust-cup", urgency=1.0 if self.reentraining else 0.5,
            minutes=10.0, skill="mechanic",
            label=(f"{self.identity} cup {self.cup_full_frac * 100:.0f}% full"
                   + ("" if self.ejector_working else ", VACUATOR NOT EJECTING")),
            why="a hardened duckbill stops the cup emptying itself; once it is full "
                "the cyclone re-entrains its own catch and delivers the element a "
                "lump of dust it was fitted to avoid")


#: ISO 12103-1 A4 coarse test dust, binned. This is the distribution
#: filter and pre-cleaner ratings are measured against, and its shape is
#: the reason "eighty-five per cent efficient" and "cuts at five microns"
#: are the same statement.
COARSE_TEST_DUST: tuple = (
    (2.0e-6, 0.06), (5.5e-6, 0.12), (11.0e-6, 0.16), (22.0e-6, 0.22),
    (45.0e-6, 0.24), (90.0e-6, 0.15), (150.0e-6, 0.05),
)


def intake_flow_m3_s(displacement_l: float, rpm: float, strokes: int = 4,
                     volumetric_efficiency: float = 0.95,
                     boost_ratio: float = 1.0) -> float:
    """Air an engine actually draws THROUGH THE FILTER.

    The element sits upstream of the compressor, so it passes the whole
    mass flow at AMBIENT density -- which is why a turbocharged engine
    needs an air cleaner sized for its boost and not for its swept
    volume. Forgetting the boost ratio undersizes the filter by however
    much the turbo is doing, which on a modern diesel is most of it."""
    per_rev = float(displacement_l) / 1000.0 / (2.0 if strokes == 4 else 1.0)
    return per_rev * (max(0.0, float(rpm)) / 60.0) * volumetric_efficiency * boost_ratio


@dataclass
class IntakeElement:
    """The paper element downstream. It fills; it does not wear out."""
    identity: str = "intake.element"
    capacity_g: float = 600.0
    loaded_g: float = 0.0
    position: tuple = (0.0, 0.0, 0.0)

    @property
    def blocked_frac(self) -> float:
        return min(1.0, self.loaded_g / max(1e-6, self.capacity_g))

    def load(self, grams: float) -> float:
        self.loaded_g += max(0.0, float(grams))
        return self.blocked_frac

    def restriction_pa(self, flow_m3_s: float, clean_pa: float = 1200.0) -> float:
        """Restriction climbs steeply as the cake builds, and the last
        ten per cent of life is most of the climb."""
        b = self.blocked_frac
        return clean_pa * (1.0 + 9.0 * b * b) * (max(0.0, flow_m3_s) / 0.6) ** 1.8

    def need(self):
        import servicing as sv
        return sv.element_need(self.identity, self.position, self.blocked_frac,
                               element="air-filter-element")


def precleaner_for_flow(flow_m3_s: float, construction: str = "tangential-multitube",
                        tube_diameter_m: float = 0.050,
                        identity: str = "intake.precleaner") -> GasCyclone:
    """Build a pre-cleaner that actually fits the engine it is on.

    Sizing is not a choice of tube count, it is a choice of INLET
    VELOCITY -- the cyclone has a working band and the count is whatever
    lands the flow inside it. Picking a tube count first is how you end
    up with two hundred metres a second and a device that is a
    restriction with a dust cup attached.

    Real multi-tube pre-cleaners on big engines genuinely do carry a
    hundred or more tubes, and this is why."""
    spec = cyclone_construction(construction)
    area_each = (tube_diameter_m * spec.inlet_width_frac
                 * tube_diameter_m * spec.inlet_height_frac)
    need_area = max(0.0, float(flow_m3_s)) / max(1.0, spec.design_velocity_m_s)
    count = max(1, int(math.ceil(need_area / max(1e-9, area_each))))
    return GasCyclone(identity=identity, tube_diameter_m=tube_diameter_m,
                      count=count, construction=construction,
                      cup_capacity_g=max(200.0, 30.0 * count))
