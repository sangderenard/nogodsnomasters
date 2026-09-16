"""Compressors: the same machine as an expander, run the other way.

A positive-displacement compressor and a steam/air expander are one
mechanism with the gas path reversed. The expander takes gas in at
pressure, lets it push a piston through a swept volume, and hands the
shaft a torque. The compressor takes torque from the shaft, pushes the
piston through the same swept volume, and hands the gas a pressure.
Bore, stroke, cylinders, double-acting, swept volume, mean effective
pressure, work per revolution, mechanical efficiency -- every term is
the same term, and this module deliberately uses expander.py's own
conventions so the two read as the pair they are:

    work_per_rev = mep * swept_volume * strokes_per_rev * cylinders
    torque       = work_per_rev / (2*pi)   ... * efficiency, if producing
                                           ... / efficiency, if absorbing

That last asymmetry is real and is not a sign slip. Friction takes its
cut out of what an expander delivers, and adds its cut on top of what a
compressor demands, so the same 85% efficiency multiplies one and
divides the other.

WHY CONSTRUCTION IS DECLARED. "Compressor" is not one machine. A shop
compressor is pistons on a crankshaft with real reciprocating mass and
a torque that swings hard within every revolution. A car's A/C
compressor is axial pistons on a swash plate -- still reciprocating,
but moving sinusoidally with shaft angle rather than through a
connecting rod. A scroll has no pistons at all and compresses
continuously. And a Roots blower does not compress internally AT ALL:
it carries gas around at suction pressure and the discharge port
back-flows into it, which is a genuinely different gas law, not a
different efficiency. Handing all of those one rotor shape and one
torque curve, as this codebase did with a flat `inertia_kg_m2 = 0.006`
and a torque linear in speed, throws away the part that matters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

P_ATM_PA = 101_325.0
T_AMBIENT_K = 293.15


@dataclass(frozen=True)
class CompressorConstruction:
    """What a class of compressor is actually built of."""
    key: str
    label: str
    internal_compression: bool   # False = Roots: the discharge port does the compressing
    reciprocating: bool          # has pistons with real reciprocating mass
    slider_crank: bool           # True = rod and crank; False = swash/wobble plate (sinusoidal)
    rotating_groups: int         # how many separate rotors turn inside the casing
    rotor_mass_frac: float       # of unit mass, TOTAL across the rotating groups
    recip_mass_frac: float       # of unit mass that reciprocates rather than rotates
    clearance_frac: float        # dead volume / swept volume -- sets volumetric efficiency
    polytropic_index: float      # real n for this construction's cooling
    mechanical_efficiency: float
    why: str


_CONSTRUCTIONS: tuple[CompressorConstruction, ...] = (
    CompressorConstruction(
        "reciprocating-crank", "reciprocating piston compressor",
        internal_compression=True, reciprocating=True, slider_crank=True,
        rotating_groups=1, rotor_mass_frac=0.18, recip_mass_frac=0.10,
        clearance_frac=0.06, polytropic_index=1.30, mechanical_efficiency=0.82,
        why="pistons on a crankshaft through connecting rods: a real engine's bottom end "
            "driven backwards, with the same reciprocating mass and the same hard torque "
            "swing inside each revolution"),
    CompressorConstruction(
        "swash-plate", "swash-plate axial-piston compressor",
        internal_compression=True, reciprocating=True, slider_crank=False,
        rotating_groups=1, rotor_mass_frac=0.22, recip_mass_frac=0.06,
        clearance_frac=0.04, polytropic_index=1.15, mechanical_efficiency=0.80,
        why="the automotive A/C compressor: five to seven pistons parallel to the shaft, "
            "driven by an angled plate, so each piston moves as a pure sinusoid of shaft "
            "angle rather than through a rod -- and the overlapping strokes make it far "
            "smoother than a twin-cylinder crank machine"),
    CompressorConstruction(
        "scroll", "scroll compressor",
        internal_compression=True, reciprocating=False, slider_crank=False,
        rotating_groups=1, rotor_mass_frac=0.28, recip_mass_frac=0.0,
        clearance_frac=0.01, polytropic_index=1.15, mechanical_efficiency=0.88,
        why="one scroll orbits inside another, compressing continuously with no valves "
            "and almost no dead volume; the orbiting mass is balanced by a counterweight "
            "rather than thrown up and down"),
    CompressorConstruction(
        "rotary-vane", "sliding-vane compressor",
        internal_compression=True, reciprocating=False, slider_crank=False,
        rotating_groups=1, rotor_mass_frac=0.30, recip_mass_frac=0.0,
        clearance_frac=0.02, polytropic_index=1.20, mechanical_efficiency=0.82,
        why="vanes slide in an offset rotor, sweeping cells that shrink toward the "
            "discharge port: smooth torque, no pistons"),
    CompressorConstruction(
        "screw", "twin-screw compressor",
        internal_compression=True, reciprocating=False, slider_crank=False,
        rotating_groups=2, rotor_mass_frac=0.34, recip_mass_frac=0.0,
        clearance_frac=0.01, polytropic_index=1.10, mechanical_efficiency=0.87,
        why="two helical rotors mesh so the trapped volume shrinks along their length; "
            "oil-flooded units run close to isothermal, which is why n is lowest here"),
    CompressorConstruction(
        "roots", "Roots blower",
        internal_compression=False, reciprocating=False, slider_crank=False,
        rotating_groups=2, rotor_mass_frac=0.36, recip_mass_frac=0.0,
        clearance_frac=0.0, polytropic_index=1.40, mechanical_efficiency=0.90,
        why="two lobed rotors carry gas around the case at SUCTION pressure and the "
            "discharge port back-flows into each pocket: the work is the pressure "
            "difference times the displacement, with no internal compression at all, "
            "which is exactly why a Roots is inefficient at high pressure ratio"),
    CompressorConstruction(
        "diaphragm", "diaphragm compressor",
        internal_compression=True, reciprocating=True, slider_crank=True,
        rotating_groups=1, rotor_mass_frac=0.16, recip_mass_frac=0.08,
        clearance_frac=0.08, polytropic_index=1.30, mechanical_efficiency=0.70,
        why="a crank drives a diaphragm rather than a piston, so nothing slides in the "
            "gas at all; small, oil-free, and comparatively lossy"),
)

BY_KEY: dict[str, CompressorConstruction] = {c.key: c for c in _CONSTRUCTIONS}


def construction(key: str) -> CompressorConstruction:
    c = BY_KEY.get(str(key))
    if c is None:
        raise KeyError(
            f"unknown compressor construction {key!r}; declare one of: {', '.join(sorted(BY_KEY))}")
    return c


# ---------------------------------------------------------------------
# the gas law
# ---------------------------------------------------------------------

def volumetric_efficiency(pressure_ratio: float, clearance_frac: float, n: float) -> float:
    """How much of the swept volume actually gets drawn in.

    Gas left in the clearance volume at discharge pressure has to
    re-expand before the suction valve can open, and that re-expansion
    eats into the stroke. The classic result

        eta_v = 1 + c - c * r^(1/n)

    is the real reason a single stage cannot reach an arbitrary
    pressure ratio: at high enough r the clearance gas fills the whole
    stroke on its own, delivery falls to nothing, and all the work goes
    into heating the same trapped charge."""
    r = max(1.0, float(pressure_ratio))
    c = max(0.0, float(clearance_frac))
    return max(0.0, 1.0 + c - c * r ** (1.0 / max(n, 1e-6)))


def compression_mep_pa(suction_pa: float, discharge_pa: float, con: CompressorConstruction
                       ) -> float:
    """Indicated work absorbed per unit of swept volume.

    This is the compressor's mean effective pressure, the exact mirror
    of expander.mean_effective_pressure_pa -- and, like it, a pressure
    that multiplies swept volume to give work per revolution."""
    p1 = max(1.0, float(suction_pa))
    p2 = max(p1, float(discharge_pa))
    r = p2 / p1
    if not con.internal_compression:
        # A ROOTS DOES NO INTERNAL COMPRESSION. Each pocket arrives at
        # the discharge port still at suction pressure and is back-filled
        # to discharge pressure, so the work is simply the pressure
        # difference through the whole displacement -- more than a
        # compressing machine needs, and increasingly so as r rises.
        return p2 - p1
    n = con.polytropic_index
    eta_v = volumetric_efficiency(r, con.clearance_frac, n)
    return (n / (n - 1.0)) * p1 * eta_v * (r ** ((n - 1.0) / n) - 1.0)


def discharge_temperature_k(suction_k: float, pressure_ratio: float,
                            con: CompressorConstruction) -> float:
    """Polytropic discharge temperature -- the real reason a compressor
    needs an aftercooler, and the real limit on single-stage ratio."""
    r = max(1.0, float(pressure_ratio))
    n = con.polytropic_index
    return float(suction_k) * r ** ((n - 1.0) / n)


# ---------------------------------------------------------------------
# the machine
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CompressorSpec:
    """A real compressor, described the way expander.py describes a
    cylinder bank -- because it is the same description."""
    construction: str
    bore_m: float
    stroke_m: float
    cylinders: int = 1
    double_acting: bool = False
    rod_length_m: float = 0.0     # slider-crank only; 0 = derive at a real rod/stroke ratio
    unit_mass_kg: float = 0.0

    @property
    def con(self) -> CompressorConstruction:
        return construction(self.construction)

    @property
    def swept_volume_m3(self) -> float:
        return math.pi * (self.bore_m / 2.0) ** 2 * self.stroke_m

    @property
    def strokes_per_rev(self) -> int:
        return 2 if self.double_acting else 1

    @property
    def displacement_m3_per_rev(self) -> float:
        return self.swept_volume_m3 * self.strokes_per_rev * self.cylinders

    @property
    def crank_radius_m(self) -> float:
        return self.stroke_m / 2.0

    @property
    def effective_rod_length_m(self) -> float:
        # the same real rod/stroke ratio engines.py uses when a build
        # does not name one
        return self.rod_length_m if self.rod_length_m > 0.0 else self.stroke_m * 1.75

    @property
    def reciprocating_mass_kg(self) -> float:
        """Pistons, rods and crossheads: real mass that is thrown back
        and forth rather than spun, so it stores no polar inertia but
        does make a real shaking force."""
        return self.unit_mass_kg * self.con.recip_mass_frac

    def piston_velocity_coeff_m_per_rad(self, theta_rad: float) -> float:
        """ds/dtheta for one piston at this shaft angle.

        A crank machine uses the exact slider-crank result (the same one
        engines.py uses for a real piston); a swash or wobble plate
        drives its pistons as a pure sinusoid of shaft angle, because
        there is no connecting rod to make it otherwise. That is a real
        construction difference, not a modelling shortcut."""
        r = self.crank_radius_m
        if r <= 0.0:
            return 0.0
        if not self.con.slider_crank:
            return r * math.sin(theta_rad)
        length = self.effective_rod_length_m
        sin_t = math.sin(theta_rad)
        cos_t = math.cos(theta_rad)
        root = math.sqrt(max(length * length - r * r * sin_t * sin_t, 1e-12))
        return r * sin_t * (1.0 + r * cos_t / root)

    # --- the load the shaft actually feels ---------------------------

    def mean_torque_nm(self, suction_pa: float, discharge_pa: float) -> float:
        """Mean torque ABSORBED (positive = load on the shaft).

        Mechanical efficiency DIVIDES here where it multiplies in the
        expander: friction adds to what the shaft must supply."""
        con = self.con
        mep = compression_mep_pa(suction_pa, discharge_pa, con)
        work_per_rev_j = mep * self.displacement_m3_per_rev
        return work_per_rev_j / (2.0 * math.pi) / max(con.mechanical_efficiency, 1e-3)

    def mass_flow_kg_s(self, omega_rad_s: float, suction_density_kg_m3: float,
                       suction_pa: float, discharge_pa: float) -> float:
        """Delivered mass flow, after clearance re-expansion has taken
        its share of the stroke."""
        con = self.con
        r = max(1.0, float(discharge_pa) / max(float(suction_pa), 1.0))
        eta_v = (1.0 if not con.internal_compression
                 else volumetric_efficiency(r, con.clearance_frac, con.polytropic_index))
        per_rev = float(suction_density_kg_m3) * self.displacement_m3_per_rev * eta_v
        return per_rev * abs(float(omega_rad_s)) / (2.0 * math.pi)

    def torque_at_angle_nm(self, theta_rad: float, suction_pa: float,
                           discharge_pa: float) -> float:
        """Instantaneous absorbed torque through one revolution.

        For a machine with pistons this is a real indicator diagram
        walked round the crank: each cylinder re-expands its clearance
        gas, draws in at suction pressure, compresses polytropically,
        and discharges against the delivery pressure, and the torque at
        any angle is the sum over cylinders of cylinder pressure times
        piston area times ds/dtheta. It is what makes a two-cylinder
        shop compressor shake its mounts and a seven-piston swash plate
        not. A machine with no pistons has no such swing and reports its
        mean torque at every angle, which is the honest answer for a
        scroll or a screw rather than an invented wobble."""
        con = self.con
        mean = self.mean_torque_nm(suction_pa, discharge_pa)
        if not con.reciprocating or self.cylinders <= 0:
            return mean
        p1 = max(1.0, float(suction_pa))
        p2 = max(p1, float(discharge_pa))
        area = math.pi * (self.bore_m / 2.0) ** 2
        n = con.polytropic_index
        c = max(1e-6, con.clearance_frac)
        r_ratio = p2 / p1
        # crank angles (from TDC) where the charge finishes re-expanding
        # and where it reaches discharge pressure, both as fractions of
        # the stroke -- the two corners of a real indicator card
        reexpand_frac = min(1.0, c * (r_ratio ** (1.0 / n) - 1.0))
        total = 0.0
        # even firing: the cylinders are spaced equally round the shaft
        for i in range(self.cylinders):
            th = theta_rad + 2.0 * math.pi * i / self.cylinders
            s = self._stroke_fraction(th)
            dsdt = self.piston_velocity_coeff_m_per_rad(th)
            descending = dsdt > 0.0   # moving away from TDC: expanding volume
            if descending:
                # re-expansion of the clearance gas, then suction
                p = (p2 * (c / max(c + s, 1e-9)) ** n) if s < reexpand_frac else p1
            else:
                # compression from suction, then discharge once it reaches p2
                comp = p1 * ((1.0 + c) / max(s + c, 1e-9)) ** n
                p = min(comp, p2)
            # gas pressure acts against the piston's motion in both
            # directions of the cycle; the sign of ds/dtheta carries it
            total += (p - p1) * area * dsdt
        swing = total / max(con.mechanical_efficiency, 1e-3)
        return swing * self.strokes_per_rev

    def _stroke_fraction(self, theta_rad: float) -> float:
        """Piston position as a fraction of the stroke, 0 at TDC."""
        r = self.crank_radius_m
        if r <= 0.0:
            return 0.0
        if not self.con.slider_crank:
            disp = r * (1.0 - math.cos(theta_rad))
        else:
            length = self.effective_rod_length_m
            sin_t = math.sin(theta_rad)
            root = math.sqrt(max(length * length - r * r * sin_t * sin_t, 1e-12))
            disp = r * (1.0 - math.cos(theta_rad)) + length - root
        return max(0.0, min(1.0, disp / self.stroke_m))

    def torque_ripple_frac(self, suction_pa: float, discharge_pa: float,
                           samples: int = 72) -> float:
        """(peak - trough) / mean over one revolution: how hard this
        machine shakes what it is bolted to."""
        mean = self.mean_torque_nm(suction_pa, discharge_pa)
        if mean <= 0.0:
            return 0.0
        vals = [self.torque_at_angle_nm(2.0 * math.pi * i / samples, suction_pa, discharge_pa)
                for i in range(samples)]
        return (max(vals) - min(vals)) / mean


def displacement_for_rating(rated_w: float, reference_omega_rad_s: float,
                            con: CompressorConstruction,
                            suction_pa: float = P_ATM_PA,
                            discharge_pa: float = P_ATM_PA * 8.0) -> float:
    """The displacement a compressor of this rating must have.

    The graph already declares `rated_w` and `reference_omega_rad_s` on
    every compressor node and refrigeration.py is explicit that this is
    the single declaration of how big a compressor is -- so size the
    real machine from it rather than adding a second, disagreeing one."""
    mep = compression_mep_pa(suction_pa, discharge_pa, con)
    if mep <= 0.0 or reference_omega_rad_s <= 0.0:
        return 0.0
    torque = float(rated_w) / float(reference_omega_rad_s)
    return torque * 2.0 * math.pi * con.mechanical_efficiency / mep


def spec_for_rating(construction_key: str, rated_w: float, reference_omega_rad_s: float,
                    unit_mass_kg: float, cylinders: int = 0,
                    suction_pa: float = P_ATM_PA,
                    discharge_pa: float = P_ATM_PA * 8.0) -> CompressorSpec:
    """A fully dimensioned compressor from the rating the graph declares.

    Bore and stroke come out of the displacement at a real square-ish
    bore/stroke ratio, the way a real design of this size would be laid
    out, so the piston kinematics have genuine dimensions to work with
    rather than a displacement with no shape."""
    con = construction(construction_key)
    if cylinders <= 0:
        # real defaults per construction: a swash plate is built with
        # many small pistons precisely to run smoothly, a crank machine
        # with few large ones
        cylinders = 7 if construction_key == "swash-plate" else 2
    disp = displacement_for_rating(rated_w, reference_omega_rad_s, con, suction_pa, discharge_pa)
    per_cyl = max(1e-9, disp / max(1, cylinders))
    # square engine proportions: stroke = bore, so V = pi/4 * b^3
    bore = (4.0 * per_cyl / math.pi) ** (1.0 / 3.0)
    return CompressorSpec(construction=construction_key, bore_m=bore, stroke_m=bore,
                          cylinders=cylinders, unit_mass_kg=unit_mass_kg)
