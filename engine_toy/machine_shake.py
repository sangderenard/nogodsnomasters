"""How a machine on its feet shakes, without solving the frame for it.

WHAT THIS DELIBERATELY IS NOT. It does not step a beam solver, it does
not re-solve which foot is on the ground every tick, and it does not
integrate anything. A frame standing on levelling feet is a rigid body
on a handful of vertical springs, and a rigid body on springs has three
modes and closed-form answers. The expensive part -- deciding which feet
are actually in contact, which is a unilateral contact problem with an
active set -- has already been done once by `feet.stand`. This takes
that answer and linearises about it.

So the cost is one three-by-three eigenproblem per stance and a division
per speed, and what comes back is the thing a sim actually wants: how
much the machine moves at the speed it is running, at what phase, and
how hard each foot is being hammered while it does.

THE COORDINATES ARE THE ONES THE CONTACT SOLVE ALREADY USES: how far
the frame settles, and the two slopes it settles at. Vertical props
constrain exactly those three freedoms and nothing else, so there are
exactly three modes and no need to carry six and watch three of them
come back singular.

  mass      T = 1/2 [ m u'^2 + sx'^2 . sum(m X^2) + sz'^2 . sum(m Z^2)
                      + 2 sx' sz' . sum(m X Z) ]
            and every one of those sums is in the inertia tensor the
            machine already computed -- including the product term,
            which is what tilts the mode shapes off the axes
  stiffness the same matrix feet.stand assembles to find the load share,
            restricted to the feet that came back carrying

WHY THE MODES MATTER MORE THAN THE AMPLITUDE. A machine run steadily
above its rocking modes barely feels them. The damage is done on the way
up and on the way down, when the running speed crosses one -- which is
why a centrifuge that is fine at speed can still destroy itself, and why
"it only shakes for a moment during spin-up" is a description of the
problem rather than a reason to ignore it.

A FOOT CANNOT PULL, AND THIS MODEL CAN. Linearising about a stance
assumes the feet stay in contact. Once the dynamic force at a foot
exceeds the static load holding it down, that foot leaves the floor
every cycle and the machine is hammering rather than vibrating. The
linear answer is wrong there and says so: `lifts_off` is the honest
boundary of this model, not a failure of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import feet as feet_mod
from feet import Ground, Stance


#: Modal damping ratio by what the pads are standing on. Soil damps a
#: footing heavily -- energy radiates away into the half space as well as
#: being lost in the material -- and a slab hardly damps at all, which is
#: why a machine on a good floor rings and the same machine on dirt does
#: not. Real, disclosed ranges rather than one paper's coefficients.
GROUND_DAMPING = {
    "soft-clay": 0.18,
    "firm-soil": 0.12,
    "compacted-gravel": 0.10,
    "tarmac": 0.06,
    "concrete-slab": 0.02,
}


@dataclass(frozen=True)
class Mode:
    """One way the frame can move, and how fast it does it."""
    index: int
    frequency_hz: float
    shape: tuple              # (heave, slope along x, slope along z)
    damping: float

    @property
    def kind(self) -> str:
        """What this mode IS, read from its own shape rather than from
        its index. Sorting by frequency and calling the first one heave
        is right until a soft floor and a wide frame swap them over."""
        heave, sx, sz = (abs(v) for v in self.shape)
        return "heave" if heave >= max(sx, sz) else (
            "rock along x" if sx >= sz else "rock along z")

    @property
    def rpm(self) -> float:
        return self.frequency_hz * 60.0


@dataclass
class ShakeModel:
    """A frame on its feet, reduced to three oscillators."""
    identity: str
    modes: list
    mass_matrix: np.ndarray = field(repr=False)
    stiffness_matrix: np.ndarray = field(repr=False)
    foot_positions: dict = field(default_factory=dict, repr=False)
    foot_stiffness: dict = field(default_factory=dict, repr=False)
    static_forces: dict = field(default_factory=dict, repr=False)
    cg: tuple = (0.0, 0.0, 0.0)

    def critical_speeds_rpm(self) -> list:
        return sorted(m.rpm for m in self.modes if m.frequency_hz > 0.0)

    def describe(self) -> str:
        lines = [f"{self.identity}: three modes on the feet it is standing on"]
        for m in self.modes:
            lines.append(f"    {m.index}  {m.frequency_hz:6.2f} Hz "
                         f"({m.rpm:7.0f} rpm)  {m.kind:12s} "
                         f"zeta {m.damping:.2f}")
        return "\n".join(lines)


def build(identity: str, stance: Stance, feet, rigid_body, g: Ground
          ) -> ShakeModel:
    """Reduce a solved stance to its three modes.

    Only the feet that came back CARRYING are in it. A foot that is not
    touching has no stiffness in the machine's path to the ground, and
    including it would stiffen a model of a frame that is, in fact,
    rocking -- the exact error that makes an unlevel machine look fine."""
    carrying = [f for f in feet if stance.forces.get(f.identity, 0.0) > 0.0]
    cg = np.asarray(rigid_body.center_of_gravity, float)
    m = float(rigid_body.total_mass_kg)
    I = np.asarray(rigid_body.inertia_tensor_kg_m2, float)

    # the three second moments this parametrisation needs, read out of
    # the tensor the machine already has
    sum_xx = (I[1, 1] + I[2, 2] - I[0, 0]) / 2.0
    sum_zz = (I[0, 0] + I[1, 1] - I[2, 2]) / 2.0
    sum_xz = -I[0, 2]
    M = np.array([[m, 0.0, 0.0],
                  [0.0, sum_xx, sum_xz],
                  [0.0, sum_xz, sum_zz]], float)

    K = np.zeros((3, 3))
    positions, stiff = {}, {}
    for f in carrying:
        k = f.stiffness_n_per_m(g)
        X = float(f.position[0]) - float(cg[0])
        Z = float(f.position[2]) - float(cg[2])
        row = np.array([1.0, X, Z])
        K += k * np.outer(row, row)
        positions[f.identity] = (X, Z)
        stiff[f.identity] = k

    zeta = GROUND_DAMPING.get(g.key, 0.10)
    modes = []
    if len(carrying) >= 3 and np.linalg.det(M) > 0:
        # generalised eigenproblem K.phi = w^2 M.phi, by symmetric
        # reduction -- M is small, symmetric and positive definite
        L = np.linalg.cholesky(M)
        Li = np.linalg.inv(L)
        A = Li @ K @ Li.T
        vals, vecs = np.linalg.eigh((A + A.T) / 2.0)
        for i, (w2, v) in enumerate(zip(vals, vecs.T)):
            phi = Li.T @ v
            freq = math.sqrt(max(w2, 0.0)) / (2.0 * math.pi)
            modes.append(Mode(index=i, frequency_hz=freq,
                              shape=tuple(float(x) for x in
                                          phi / np.linalg.norm(phi)),
                              damping=zeta))
    return ShakeModel(identity=identity, modes=modes, mass_matrix=M,
                      stiffness_matrix=K, foot_positions=positions,
                      foot_stiffness=stiff,
                      static_forces=dict(stance.forces),
                      cg=tuple(float(v) for v in cg))


@dataclass(frozen=True)
class Response:
    """What the machine is doing at one running speed."""
    rpm: float
    hz: float
    heave_m: float
    slope_x: float
    slope_z: float
    foot_force_n: dict          # the DYNAMIC force, on top of the static
    lifts_off: tuple            # feet whose static load is being overcome
    nearest_mode: "Mode | None"
    amplification: float        # how far above the static deflection

    @property
    def hammering(self) -> bool:
        return bool(self.lifts_off)

    def describe(self) -> str:
        line = (f"  {self.rpm:7.0f} rpm ({self.hz:5.2f} Hz): "
                f"{self.heave_m * 1e6:8.1f} um heave, "
                f"x{self.amplification:6.2f} static")
        if self.nearest_mode is not None:
            line += f", nearest mode {self.nearest_mode.frequency_hz:.2f} Hz"
        if self.lifts_off:
            line += f"  HAMMERING: {', '.join(self.lifts_off)} leaves the floor"
        return line


def unbalance_force_n(unbalance_kg_m: float, rpm: float) -> float:
    """A rotor's imbalance is a rotating force, and it goes as the SQUARE
    of speed.

    That is the fact that makes this worth modelling: doubling the speed
    quadruples the force, so a rotor that is acceptable at half speed is
    four times worse at full. `unbalance_kg_m` is the residual mass times
    its radius, which is how balancing machines report it."""
    w = rpm * 2.0 * math.pi / 60.0
    return float(unbalance_kg_m) * w * w


def respond(model: ShakeModel, rpm: float, *, unbalance_kg_m: float,
            at=(0.0, 0.0, 0.0)) -> Response:
    """Steady-state response at one speed, in closed form.

    One division per mode. The forcing is the vertical component of the
    rotating unbalance force applied at the rotor's own position, which
    is what makes it excite the rocking modes and not only heave -- a
    rotor over the middle of the frame shakes it up and down, and the
    same rotor in a corner rocks it."""
    hz = rpm / 60.0
    w = hz * 2.0 * math.pi
    force = unbalance_force_n(unbalance_kg_m, rpm)
    rx = float(at[0]) - model.cg[0]
    rz = float(at[2]) - model.cg[2]
    load = force * np.array([1.0, rx, rz])

    q = np.zeros(3)
    static = np.zeros(3)
    for mode in model.modes:
        phi = np.asarray(mode.shape, float)
        mm = float(phi @ model.mass_matrix @ phi)
        kk = float(phi @ model.stiffness_matrix @ phi)
        if mm <= 0.0 or kk <= 0.0:
            continue
        wn = math.sqrt(kk / mm)
        fm = float(phi @ load)
        denom = math.hypot(kk - w * w * mm, 2.0 * mode.damping * wn * w * mm)
        q += phi * (fm / max(denom, 1e-9))
        static += phi * (fm / kk)

    dyn = {}
    lifts = []
    for ident, (X, Z) in model.foot_positions.items():
        k = model.foot_stiffness[ident]
        travel = q[0] + q[1] * X + q[2] * Z
        f = abs(k * travel)
        dyn[ident] = f
        # A FOOT CANNOT PULL. Once the alternating force exceeds the
        # static load pinning that foot down, it leaves the floor on
        # every cycle -- and from there this linear model is describing
        # something that is no longer happening.
        if f > model.static_forces.get(ident, 0.0):
            lifts.append(ident)

    nearest = None
    if model.modes:
        nearest = min(model.modes, key=lambda m: abs(m.frequency_hz - hz))
    static_norm = float(np.linalg.norm(static))
    amp = float(np.linalg.norm(q)) / static_norm if static_norm > 0 else 1.0
    return Response(rpm=rpm, hz=hz, heave_m=abs(float(q[0])),
                    slope_x=float(q[1]), slope_z=float(q[2]),
                    foot_force_n=dyn, lifts_off=tuple(sorted(lifts)),
                    nearest_mode=nearest, amplification=amp)


def sweep(model: ShakeModel, *, unbalance_kg_m: float, at=(0.0, 0.0, 0.0),
          from_rpm: float = 60.0, to_rpm: float = 3600.0,
          steps: int = 60) -> list:
    """Run the machine up and see what it passes through on the way.

    The point of a sweep rather than a single speed: the damage is done
    crossing a mode, not sitting above it."""
    return [respond(model, from_rpm + (to_rpm - from_rpm) * i / max(steps - 1, 1),
                    unbalance_kg_m=unbalance_kg_m, at=at)
            for i in range(steps)]
