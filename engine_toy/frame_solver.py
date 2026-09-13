"""K u = f. The piece everything else was waiting on.

Four separate things in this project were blocked on the same absence:
gravity could not be applied because gravity is a FORCE and nothing here
turned force into displacement; the shakedown tested three members of a
hundred and twenty-nine because prescribed positions only strain their
immediate neighbours; the beam kinematics had validated equations and no
displacement field to feed them; and the camber could not be computed
because nobody knew where the structure settled.

All four are the same missing solve, and it is not a large one. Fifty-one
nodes at six degrees of freedom each is three hundred and six unknowns --
a linear system small enough that assembling it honestly costs less than
the workarounds did.

WHAT IS AND IS NOT HERE. This is linear static analysis: small
displacements, elastic response, one solve. It does not do the plasticity
-- `vehicle_mechanical_material` already does, and does it better than
anything written here would. The division is deliberate: this finds where
the structure goes and what each member carries, and hands those strains
to the game's own material law to decide what the metal does about it.

ROUND TUBES MAKE THE ELEMENT SIMPLE, and it is worth saying why rather
than quietly enjoying it. A circular section is axisymmetric: the second
moment is identical about every axis, so the local frame can be built any
way that is perpendicular to the member and the answer is the same. There
is no principal axis to track, no orientation to author, and no way to
get a member "rolled" wrong. For an I-beam every one of those would be a
real parameter and a real source of error.

TIMOSHENKO, NOT EULER-BERNOULLI. The shear flexibility term is carried
through (phi in the element below). It matters precisely where this
structure is stubby -- trunnion braces, pedestals, race spokes -- and
leaving it out makes short members look stiffer than they are, which is
the error that flatters exactly the parts carrying the most load.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from milspec import MATERIAL_BY_KEY
from joints import (MEMBER_CONSTRAINTS, DOF_NAMES, ROUTED_TOKENS,
                    constraint_of)
from beam_theory import shear_coefficient

DOF_PER_NODE = 6
#: the last resort under a rotational diagonal that would
#: otherwise be zero. Only nodes with no mass and no extent
#: reach it.
ROTARY_FLOOR_KG_M2 = 1e-6
#: below this eigenvalue a mode has no stiffness behind it and
#: is an articulation rather than a deformation.
MECHANISM_EIGENVALUE = 1e-6
GRAVITY = 9.80665


def _local_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A 3x3 rotation whose first row is the member axis.

    The other two rows are any perpendicular pair -- which is legitimate
    ONLY because the section is circular. Picking them arbitrarily for a
    non-axisymmetric section would silently rotate its strong axis."""
    x = b - a
    length = float(np.linalg.norm(x))
    if length < 1e-12:
        return np.eye(3)
    x = x / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(helper, x)
    y /= max(float(np.linalg.norm(y)), 1e-12)
    z = np.cross(x, y)
    return np.vstack([x, y, z])


def _routed(edge) -> bool:
    """One integer membership test in the common case.

    The token is on the edge, ROUTED_TOKENS is a frozenset built once at
    import, and this runs sixteen hundred times per assembly. The string
    paths below are the fallbacks for a document written before tokens
    existed -- correct, and slower, which is the right way round."""
    tok = edge.get("constraint_token")
    if tok is not None:
        return tok in ROUTED_TOKENS
    if edge.get("routed") is not None:
        return bool(edge["routed"])
    mc = MEMBER_CONSTRAINTS.get(edge.get("constraint"))
    return bool(mc and mc.routed)


def _constraint_freedoms(edge) -> tuple:
    tok = edge.get("constraint_token")
    if tok is not None:
        return constraint_of(tok).freedoms()
    mc = MEMBER_CONSTRAINTS.get(edge.get("constraint"))
    return mc.freedoms() if mc else ()


def _released_dofs(edge) -> list:
    """Which of the twelve element freedoms this member does not carry.

    THE JOINT IS THE BOUNDARY CONDITION. A slider welded into a frame
    is a bar, and a frame with thirty-nine of them is a different
    machine from the one that was drawn -- stiffer everywhere, and
    stiffest exactly where the design intends to be free. So the edge
    declares what it releases and this returns the local indices: 0-5
    for end a, 6-11 for end b."""
    names = edge.get("freedoms_released")
    ends = edge.get("release_ends")
    if names is None:
        tok = edge.get("constraint_token")
        mc = (constraint_of(tok) if tok is not None
              else MEMBER_CONSTRAINTS.get(edge.get("constraint")))
        if mc is None:
            return []
        names, ends = list(mc.freedoms()), mc.release_ends
    if not names:
        return []
    idx = [DOF_NAMES.index(n) for n in names if n in DOF_NAMES]
    out = [6 + i for i in idx]                 # end b always
    if ends == "both":
        out = [i for i in idx] + out
    return out


def _condense(k, released):
    """Remove released freedoms from an element matrix by static
    condensation -- the exact way, not by zeroing rows.

    Zeroing a row and column says "this freedom carries no force AND is
    held at zero", which is a different structure: it fixes the freedom
    rather than freeing it. Condensation eliminates it properly, so the
    force that would have gone through it redistributes into the
    freedoms that remain, which is what a real pin or slide does."""
    if not released:
        return k
    keep = [i for i in range(k.shape[0]) if i not in set(released)]
    r = list(set(released))
    krr = k[np.ix_(r, r)]
    # a released set that is itself singular means the member has been
    # freed into a mechanism; fall back to the pseudo-inverse rather
    # than returning a matrix full of infinities
    try:
        inv = np.linalg.inv(krr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(krr)
    reduced = k[np.ix_(keep, keep)] - k[np.ix_(keep, r)] @ inv @ k[np.ix_(r, keep)]
    out = np.zeros_like(k)
    out[np.ix_(keep, keep)] = reduced
    return out


def element_stiffness(E, G, A, I, J, L, kappa):
    """The 12x12 local stiffness of a 3D Timoshenko beam element.

    Standard form. `phi` is the shear-flexibility parameter: zero
    recovers Euler-Bernoulli exactly, and it grows as the member gets
    short and fat, which is when shear stops being negligible."""
    k = np.zeros((12, 12))
    As = kappa * A
    phi = 12.0 * E * I / (G * As * L * L) if G * As > 0 else 0.0
    # --- axial ---
    ea = E * A / L
    k[0, 0] = k[6, 6] = ea
    k[0, 6] = k[6, 0] = -ea
    # --- torsion ---
    gj = G * J / L
    k[3, 3] = k[9, 9] = gj
    k[3, 9] = k[9, 3] = -gj
    # --- bending, both planes (identical: the section is axisymmetric) ---
    f = E * I / (L ** 3 * (1.0 + phi))
    kv = 12.0 * f
    km = 6.0 * L * f
    ka = (4.0 + phi) * L * L * f
    kb = (2.0 - phi) * L * L * f
    # bending in the x-y plane: v (1,7) with theta_z (5,11)
    for v1, t1, v2, t2, sign in ((1, 5, 7, 11, +1.0), (2, 4, 8, 10, -1.0)):
        k[v1, v1] = k[v2, v2] = kv
        k[v1, v2] = k[v2, v1] = -kv
        k[t1, t1] = k[t2, t2] = ka
        k[t1, t2] = k[t2, t1] = kb
        k[v1, t1] = k[t1, v1] = sign * km
        k[v1, t2] = k[t2, v1] = sign * km
        k[v2, t1] = k[t1, v2] = -sign * km
        k[v2, t2] = k[t2, v2] = -sign * km
    return k


@dataclass
class FrameSolver:
    """A production graph as a linear elastic frame."""
    document: dict
    #: extra point loads, identity -> (fx, fy, fz) newtons
    loads: dict = field(default_factory=dict)
    gravity: bool = True
    index: dict = field(default_factory=dict)
    members: list = field(default_factory=list)

    def __post_init__(self) -> None:
        nodes = self.document["nodes"]
        self.index = {n["identity"]: i for i, n in enumerate(nodes)}
        self.position = np.array([n["reference_position"] for n in nodes], float)
        # A ROUTED LINE IS NOT A MEMBER. It carries fluid, and eighty-four
        # of them were being assembled as structural steel around the
        # barrel -- a water jacket that stiffened the gun in the model
        # and does nothing of the kind in metal.
        # ONE GATHER, not sixteen hundred tests. graph_columns builds an
        # int32 token column once per document and masks over the forty
        # declared constraints, so "which of these are structure" is an
        # indexing operation rather than a loop.
        from graph_columns import structural
        edges = self.document["edges"]
        self.members = [edges[i] for i in structural(self.document)]

    # ------------------------------------------------------------------
    #: How much stiffer than the stiffest REAL member a rigid link is
    #: made. A rigid link has no section of its own to solve -- it is an
    #: idealisation, and the only requirement on the number is that it
    #: be large enough that the link contributes no compliance and small
    #: enough not to wreck the conditioning of K. Three decades does
    #: both, and it is declared here rather than buried in a formula.
    RIGID_LINK_FACTOR = 1.0e2

    def _rigid_reference(self):
        """The stiffest real member in the graph -- as STIFFNESS, not as
        section, which is the distinction that decides whether this
        works at all.

        A rigid link gets a section so that it out-stiffens real
        structure. Sizing that section on EA alone looks equivalent and
        is not: stiffness is EA/L, and the shell edges around a small
        body are millimetres long, so a section matched to the stiffest
        member gives them a stiffness thousands of times larger again.
        K then has a condition number no solve survives -- this graph
        returned twelve kilometres of deflection and ten strain, which
        is not a soft structure, it is arithmetic that has run out of
        significant figures. Matching STIFFNESS per unit length keeps
        every rigid link at the same multiple of real structure
        regardless of how short it happens to be."""
        if getattr(self, "_rigid_ref", None) is not None:
            return self._rigid_ref
        k_ax = k_bend = 0.0
        for edge in self.members:
            if edge.get("rigid"):
                continue
            length = float(edge.get("rest_length", 0.0))
            if length < 1e-6:
                continue
            d = edge.get("damage") or {}
            area = float(d.get("section_area_m2", 1e-4))
            second = float(d.get("second_moment_m4", area * area
                                 / (4.0 * math.pi)))
            mat = MATERIAL_BY_KEY.get(d.get("material", "4130n"),
                                      MATERIAL_BY_KEY["4130n"])
            e = float(d.get("youngs_modulus_pa", mat.youngs_pa))
            k_ax = max(k_ax, e * area / length)
            k_bend = max(k_bend, e * second / length ** 3)
        self._rigid_ref = (max(k_ax, 1.0), max(k_bend, 1e-9))
        return self._rigid_ref

    def _section(self, edge):
        # A RIGID EDGE IS NOT A TUBE. It is a body's own skin, or a
        # bolted seat, and it was emitted at an insignificant radius
        # precisely so that nothing would mistake it for structure. If
        # that radius is then handed to a beam element, the body turns
        # into a cage of 4 mm noodles and the whole machine goes soft:
        # this graph solved to 1069 mm of deflection before rigid was
        # honoured here, which is a metre of travel invented entirely by
        # modelling skin as tubing. So a rigid edge is assembled as a
        # RIGID LINK -- the material it declares, at a section chosen to
        # out-stiffen the stiffest real member in the graph.
        if edge.get("rigid"):
            d = edge.get("damage") or {}
            alloy = d.get("material", "4130n")
            mat = MATERIAL_BY_KEY.get(alloy, MATERIAL_BY_KEY["4130n"])
            e = float(d.get("youngs_modulus_pa", mat.youngs_pa))
            k_ax, k_bend = self._rigid_reference()
            length = max(float(edge.get("rest_length", 0.0)), 1e-4)
            area = self.RIGID_LINK_FACTOR * k_ax * length / e
            second = self.RIGID_LINK_FACTOR * k_bend * length ** 3 / e
            return (e, float(mat.shear_pa), area, second, 2.0 * second,
                    1.0, mat)
        d = edge["damage"]
        alloy = d.get("material", "4130n")
        mat = MATERIAL_BY_KEY.get(alloy, MATERIAL_BY_KEY["4130n"])
        area = float(d.get("section_area_m2", 1e-4))
        second = float(d.get("second_moment_m4", area * area / (4.0 * math.pi)))
        # a circular tube's radii, back out of area and second moment
        ro = float(edge.get("radius", 0.012))
        ri = max(0.0, math.sqrt(max(ro * ro - area / math.pi, 0.0)))
        return (float(d.get("youngs_modulus_pa", mat.youngs_pa)),
                float(d.get("shear_modulus_pa", mat.shear_pa)),
                area, second, 2.0 * second,      # J = 2I, exact for a circle
                shear_coefficient(ro, ri), mat)

    def _fixed_dofs(self) -> list:
        """Nodes bolted to the world. Their six freedoms are removed --
        which is what makes the system solvable at all: a structure with
        no support has six rigid-body modes and a singular stiffness."""
        fixed = []
        for n in self.document["nodes"]:
            if n.get("fixed_to") or n.get("kind") == "structural-body-pin-frame-foot":
                base = self.index[n["identity"]] * DOF_PER_NODE
                fixed.extend(range(base, base + DOF_PER_NODE))
        return fixed

    # ------------------------------------------------------------------
    def solve(self) -> dict:
        n_dof = len(self.document["nodes"]) * DOF_PER_NODE
        K = np.zeros((n_dof, n_dof))
        f = np.zeros(n_dof)

        for edge in self.members:
            ia, ib = self.index[edge["a"]], self.index[edge["b"]]
            a, b = self.position[ia], self.position[ib]
            L = float(np.linalg.norm(b - a))
            if L < 1e-9:
                continue
            E, G, A, I, J, kappa, _mat = self._section(edge)
            kl = element_stiffness(E, G, A, I, J, L, kappa)
            # THE JOINT, APPLIED. Released freedoms are condensed out of
            # the element before it is rotated into the structure, which
            # is the difference between a recoil slide and a bar.
            kl = _condense(kl, _released_dofs(edge))
            R = _local_frame(a, b)
            T = np.zeros((12, 12))
            for blk in range(4):
                T[blk * 3:blk * 3 + 3, blk * 3:blk * 3 + 3] = R
            kg = T.T @ kl @ T
            dofs = (list(range(ia * DOF_PER_NODE, ia * DOF_PER_NODE + 6))
                    + list(range(ib * DOF_PER_NODE, ib * DOF_PER_NODE + 6)))
            K[np.ix_(dofs, dofs)] += kg

        # --- the loads -------------------------------------------------
        if self.gravity:
            # LUMPED AT THE NODES. Half of each member's mass to each of
            # its ends, plus whatever the node itself weighs. Crude
            # against a consistent mass matrix and entirely adequate for
            # a static settle, which is what this is for.
            for edge in self.members:
                # A RIGID LINK WEIGHS NOTHING. Its section is an
                # idealisation sized to out-stiffen real structure, so
                # multiplying it by a density gives a body's own skin a
                # mass thousands of times the body's -- the mass is
                # already on the body node it belongs to.
                if edge.get("rigid"):
                    continue
                E, G, A, I, J, kappa, mat = self._section(edge)
                ia, ib = self.index[edge["a"]], self.index[edge["b"]]
                L = float(np.linalg.norm(self.position[ib] - self.position[ia]))
                w = A * L * mat.density_kg_m3 * GRAVITY / 2.0
                f[ia * DOF_PER_NODE + 1] -= w
                f[ib * DOF_PER_NODE + 1] -= w
            for n in self.document["nodes"]:
                i = self.index[n["identity"]]
                f[i * DOF_PER_NODE + 1] -= float(n.get("mass_kg", 0.0)) * GRAVITY
        for ident, force in self.loads.items():
            i = self.index[ident]
            f[i * DOF_PER_NODE:i * DOF_PER_NODE + 3] += np.asarray(force, float)

        # --- supports and solve ---------------------------------------
        fixed = set(self._fixed_dofs())
        free = np.array([d for d in range(n_dof) if d not in fixed])
        if len(free) == 0:
            raise ValueError("every degree of freedom is fixed")
        u = np.zeros(n_dof)
        Kff = K[np.ix_(free, free)]
        try:
            u[free] = np.linalg.solve(Kff, f[free])
        except np.linalg.LinAlgError:
            # a singular stiffness means a mechanism, not a structure:
            # some part of the graph can move without straining anything
            u[free] = np.linalg.lstsq(Kff, f[free], rcond=None)[0]
        return self._recover(u, K, f, fixed)

    # ------------------------------------------------------------------
    def _recover(self, u, K, f, fixed) -> dict:
        """Per-member strains, in the form the game's material law wants."""
        out = {"identities": [], "axial_strain": [], "bending_strain": [],
               "shear_strain": [], "axial_force_n": [], "length_m": []}
        for edge in self.members:
            ia, ib = self.index[edge["a"]], self.index[edge["b"]]
            a, b = self.position[ia], self.position[ib]
            L = float(np.linalg.norm(b - a))
            if L < 1e-9:
                continue
            E, G, A, I, J, kappa, _m = self._section(edge)
            R = _local_frame(a, b)
            ua = u[ia * DOF_PER_NODE:ia * DOF_PER_NODE + 6]
            ub = u[ib * DOF_PER_NODE:ib * DOF_PER_NODE + 6]
            # into the member's own frame
            da, ra = R @ ua[:3], R @ ua[3:]
            db, rb = R @ ub[:3], R @ ub[3:]
            axial = (db[0] - da[0]) / L
            ro = float(edge.get("radius", 0.012))
            curv_y = (rb[1] - ra[1]) / L
            curv_z = (rb[2] - ra[2]) / L
            bending = math.sqrt(curv_y ** 2 + curv_z ** 2) * ro
            shear_y = (db[1] - da[1]) / L - (ra[2] + rb[2]) / 2.0
            shear_z = (db[2] - da[2]) / L + (ra[1] + rb[1]) / 2.0
            twist = (rb[0] - ra[0]) / L
            shear = math.sqrt(shear_y ** 2 + shear_z ** 2 + (twist * ro) ** 2)
            # A RELEASED FREEDOM CARRIES NOTHING, and its relative motion
            # is therefore not strain. A slider's ends move apart by the
            # whole of its travel; multiplying that by EA reported an
            # outrigger jack at 845 MN, which is not a load, it is the
            # stroke of the jack read as if the jack were a bar. The
            # force along such a member comes from its own law, not from
            # its section, and this solve is not where that lives.
            released = set(edge.get("freedoms_released")
                           or _constraint_freedoms(edge))
            if "x" in released:
                axial = 0.0
            if released & {"ry", "rz"}:
                bending = 0.0
            if released & {"y", "z", "rx"}:
                shear = 0.0
            out["identities"].append(edge["identity"])
            out["axial_strain"].append(axial)
            out["bending_strain"].append(bending)
            out["shear_strain"].append(shear)
            out["axial_force_n"].append(axial * E * A)
            out["length_m"].append(L)
        for key in ("axial_strain", "bending_strain", "shear_strain",
                    "axial_force_n", "length_m"):
            out[key] = np.array(out[key])
        out["displacement"] = u.reshape(-1, DOF_PER_NODE)
        out["reaction"] = (K @ u - f).reshape(-1, DOF_PER_NODE)
        out["max_deflection_m"] = float(np.abs(u.reshape(-1, DOF_PER_NODE)[:, :3]).max())
        return out

    def settled_position(self, solved: dict) -> np.ndarray:
        """Where the structure actually sits once it is carrying itself."""
        return self.position + solved["displacement"][:, :3]


    def modes(self, count: int = 6) -> dict:
        """Natural frequencies of the structure.

        THE NUMBER THAT DECIDES WHETHER SOFTNESS MATTERS. Static droop
        is a zeroing problem -- it is the same every time, so it can be
        boresighted out or measured by a muzzle reference and handed to
        the fire control. What cannot be compensated is the mount
        RINGING: every shot is an impulse, the structure rings at its own
        frequencies afterwards, and if it has not settled before the next
        round leaves, that round departs from a moving gun.

        So the question is never "how far does it sag" but "how fast does
        it come back", and that is the first natural frequency against
        the cyclic rate.
        """
        n_dof = len(self.document["nodes"]) * DOF_PER_NODE
        K = np.zeros((n_dof, n_dof))
        M = np.zeros(n_dof)
        for edge in self.members:
            ia, ib = self.index[edge["a"]], self.index[edge["b"]]
            a, b = self.position[ia], self.position[ib]
            L = float(np.linalg.norm(b - a))
            if L < 1e-9:
                continue
            E, G, A, I, J, kappa, mat = self._section(edge)
            kl = _condense(element_stiffness(E, G, A, I, J, L, kappa),
                           _released_dofs(edge))
            R = _local_frame(a, b)
            T = np.zeros((12, 12))
            for blk in range(4):
                T[blk * 3:blk * 3 + 3, blk * 3:blk * 3 + 3] = R
            dofs = (list(range(ia * DOF_PER_NODE, ia * DOF_PER_NODE + 6))
                    + list(range(ib * DOF_PER_NODE, ib * DOF_PER_NODE + 6)))
            K[np.ix_(dofs, dofs)] += T.T @ kl @ T
            # A RIGID LINK WEIGHS NOTHING HERE EITHER. Its section is an
            # idealisation sized to out-stiffen real structure, so
            # A*L*rho is a body's own skin weighing thousands of times
            # what the body does. The static solve was corrected for
            # this; the eigenproblem was not, and a mass matrix wrong by
            # that much moves every frequency it reports.
            if edge.get("rigid"):
                continue
            half = A * L * mat.density_kg_m3 / 2.0
            for i in (ia, ib):
                M[i * DOF_PER_NODE:i * DOF_PER_NODE + 3] += half
                # AND ITS ROTARY INERTIA, which is not optional.
                # The standard lumped-mass beam puts half the mass at
                # each end together with half the rod's inertia about
                # that end: m/2 * (r^2/4 + L^2/12). Both terms are the
                # member's own declared section and length.
                r_o = math.sqrt(max(A, 0.0) / math.pi)
                M[i * DOF_PER_NODE + 3:i * DOF_PER_NODE + 6] += \
                    half * (r_o * r_o / 4.0 + L * L / 12.0)
        for n in self.document["nodes"]:
            i = self.index[n["identity"]]
            m_n = float(n.get("mass_kg", 0.0))
            M[i * DOF_PER_NODE:i * DOF_PER_NODE + 3] += m_n
            # A NODE'S OWN ROTARY INERTIA, off the extent it declared.
            # A body of half-extents (a, b, c) has I = m(b^2+c^2)/3 and
            # its permutations; every node here carries `half_extent_m`
            # because the mesher needs it to draw the thing, so this is
            # read rather than guessed.
            h = n.get("half_extent_m") or ()
            if m_n > 0.0 and len(h) == 3:
                a2, b2, c2 = (float(v) ** 2 for v in h)
                for k, pair in enumerate(((b2 + c2), (a2 + c2), (a2 + b2))):
                    M[i * DOF_PER_NODE + 3 + k] += m_n * pair / 3.0
        # ---- WHAT THE FLOOR WAS DOING, AND WHY IT IS NOW A FLOOR -----
        # This used to be `M[M <= 0] = 1e-6` with no rotary inertia
        # assembled at all, so EVERY rotational freedom in the structure
        # carried one microgram. That is not a small error in a
        # frequency, it is a different problem: a station-sized graph
        # came back with a hundred and nineteen modes below 0.05 Hz,
        # all of them 99 per cent rotational with mode shapes two
        # thousand times the size of the real ones. They sit below every
        # elastic mode, so anything that retains "the lowest N" retains
        # the artefact and misses the structure, and they absorb modal
        # force without producing strain -- a 30 kN.s shot moved the gun
        # a tenth of a millimetre and no member changed colour.
        #
        # The floor stays, because the eigenproblem still needs the
        # diagonal to be positive at a node that genuinely has nothing,
        # but it is now a floor under real inertia instead of a
        # substitute for it.
        M[M <= 0.0] = ROTARY_FLOOR_KG_M2
        fixed = set(self._fixed_dofs())
        free = np.array([d for d in range(n_dof) if d not in fixed])
        Kff = K[np.ix_(free, free)]
        Mff = np.diag(1.0 / np.sqrt(M[free]))
        # symmetric standard form: M^-1/2 K M^-1/2
        A_sym = Mff @ Kff @ Mff
        vals, vecs = np.linalg.eigh((A_sym + A_sym.T) / 2.0)
        keep = vals > MECHANISM_EIGENVALUE
        # ---- THE FREEDOMS THAT WERE BEING THROWN AWAY ---------------
        # A mode with no stiffness behind it is not noise. On a frame
        # it is a rigid-body motion; on THIS graph it is the machine's
        # own articulation -- the gun sliding in its cradle, both
        # parallelograms folding, every released pin. Dropping them
        # leaves a basis that can only deform the structure in place,
        # which is right for asking how it rings and useless for asking
        # what it DOES: a shot applied to a basis with no recoil
        # freedom in it simply has nowhere to go.
        #
        # They come back separately rather than mixed in, because they
        # integrate differently: omega is zero, so they have no
        # restoring term and no modal damping, and whatever resists
        # them has to come from the springs, dampers and actuators the
        # joints declared. That is the caller's business, not the
        # eigensolver's.
        mech = np.zeros((n_dof, int((~keep).sum())))
        mech[free, :] = Mff @ vecs[:, ~keep]
        vals, vecs = vals[keep], vecs[:, keep]
        freqs = np.sqrt(vals) / (2.0 * math.pi)
        # BACK OUT OF THE MASS-NORMALISED FORM. The eigenvectors above
        # are of M^-1/2 K M^-1/2, so they are shapes in a scaled space;
        # the shape the structure actually moves in is M^-1/2 times
        # that. Skipping this step gives modes that look plausible and
        # put the motion in the wrong places -- heaviest where the
        # lumped mass is largest rather than where the structure is
        # softest.
        shapes = np.zeros((n_dof, vecs.shape[1]))
        shapes[free, :] = Mff @ vecs
        # THE LUMPED MASS GOES WITH THEM. The shapes are mass-
        # orthonormal (phi^T M phi = I), so any initial condition has to
        # be projected THROUGH M to become modal coordinates -- q0 =
        # phi^T M u0. Returning the shapes without M leaves the caller
        # to guess that, and the guess that looks right (phi^T u0) is
        # wrong by the mass distribution.
        return {"frequencies_hz": freqs[:count],
                "first_hz": float(freqs[0]) if len(freqs) else 0.0,
                "all_frequencies_hz": freqs,
                "shapes": shapes,
                # the articulation: zero-stiffness, mass-orthonormal,
                # and integrated by whoever knows what resists them
                "mechanism_shapes": mech,
                "mechanism_count": int(mech.shape[1]),
                "free": free,
                "lumped_mass": M,
                "modal_mass": np.ones(len(freqs))}
