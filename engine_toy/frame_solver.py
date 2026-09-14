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


def _local_frame(a: np.ndarray, b: np.ndarray, section_up=None) -> np.ndarray:
    """A 3x3 rotation whose first row is the member axis.

    ``section_up`` fixes local z for a non-axisymmetric section.  It is
    projected normal to the member so slightly imperfect authored geometry
    cannot rotate the section or corrupt orthogonality."""
    x = b - a
    length = float(np.linalg.norm(x))
    if length < 1e-12:
        return np.eye(3)
    x = x / length
    if section_up is not None:
        z = np.asarray(section_up, dtype=float)
        z = z - float(z @ x) * x
        if float(np.linalg.norm(z)) <= 1.0e-10:
            raise ValueError("section_up must not be parallel to the member")
        z /= float(np.linalg.norm(z))
        y = np.cross(z, x)
        y /= max(float(np.linalg.norm(y)), 1e-12)
        z = np.cross(x, y)
        return np.vstack([x, y, z])
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(helper, x)
    y /= max(float(np.linalg.norm(y)), 1e-12)
    z = np.cross(x, y)
    return np.vstack([x, y, z])


def _element_frame(edge: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Orient a travelling joint by its declared motion axis.

    A guide's endpoints locate its two bodies; they do not necessarily lie
    on the guide axis.  Condensing local ``x`` in the endpoint-to-endpoint
    frame therefore released the mounting offset instead of the declared
    slide direction.  Ordinary members still use their geometric axis.
    """
    declared = edge.get("slide_axis")
    if declared is None:
        return _local_frame(a, b, edge.get("section_up"))
    axis = np.asarray(declared, dtype=float)
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-12:
        raise ValueError(f"{edge['identity']}: slide_axis must be nonzero")
    return _local_frame(np.zeros(3), axis / length, edge.get("section_up"))


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


def element_stiffness(E, G, A, Iy, Iz, J, L, kappa):
    """The 12x12 local stiffness of a 3D Timoshenko beam element.

    Standard form. `phi` is the shear-flexibility parameter: zero
    recovers Euler-Bernoulli exactly, and it grows as the member gets
    short and fat, which is when shear stops being negligible."""
    k = np.zeros((12, 12))
    As = kappa * A
    # --- axial ---
    ea = E * A / L
    k[0, 0] = k[6, 6] = ea
    k[0, 6] = k[6, 0] = -ea
    # --- torsion ---
    gj = G * J / L
    k[3, 3] = k[9, 9] = gj
    k[3, 9] = k[9, 3] = -gj
    # local y deflection bends about z; local z deflection bends about y.
    for inertia, v1, t1, v2, t2, sign in (
            (Iz, 1, 5, 7, 11, +1.0), (Iy, 2, 4, 8, 10, -1.0)):
        phi = (12.0 * E * inertia / (G * As * L * L)
               if G * As > 0 else 0.0)
        f = E * inertia / (L ** 3 * (1.0 + phi))
        kv = 12.0 * f
        km = 6.0 * L * f
        ka = (4.0 + phi) * L * L * f
        kb = (2.0 - phi) * L * L * f
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
    gravity_scale: float = 1.0
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
        from graph_columns import structural, structural_nodes
        edges = self.document["edges"]
        self.structural_node = structural_nodes(self.document)
        self.solver_master_index = np.arange(len(nodes), dtype=np.int32)
        self.solver_endpoint_transform = np.repeat(
            np.eye(DOF_PER_NODE, dtype=float)[None, :, :], len(nodes), axis=0)
        for i, node in enumerate(nodes):
            target = node.get("solver_condensed_into")
            if not target:
                continue
            if target not in self.index:
                raise KeyError(
                    f"{node['identity']}: missing solver gestalt {target!r}")
            j = self.index[target]
            if nodes[j].get("solver_condensed_into"):
                raise ValueError(
                    f"{node['identity']}: solver gestalt chains are not allowed")
            self.solver_master_index[i] = j
            offset = self.position[i] - self.position[j]
            skew = np.asarray(((0.0, -offset[2], offset[1]),
                               (offset[2], 0.0, -offset[0]),
                               (-offset[1], offset[0], 0.0)))
            # u(point) = u(master) + theta x offset.
            self.solver_endpoint_transform[i, :3, 3:] = -skew
        self.solver_master_dof = (
            self.solver_master_index[:, None] * DOF_PER_NODE
            + np.arange(DOF_PER_NODE, dtype=np.int32)[None, :])
        self.members = []
        for edge_i in structural(self.document):
            edge = edges[int(edge_i)]
            ia = int(self.solver_master_index[self.index[edge["a"]]])
            ib = int(self.solver_master_index[self.index[edge["b"]]])
            if ia == ib:
                # Both endpoints are points on the same rigid gestalt.
                # Their exact kinematics already satisfy its internal link.
                continue
            if self.structural_node[ia] and self.structural_node[ib]:
                self.members.append(edge)
        # One baked machine is one inertial body. Internal state/render parts
        # can name that body explicitly; their masses and box inertias are
        # condensed onto it rather than becoming extra free bodies.
        self.solver_node_mass = np.zeros(len(nodes), dtype=float)
        self.solver_node_inertia = np.zeros((len(nodes), 3), dtype=float)
        for i, node in enumerate(nodes):
            mass = float(node.get("mass_kg", 0.0))
            half = tuple(float(v) for v in (
                node.get("body_half_extent_m") or ()))
            if len(half) == 3:
                a2, b2, c2 = (v * v for v in half)
                inertia = mass * np.asarray(
                    (b2 + c2, a2 + c2, a2 + b2)) / 3.0
            else:
                inertia = np.zeros(3, dtype=float)
            target = node.get("solver_condensed_into")
            if target:
                j = int(self.solver_master_index[i])
                offset = self.position[i] - self.position[j]
                if node.get("solver_condensed_mass", False):
                    self.solver_node_mass[j] += mass
                    self.solver_node_inertia[j] += inertia + mass * np.asarray((
                        offset[1] ** 2 + offset[2] ** 2,
                        offset[0] ** 2 + offset[2] ** 2,
                        offset[0] ** 2 + offset[1] ** 2))
                node["solver_mass_condensed_kg"] = mass
            elif self.structural_node[i]:
                self.solver_node_mass[i] += mass
                self.solver_node_inertia[i] += inertia
            elif mass:
                node["solver_mass_ignored_kg"] = mass

    def _master_dofs(self, endpoint_index: int) -> list[int]:
        master = int(self.solver_master_index[endpoint_index])
        return list(range(master * DOF_PER_NODE,
                          (master + 1) * DOF_PER_NODE))

    def _assemble_endpoint_matrix(self, matrix: np.ndarray, ia: int, ib: int,
                                  element: np.ndarray) -> None:
        """Assemble a 12x12 matrix through rigid surface-point MPCs."""
        indices = (ia, ib)
        for p, ip in enumerate(indices):
            Pp = self.solver_endpoint_transform[ip]
            dp = self._master_dofs(ip)
            for q, iq in enumerate(indices):
                Pq = self.solver_endpoint_transform[iq]
                dq = self._master_dofs(iq)
                block = element[p * 6:(p + 1) * 6, q * 6:(q + 1) * 6]
                matrix[np.ix_(dp, dq)] += Pp.T @ block @ Pq

    def _add_endpoint_wrench(self, vector: np.ndarray, endpoint_index: int,
                             wrench) -> None:
        transformed = (self.solver_endpoint_transform[endpoint_index].T
                       @ np.asarray(wrench, dtype=float))
        vector[self._master_dofs(endpoint_index)] += transformed

    def _endpoint_motion(self, displacement: np.ndarray,
                         endpoint_index: int) -> np.ndarray:
        return (self.solver_endpoint_transform[endpoint_index]
                @ displacement[self._master_dofs(endpoint_index)])

    def endpoint_motion(self, displacement) -> np.ndarray:
        """Materialise every graph point from the solver's master bodies.

        Condensed surface ports own no independent DOFs.  Their translation
        is the master translation plus ``theta x offset`` and their rotation
        is the master's rotation.  Keeping this gather here makes live
        rendering, strain recovery and joint kinematics use the same MPC as
        stiffness/load assembly instead of reading the ports' fixed zero
        slots and inventing strain against an invisible world anchor.
        """
        flat = np.asarray(displacement, dtype=float).reshape(-1)
        master_motion = flat[self.solver_master_dof]
        return np.einsum("nij,nj->ni", self.solver_endpoint_transform,
                         master_motion)

    def endpoint_motion_batch(self, displacements) -> np.ndarray:
        """Materialise several complete states through one MPC gather."""
        values = np.asarray(displacements, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.document["nodes"]) * DOF_PER_NODE:
            raise ValueError("endpoint motion batch must have shape (states, dofs)")
        master_motion = values[:, self.solver_master_dof]
        return np.einsum("nij,snj->sni", self.solver_endpoint_transform,
                         master_motion)

    def _add_endpoint_lumped_mass(self, diagonal: np.ndarray,
                                  endpoint_index: int, mass: float,
                                  rotary: float = 0.0) -> None:
        master = int(self.solver_master_index[endpoint_index])
        base = master * DOF_PER_NODE
        diagonal[base:base + 3] += mass
        offset = self.position[endpoint_index] - self.position[master]
        diagonal[base + 3:base + 6] += rotary + mass * np.asarray((
            offset[1] ** 2 + offset[2] ** 2,
            offset[0] ** 2 + offset[2] ** 2,
            offset[0] ** 2 + offset[1] ** 2))

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
            second = max(float(d.get("second_moment_y_m4", 0.0)),
                         float(d.get("second_moment_z_m4", 0.0)),
                         float(d.get("second_moment_m4", area * area
                                     / (4.0 * math.pi))))
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
            return (e, float(mat.shear_pa), area, second, second,
                    2.0 * second,
                    1.0, mat)
        d = edge["damage"]
        alloy = d.get("material", "4130n")
        mat = MATERIAL_BY_KEY.get(alloy, MATERIAL_BY_KEY["4130n"])
        area = float(d.get("section_area_m2", 1e-4))
        second = float(d.get("second_moment_m4", area * area / (4.0 * math.pi)))
        iy = float(d.get("second_moment_y_m4", second))
        iz = float(d.get("second_moment_z_m4", second))
        # a circular tube's radii, back out of area and second moment
        ro = float(edge.get("radius", 0.012))
        ri = max(0.0, math.sqrt(max(ro * ro - area / math.pi, 0.0)))
        return (float(d.get("youngs_modulus_pa", mat.youngs_pa)),
                float(d.get("shear_modulus_pa", mat.shear_pa)),
                area, iy, iz, float(d.get("torsion_constant_m4",
                                          2.0 * second)),
                shear_coefficient(ro, ri), mat)

    def _fixed_dofs(self) -> list:
        """Nodes bolted to the world. Their six freedoms are removed --
        which is what makes the system solvable at all: a structure with
        no support has six rigid-body modes and a singular stiffness."""
        fixed = []
        for n in self.document["nodes"]:
            i = self.index[n["identity"]]
            if (not self.structural_node[i] or n.get("fixed_to")
                    or n.get("kind") == "structural-body-pin-frame-foot"):
                base = self.index[n["identity"]] * DOF_PER_NODE
                fixed.extend(range(base, base + DOF_PER_NODE))
        return fixed

    def _assemble_plain_springs(self, stiffness: np.ndarray | None,
                                load: np.ndarray | None = None,
                                edge_identities: set[str] | None = None) -> None:
        """Put graph-declared linear springs into the frame equilibrium.

        These edges release axial beam stiffness because their constitutive
        law owns that freedom.  A plain spring/damper (one with no piston
        geometry) is linear, so its tangent belongs directly in the global
        stiffness matrix.  Its installed preload is a real static load and is
        included when ``load`` is supplied.  Pneumatic/orifice elements stay
        with the nonlinear joint bank.
        """
        for edge in self.document["edges"]:
            if (edge_identities is not None
                    and edge["identity"] not in edge_identities):
                continue
            is_belleville = edge.get("constraint") == "belleville-preload-stack"
            is_chain_winch = edge.get("constraint") == "chain-winch-hoist"
            is_platform_preload = edge.get("part_role") == "platform-actuator"
            is_strap_tensioner = edge.get("constraint") == "strap-tensioner"
            is_linear_spring = (
                edge.get("constraint") == "spring-damper"
                and (float(edge.get("piston_area_m2", 0.0)) <= 0.0
                     or "linear_damping_n_s_per_m" in edge
                     or set(edge.get("force_components") or ()) ==
                     {"spring-recuperator"}))
            if not (is_belleville or is_chain_winch or is_strap_tensioner
                    or is_platform_preload or is_linear_spring):
                continue
            ia, ib = self.index[edge["a"]], self.index[edge["b"]]
            delta = self.position[ib] - self.position[ia]
            length = float(np.linalg.norm(delta))
            if length <= 1.0e-12:
                continue
            axis = _element_frame(edge, self.position[ia],
                                  self.position[ib])[0]
            rate = float(edge.get(
                "stiffness_n_per_m",
                edge.get("spring_rate_n_per_m",
                         edge.get("stack_rate_n_per_m", 0.0))))
            if rate < 0.0:
                raise ValueError(f"{edge['identity']}: negative spring rate")
            axial = rate * np.outer(axis, axis)
            ke = np.zeros((12, 12), dtype=float)
            ke[:3, :3] = axial
            ke[6:9, 6:9] = axial
            ke[:3, 6:9] = -axial
            ke[6:9, :3] = -axial
            if stiffness is not None:
                self._assemble_endpoint_matrix(stiffness, ia, ib, ke)
            if load is not None:
                preload = float(edge.get("preload_force_n",
                                         edge.get("spring_preload_n",
                                                  edge.get("preload_n", 0.0))))
                self._add_endpoint_wrench(
                    load, ia, np.r_[-axis * preload, np.zeros(3)])
                self._add_endpoint_wrench(
                    load, ib, np.r_[axis * preload, np.zeros(3)])

    def assemble_component_matrices(
            self, member_identities,
            *, node_mass_fractions: dict[str, float] | None = None) -> dict:
        """Assemble one component from the members it owns.

        This deliberately assembles element contributions before slicing the
        component coordinates. Slicing the station's completed global matrix
        would lose ownership at shared interfaces. ``node_mass_fractions``
        makes shared body-mass partition explicit; omitted fractions are zero.
        Member distributed mass always belongs to its member.
        """
        selected = set(member_identities)
        known = {edge["identity"] for edge in self.document["edges"]}
        missing = selected - known
        if missing:
            raise KeyError(f"unknown component member {sorted(missing)[0]!r}")
        fractions = dict(node_mass_fractions or {})
        for identity, fraction in fractions.items():
            if identity not in self.index:
                raise KeyError(f"unknown component mass node {identity!r}")
            if not math.isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
                raise ValueError(f"{identity}: node mass fraction must be in [0, 1]")

        n_dof = len(self.document["nodes"]) * DOF_PER_NODE
        stiffness = np.zeros((n_dof, n_dof), dtype=float)
        mass = np.zeros(n_dof, dtype=float)
        touched_masters: set[int] = set()
        selected_members = [edge for edge in self.members
                            if edge["identity"] in selected]
        for edge in selected_members:
            ia, ib = self.index[edge["a"]], self.index[edge["b"]]
            touched_masters.update((int(self.solver_master_index[ia]),
                                    int(self.solver_master_index[ib])))
            a, b = self.position[ia], self.position[ib]
            length = float(np.linalg.norm(b - a))
            if length < 1.0e-9:
                continue
            E, G, A, Iy, Iz, J, kappa, material = self._section(edge)
            local = _condense(
                element_stiffness(E, G, A, Iy, Iz, J, length, kappa),
                _released_dofs(edge))
            rotation = _element_frame(edge, a, b)
            transform = np.zeros((12, 12), dtype=float)
            for block in range(4):
                transform[block * 3:block * 3 + 3,
                          block * 3:block * 3 + 3] = rotation
            self._assemble_endpoint_matrix(
                stiffness, ia, ib, transform.T @ local @ transform)
            if edge.get("rigid"):
                continue
            half = A * length * material.density_kg_m3 / 2.0
            outer = math.sqrt(max(A, 0.0) / math.pi)
            rotary = half * (outer * outer / 4.0 + length * length / 12.0)
            self._add_endpoint_lumped_mass(mass, ia, half, rotary)
            self._add_endpoint_lumped_mass(mass, ib, half, rotary)

        self._assemble_plain_springs(
            stiffness, edge_identities=selected)
        # Spring-only components still own their endpoint coordinates.
        for edge in self.document["edges"]:
            if edge["identity"] not in selected:
                continue
            for endpoint in (edge["a"], edge["b"]):
                touched_masters.add(int(self.solver_master_index[
                    self.index[endpoint]]))
        for identity, fraction in fractions.items():
            i = self.index[identity]
            master = int(self.solver_master_index[i])
            if master != i:
                raise ValueError(
                    f"{identity}: assign condensed mass to its gestalt master")
            touched_masters.add(master)
            base = master * DOF_PER_NODE
            mass[base:base + 3] += self.solver_node_mass[master] * fraction
            mass[base + 3:base + 6] += (
                self.solver_node_inertia[master] * fraction)
        nodes = np.asarray(sorted(touched_masters), dtype=np.int64)
        physical_dofs = (nodes[:, None] * DOF_PER_NODE
                         + np.arange(DOF_PER_NODE)[None, :]).reshape(-1)
        return {
            "node_indices": nodes,
            "node_positions": self.position[nodes].copy(),
            "physical_dofs": physical_dofs,
            "stiffness_matrix": stiffness[np.ix_(physical_dofs, physical_dofs)],
            "mass_matrix": np.diag(mass[physical_dofs]),
        }

    def applied_force_vector(self) -> np.ndarray:
        """Constant graph loads in physical coordinates.

        This is also the runtime gravity/preload vector when a mechanism is
        deliberately started from its authored pose and allowed to settle
        through its nonlinear joint laws.  Keeping the construction here
        prevents the live engine from inventing a second gravity model.
        """
        f = np.zeros(len(self.document["nodes"]) * DOF_PER_NODE)
        self._assemble_plain_springs(None, f)
        if self.gravity:
            for edge in self.members:
                if edge.get("rigid"):
                    continue
                _E, _G, A, _Iy, _Iz, _J, _kappa, mat = self._section(edge)
                ia, ib = self.index[edge["a"]], self.index[edge["b"]]
                L = float(np.linalg.norm(self.position[ib] - self.position[ia]))
                w = (A * L * mat.density_kg_m3 * GRAVITY
                     * float(self.gravity_scale) / 2.0)
                gravity_wrench = np.asarray(
                    (0.0, -w, 0.0, 0.0, 0.0, 0.0))
                self._add_endpoint_wrench(f, ia, gravity_wrench)
                self._add_endpoint_wrench(f, ib, gravity_wrench)
            for i, _node in enumerate(self.document["nodes"]):
                f[i * DOF_PER_NODE + 1] -= (
                    self.solver_node_mass[i] * GRAVITY
                    * float(self.gravity_scale))
        for ident, force in self.loads.items():
            i = self.index[ident]
            self._add_endpoint_wrench(
                f, i, np.r_[np.asarray(force, float), np.zeros(3)])
        return f

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
            E, G, A, Iy, Iz, J, kappa, _mat = self._section(edge)
            kl = element_stiffness(E, G, A, Iy, Iz, J, L, kappa)
            # THE JOINT, APPLIED. Released freedoms are condensed out of
            # the element before it is rotated into the structure, which
            # is the difference between a recoil slide and a bar.
            kl = _condense(kl, _released_dofs(edge))
            R = _element_frame(edge, a, b)
            T = np.zeros((12, 12))
            for blk in range(4):
                T[blk * 3:blk * 3 + 3, blk * 3:blk * 3 + 3] = R
            kg = T.T @ kl @ T
            self._assemble_endpoint_matrix(K, ia, ib, kg)

        # The installed spring rate and preload participate in the same
        # equilibrium as gravity and beam deformation.  Omitting them here
        # and trying to introduce them on the first live tick creates a false
        # store of energy in a pose that was never jointly settled.
        self._assemble_plain_springs(K)
        f = self.applied_force_vector()

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
            E, G, A, Iy, Iz, J, kappa, _m = self._section(edge)
            R = _element_frame(edge, a, b)
            ua = self._endpoint_motion(u, ia)
            ub = self._endpoint_motion(u, ib)
            # into the member's own frame
            da, ra = R @ ua[:3], R @ ua[3:]
            db, rb = R @ ub[:3], R @ ub[3:]
            axial = (db[0] - da[0]) / L
            d = edge.get("damage") or {}
            outer_y = float(d.get("section_outer_y_m",
                                  edge.get("radius", 0.012)))
            outer_z = float(d.get("section_outer_z_m",
                                  edge.get("radius", 0.012)))
            curv_y = (rb[1] - ra[1]) / L
            curv_z = (rb[2] - ra[2]) / L
            if ("section_outer_y_m" in d or "section_outer_z_m" in d):
                bending = abs(curv_y) * outer_z + abs(curv_z) * outer_y
            else:
                bending = math.hypot(curv_y, curv_z) * outer_y
            shear_y = (db[1] - da[1]) / L - (ra[2] + rb[2]) / 2.0
            shear_z = (db[2] - da[2]) / L + (ra[1] + rb[1]) / 2.0
            twist = (rb[0] - ra[0]) / L
            shear = math.sqrt(shear_y ** 2 + shear_z ** 2
                              + (twist * max(outer_y, outer_z)) ** 2)
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
            E, G, A, Iy, Iz, J, kappa, mat = self._section(edge)
            kl = _condense(element_stiffness(E, G, A, Iy, Iz, J, L, kappa),
                           _released_dofs(edge))
            R = _element_frame(edge, a, b)
            T = np.zeros((12, 12))
            for blk in range(4):
                T[blk * 3:blk * 3 + 3, blk * 3:blk * 3 + 3] = R
            self._assemble_endpoint_matrix(K, ia, ib, T.T @ kl @ T)
            # A RIGID LINK WEIGHS NOTHING HERE EITHER. Its section is an
            # idealisation sized to out-stiffen real structure, so
            # A*L*rho is a body's own skin weighing thousands of times
            # what the body does. The static solve was corrected for
            # this; the eigenproblem was not, and a mass matrix wrong by
            # that much moves every frequency it reports.
            if edge.get("rigid"):
                continue
            half = A * L * mat.density_kg_m3 / 2.0
            # AND ITS ROTARY INERTIA, which is not optional.  When an end
            # is a condensed surface point, both quantities are lumped on
            # the body master with the parallel-axis contribution.
            r_o = math.sqrt(max(A, 0.0) / math.pi)
            rotary = half * (r_o * r_o / 4.0 + L * L / 12.0)
            for i in (ia, ib):
                self._add_endpoint_lumped_mass(M, i, half, rotary)
        # Plain graph springs close the axial freedoms their released beam
        # members deliberately leave open.  They must be present before the
        # eigensolve or genuine spring-supported motion is misclassified as a
        # zero-frequency mechanism.
        self._assemble_plain_springs(K)
        for i, n in enumerate(self.document["nodes"]):
            m_n = self.solver_node_mass[i]
            M[i * DOF_PER_NODE:i * DOF_PER_NODE + 3] += m_n
            # A NODE'S OWN ROTARY INERTIA, off the extent it declared.
            # A body of half-extents (a, b, c) has I = m(b^2+c^2)/3 and
            # its permutations; every node here carries `half_extent_m`
            # because the mesher needs it to draw the thing, so this is
            # read rather than guessed.
            if m_n > 0.0:
                M[i * DOF_PER_NODE + 3:i * DOF_PER_NODE + 6] += \
                    self.solver_node_inertia[i]
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
        raw_eigenvalues = vals.copy()
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
                "mechanism_eigenvalues": raw_eigenvalues[~keep],
                "eigenvalue_scale": float(np.max(np.abs(raw_eigenvalues)))
                if len(raw_eigenvalues) else 0.0,
                "mechanism_count": int(mech.shape[1]),
                "free": free,
                # The live solver advances the complete physical-coordinate
                # beam system. Returning the assembly that produced the
                # eigensystem prevents it rebuilding a second K.
                "stiffness_matrix": K,
                "lumped_mass": M,
                "modal_mass": np.ones(len(freqs))}
