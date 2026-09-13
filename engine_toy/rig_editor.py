"""The chassis editor as part of the validator rig.

The rig is already the right place for this, and reading it settles a
lot of design questions that were open a moment ago. Its own docstring
is the important sentence:

    "The rig is a world object, not a page, modal, scheduler, or second
     physics runtime."

So the editor does not get its own viewport, its own clock or its own
physics. It becomes a mode of the rig's construction stage, and the
object being edited is held in the world where you can walk around it.

WHAT THE RIG ALREADY PROVIDES

  A STAGE. `form.stage_identity` with its own offset and half extent --
  a workspace beside the rig box. That is where an object under
  construction sits.

  CLAMPS, and they are exactly what was wanted. `place_validator_support`
  installs a body/world support "through the same actions as a player",
  and its own comment is careful: "placement does not imply a second
  physics runtime or prescribe the body's motion". Bound through
  `bind_placed_rig_point`, a support is a SOFT hold -- per-axis
  stiffness and damping with a maximum force -- not a weld. Give it a
  target velocity and it drags the held body along, which is how a
  clamped object gets moved around while you work on it.

  A MATERIAL LEDGER. Parts progress `projected` -> `awaiting-material`
  -> `installed`, drawn first as faint construction-line projection and
  then with the real shaded material once they exist. An editor
  document maps onto that directly: every segment you draw is a
  projected part until it is funded, and then it is a real member.

  A TICK ENVELOPE. `execution.mode = "assigned-ticks"`, envelope
  `[tick, dt, subdt, substeps]`, dispatch order `validator_rig,
  validator, initial_vehicle`, `inactive_until_called: True`, and
  "all-consumers-receive-the-same-tick-envelope". Nothing here starts a
  clock; it is called, it prepares values, it returns.

WHAT THIS MODULE ADDS

  The translation between an editor document (chassis_editor.py) and
  those three things: parts for the construction ledger, clamps for the
  object being held, and a tick that only ever prepares values.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_TURING_ROOT = Path(__file__).resolve().parents[1] / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

# The clamp's own numbers, taken from the rig's real use in
# tools/run_vehicle_native_assembly.py rather than invented: a support
# there holds with 80/120/80 kN/m and 0.8/1.2/0.8 kN.s/m, capped at
# 60 kN. Stiffer vertically than laterally, because what it is mostly
# doing is holding weight.
CLAMP_STIFFNESS_N_PER_M = (80_000.0, 120_000.0, 80_000.0)
CLAMP_DAMPING_N_S_PER_M = (800.0, 1_200.0, 800.0)
CLAMP_MAXIMUM_FORCE_N = 60_000.0
# how much material one segment costs the hopper ledger. The rig counts
# in "physics-ball-material-unit"s that the player feeds in, so a part
# list has to be expressed in them.
MATERIAL_UNITS_PER_NODE = 1
MATERIAL_UNITS_PER_SEGMENT = 1


@dataclass
class RigClamp:
    """One body/world support holding the work in place."""
    identity: str
    node: str                       # the editor node it grips
    local_point: tuple
    world_point: tuple
    target_velocity: tuple = (0.0, 0.0, 0.0)
    enabled: bool = True

    def command(self) -> dict:
        """The binding command `bind_placed_rig_point` takes.

        `mode=1` is the held mode. A target velocity of zero means hold
        still; a non-zero one means drag the held body at that rate,
        which is how a clamped object is moved around the stage without
        anybody teleporting it."""
        return dict(enabled=1 if self.enabled else 0, mode=1,
                    target_velocity=tuple(float(v) for v in self.target_velocity),
                    force=(0.0, 0.0, 0.0),
                    stiffness=CLAMP_STIFFNESS_N_PER_M,
                    damping=CLAMP_DAMPING_N_S_PER_M,
                    maximum_force=CLAMP_MAXIMUM_FORCE_N)


@dataclass
class RigEditSession:
    """An editor document being built on the rig's stage."""
    document: object                        # chassis_editor.EditorDocument
    rig_identity: str = "world/validator-rig"
    body_identity: str = "world/validator-rig/work"
    actor: str = "player"
    clamps: list = field(default_factory=list)
    installed: set = field(default_factory=set)
    log: list = field(default_factory=list)

    # ---------------- the construction ledger ----------------
    @property
    def stage_identity(self) -> str:
        return f"{self.rig_identity}/construction-stage"

    def part_list(self) -> list:
        """The document as rig construction parts.

        Grouped the way the rig groups a vehicle -- by what a person
        would fit in one go -- rather than one part per member, because
        the ledger is what a player feeds material into and a hundred
        separate line items is not a build, it is a spreadsheet."""
        import segment_types as st
        doc = self.document
        groups: dict[str, list] = {}
        for seg in doc.segments:
            family = st.get(seg.type_key).family
            groups.setdefault(family, []).append(seg.identity)
        parts = []
        order = [f for f in st.FAMILIES if f in groups]
        for index, family in enumerate(order):
            members = groups[family]
            identity = f"{self.stage_identity}/parts/{family}"
            parts.append({
                "identity": identity,
                "name": family,
                "sequence": index,
                "material_units": len(members) * MATERIAL_UNITS_PER_SEGMENT,
                "members": tuple(members),
                "state": ("installed" if identity in self.installed
                          else "projected" if index == 0 else "awaiting-material"),
                "custody": "rig-projection" if index == 0 else "rig-recipe",
                "handoff": ("material-ball->hopper-ledger->line-projection->"
                            "rig-actuator->vehicle-installed-phong"),
            })
        return parts

    def material_required(self) -> int:
        doc = self.document
        return (len(doc.nodes) * MATERIAL_UNITS_PER_NODE
                + len(doc.segments) * MATERIAL_UNITS_PER_SEGMENT)

    # ---------------- clamps ----------------
    def clamp(self, node_identity: str) -> RigClamp:
        """Pin one node of the work in space.

        This is what makes editing possible at all: a part-built frame
        is not a structure yet, so without something holding it, it
        falls over the moment the physics touches it. A clamp is the
        hand that holds the work."""
        node = self.document.node_by_id(node_identity)
        if node is None:
            raise KeyError(f"no such node {node_identity!r}")
        c = RigClamp(identity=f"{self.rig_identity}/supports/{len(self.clamps)}",
                     node=node_identity,
                     local_point=tuple(node.position), world_point=tuple(node.position))
        self.clamps.append(c)
        self.log.append(f"clamped {node_identity} at {tuple(round(v, 3) for v in node.position)}")
        return c

    def release(self, node_identity: str) -> None:
        before = len(self.clamps)
        self.clamps = [c for c in self.clamps if c.node != node_identity]
        if len(self.clamps) != before:
            self.log.append(f"released {node_identity}")

    def drag(self, node_identity: str, velocity) -> None:
        """Move a clamped node by asking its clamp to travel.

        Note what this does NOT do: it does not set a position. The
        clamp is a stiffness and a damping with a force ceiling, so it
        pulls the work toward where it is going and the physics decides
        what actually happens -- including failing to move something too
        heavy, or bending it if it is held at both ends."""
        for c in self.clamps:
            if c.node == node_identity:
                c.target_velocity = tuple(float(v) for v in velocity)
                self.log.append(f"dragging {node_identity} at {c.target_velocity}")

    def install_on(self, rig_document, *, place_support, bind_point):
        """Install every clamp into a real rig document.

        `place_support` and `bind_point` are passed in rather than
        imported so this module can be exercised without standing up a
        whole world -- and so the caller keeps ownership of the
        document, which is the rig's own custody rule."""
        bindings = []
        for slot, c in enumerate(self.clamps):
            rig_document = place_support(
                rig_document, actor=self.actor, identity=c.identity,
                body=self.body_identity, local_point=c.local_point,
                world_point=c.world_point)
            bindings.append((slot, bind_point(rig_document, c.identity,
                                              self.body_identity, c.command())))
        return rig_document, bindings

    # ---------------- the tick ----------------
    def tick(self, envelope: dict) -> dict:
        """Called by the world's clock; owns no clock of its own.

        The rig declares `inactive_until_called` and says every consumer
        receives the same envelope, so this takes `[tick, dt, subdt,
        substeps]` and returns what it prepared. It integrates nothing.
        """
        dt = float(envelope.get("dt", 0.0))
        prepared = {
            "tick": envelope.get("tick"),
            "dt": dt,
            "stage": self.stage_identity,
            "clamps": [{"identity": c.identity, "node": c.node,
                        "command": c.command()} for c in self.clamps if c.enabled],
            "parts": self.part_list(),
            "material_required": self.material_required(),
        }
        return prepared

    def summary(self) -> list:
        parts = self.part_list()
        out = [f"  rig edit session on {self.stage_identity}",
               f"    {len(self.document.nodes)} nodes, {len(self.document.segments)} segments, "
               f"{self.material_required()} material units",
               f"    {len(self.clamps)} clamp(s) holding the work"]
        for p in parts:
            out.append(f"      {p['name']:11s} {len(p['members']):3d} members  "
                       f"{p['material_units']:3d} units  {p['state']}")
        out.extend(f"    {line}" for line in self.log[-4:])
        return out
