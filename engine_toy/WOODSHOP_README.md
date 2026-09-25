# Python woodshop slice

This is the first playable material-fabrication loop. It does not use the
Living Data Map's JavaScript. Its Python frontend does use that interface's
first-person camera contract: captured mouse look, WASD movement, a centre
crosshair, and the same perspective constants. Every frame it resolves the
live Machine parts into a geometry state and hands that state to Pluck's
existing `BaseGLRenderer` and `base_material.vert/frag.glsl` shader. It also
keeps the ten-slot hotbar, independent two-hand equipment, pickup/drop, and
press/hold/release tool actions.

Run from this directory:

```powershell
python woodshop_pygame.py
```

The starting world contains these initial Machines:

- eight kiln-dried nominal 2x4x8 pine studs, dressed to 1.5 by 3.5 by 96
  inches, with one loose and seven in a physical pile;
- one hand saw made from a steel blade body, a wood handle body, and a finite
  working edge carrying endpoints, thickness, tooth-set wave, and tooth
  pattern;
- four identical one-piece resin sawhorse brackets;
- two F-style clamps and two pipe clamps.

Both are `machines.Machine` instances running through `MachineSim` and
`MachineSystem`. The cutting engine is a `DtCompatibleEngine`; all systems
share one `StateTable`. It evaluates the executable WO4 honorary-law
pieces for cutting force, edge speed, power, removal rate, and kerf advance.
The pieces are cached LLVM artifacts; the gameplay process does not compile
them again when the matching artifact is already present.

World motion uses the real LLVM dt system. `WoodshopWorldRules` manifests the
existing N4.1, N1.2 and N1.1 equations as three full-batch LLVM pieces:
gravity, then momentum, then position. A sequential `RoundNode` states those
same-step dependencies. Each piece is called once over the complete object
columns; Newton laws are not called once per Machine or once per coordinate.
N5.1--N5.7 govern nonpenetration, normal impulse, restitution and Coulomb
friction after geometry identifies contacts from the oriented real part
boxes. Released world Machines fall and collide; held Machines remain in the
same batch with their active lane disabled. Momentum and resolved contacts
are published into the shared `StateTable`.

`WoodshopWorldRules.lower_newton_dt_system(path)` invokes the repository's
whole-dt native lowerer over the same three pieces. At the present compiler
revision, that route reaches ProcessGraph source-closure construction and
then fails in `graph_express2._resolve_ast_parent_reference` while recursively
resolving `ExtractionContract.program_abi/path`; it does not reach C emission.
The managed LLVM-piece dt graph is operational despite that distinct
whole-program compiler frontier.

Controls are shown in `woodshop_pygame.py`. Number keys equip the right hand;
Shift plus a number equips the left. Left and right mouse buttons hold the
corresponding hand's primary action. Holding the full-length stud in one hand
while sawing with the other is allowed, but its poor support creates a large
line error and sends most applied work into motion instead of material
removal.

Machines may declare `selected`, `used-1`, and `used-2` hand poses. The saw's
selected pose is held near the player; holding its action extends it and
alternates the two stroke poses. Objects without declarations receive the
generic selected pose, and the default pickup operation tries the preferred
hand and then the other free hand.

Q (or Shift+Q for the left hand) begins drop placement instead of releasing
immediately. The visible red/green/blue gimbal edits the live world-object
orientation: left-drag changes roll/yaw and right-drag changes pitch. Enter or
Q commits and releases the item; Escape cancels. Floor contact is recomputed
from the oriented Machine-part corners, so a 90-degree roll places the stud on
its narrow 1.5-inch edge rather than leaving it intersecting or hovering.

F5 saves the whole live world and F9 restores it. The checkpoint starts with
the existing dt `StateTable.snapshot()` and carries the real Machine object
graph with it, so positions, orientations, custody/hotbar state, momentum,
damage, kerfs and SDF material, chips and fission lineage, physical edges,
clamp/sleeve joints, and engine clocks return together. The default save is
stored under the user's local application-data `engine_toy` directory; the
write uses a temporary file and atomic replacement.

The workpiece remains one Machine while the saw subtracts finite capsule SDF
sweeps from its mesh-derived material kernel. Occupied volume is the mass
authority, so every removal reduces workpiece mass and increases a real
chip-form Machine's mass immediately. Cut depth alone never creates an object.
The surviving material field is tested for separate contiguous islands. While
any wood still bridges the kerf there is one object. The final subtraction
removes that bridge; the two islands have no connection of any kind. At that
instant the source Machine and divided part retire and each island is minted
with a new global identity and explicit lineage.

At world level this is one atomic node replacement. The predecessor node is
retired and two new object nodes are inserted. No edge is created between
them. The lineage ledger is separate from the physical-edge table, so ancestry
cannot make the pieces mechanically or logically attached.

## Woodworking joints and clamps

The starter object set includes two F-style bar-clamp Machines and two pipe-
clamp Machines. `WoodworkingJointInterface.auto_deploy` filters them by opening
and throat, determines count from seam spacing and required total force, then
uses WO5 to resolve screw force, pad pressure, friction capacity and frame
deflection. Each deployed clamp contributes two temporary jaw-contact edges:
member A to clamp and clamp to member B. There is never a direct member-to-
member edge. Releasing the planned joint removes both jaw edges.

## Resin sawhorse jig and compression sleeves

Four identical one-piece structural-resin sawhorse brackets are present in the
starting world. Each is one MachinePart carrying three ordinary PartPorts: a
top-beam sleeve and two splayed leg sleeves. Every sleeve accepts the dressed
1.5-by-3.5-inch profile at any stock length. Its declaration includes capture
distance, maximum penetration, compression force, surface friction, lateral
and moment capacity, a release vector, finite fastener target regions and
finite glueable surfaces. The areas do not prescribe screws or any other
particular fastener; a later selected fastener must satisfy WO3 against the
declared region. A later adhesive forms a WO3 cohesive bondline over the
declared glueable area.

With a compatible 2x4 held, primary-clicking a bracket in the other hand or
under the crosshair aligns the stock to the first available sleeve and inserts
it to the stop. WO5.3 evaluates holding force from the sleeve's real
compression and friction. The resulting world-object edge participates in
rigid condensation while it holds; it is explicitly a compression-sleeve fit,
not a weld. A force exceeding its holding capacity only when projected along
the declared pullout vector removes the edge and returns the two retained
identities to independent objects.

One active sleeve fit raises the workpiece's support quality substantially
above floor support; two hard-point fits raise it again, while a deployed
clamp remains the fully restrained case.
