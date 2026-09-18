# Handheld portable vacuum pump: infrastructure survey

Research only. Nothing in this repo was modified to produce this report.

## Summary

Yes, a real player-tool / inventory / equip system exists, but it lives in
`turing/src/compiler/abstract_ui_tools.py` (the AbstractUI compiler layer),
not in `engine_toy/`. It is a genuine `EntityInventory` / `InventoryItem` /
`ActiveTool` / `Hotbar` model with `equip()`/`remove()`/`add()` methods, and
`engine_toy/world_gun.py` already builds an `AbstractUITool` (a ballistic
test gun with fire/cycle-calibre hooks and per-slot `Attachment` objects that
convert to `InventoryItem`s) on top of it. That matters for this task because
it means the handheld vacuum pump does not need a new interaction substrate
invented from nothing -- it needs the same `AbstractUITool` + `InventoryItem`
wiring `world_gun.py` already demonstrates, aimed at a port/closure instead
of a ray. What's genuinely missing is the physical-world half: nothing
connects an `InventoryItem` to a `PartPort`/vessel-port `closure`, and there
is no electrical model at hand-tool scale (everything electrical in
`engine_toy/` is sized for vehicles, plant, or stations, down to a 20 Ah
"keep-alive" battery at the smallest).

---

## 1. Player tools / interactable objects

**Exists, but in `turing/`, not `engine_toy/`.**

- `turing/src/compiler/abstract_ui_tools.py` -- the real thing. Key classes:
  - `InventoryItem` (line 36): `identity`, `entity`, `name`, `is_tool`,
    `color`, `properties`, `slot`, `quantity`, `maximum_stack`, `stack_key`.
    Stackable, slotted, frozen dataclass.
  - `ActiveTool` (line 54): `inventory`, `item`, `entity` -- what's
    currently equipped.
  - `ToolHook` (line 61) / `ToolMode` (line 75): a tool declares
    primary/secondary action hooks and named modes (e.g. "hold secondary to
    charge, fire on release").
  - `AbstractUITool` (line 143): identity, name, hooks, optional dialogue,
    modes, default mode. `to_data()` serializes it.
  - `EntityInventory` (line 222): the actual inventory container --
    `add()`, `remove()`, `item()`, `equip()` (line 259, raises if the item
    isn't `is_tool`), `clear_tool()`. Enforces one-based slots, no duplicate
    entities, stack-size validation in `InventoryItem.__post_init__`.
  - `Hotbar` / `HotbarSlot` (line 328): a 10-slot editable view onto an
    inventory's first ten slots, with `from_inventory()`.
  - Two example tools already built on this: `form_tool()` (aesthetic
    property editor with presets) and `depth_map_tool()` (terrain
    sculpting, sculpt/middle modes).
  - `EntityInventory(...)` and `.equip(...)` are consumed for real in
    `turing/src/compiler/abstract_ui_div_map.py` (rendering/serialization)
    and covered by `turing/tests/test_abstract_ui_tools.py`. So this is not
    dead code -- it's wired into the compiler's UI-document pipeline.

- `engine_toy/world_gun.py` -- **the one engine_toy consumer**, and the
  closest existing precedent for "a tool the player carries in the world"
  (its own docstring, line 1). Concretely:
  - `ballistic_gun_tool()` (line 60) builds an `AbstractUITool` with one
    `ToolMode` per calibre and two hooks (`fire-ballistic-ray`,
    `cycle-calibre`) -- imports `AbstractUITool, ToolHook, ToolMode` from
    `turing`'s `abstract_ui_tools` directly (line 68).
  - `Attachment` (line 93): `identity`, `slot` ("optic"|"muzzle"|"mount"|
    "magazine"), `name`, `note`. `.to_inventory_item(owner)` (line 100)
    converts it to a real `InventoryItem` -- the docstring is explicit:
    *"the same object as the world's own inventory sees it, so an
    attachment is a thing you carry and fit rather than a flag."*
  - `WorldGun` (line 178): `.attach(attachment)` / `.detach(slot)` (lines
    190, 193) -- a genuine slot-keyed attach/detach API, one item per slot,
    returns the removed object. This is the single cleanest **removable,
    player-attachable object** pattern anywhere in the repo (see item 3).
  - `WorldGun.fire()`/`fire_scoped()` actually ray-casts into a real engine
    mesh (`engine_rays.RayMesh`) and records penetration/damage
    (`damage_state.record_penetration`).
  - Real consumers: `targets.py` (imports `WorldGun`, `record_penetration`,
    `ShotReport`), `turret_demo.py` (`gun_for_world_object`),
    `symbolic_parts.py` (`_trajectory_artifact`). Not a demo stub -- it's
    load-bearing for the ballistics/targets subsystem.
  - It is also **wired to live mouse input**: `main_pygame.py` lines
    277-309 do real screen-ray picking (`view.pick_ray`, `RayMesh.pick`)
    on left-click and fire a calibre-selected shot on right-click, with a
    row of on-screen weapon-select buttons (`WEAPON_BUTTONS`, line 113).
    That is a real click-driven interaction loop, though it fires directly
    against `engine_rays`/`calibres`, not through `WorldGun`/the
    `AbstractUITool` hooks -- the two "gun" implementations (world_gun.py's
    `WorldGun` and main_pygame.py's inline right-click handler) are
    parallel, not the same code path.

- **No generic player entity, no avatar, no "carry list" at the engine_toy
  simulation layer.** Nothing in `engine_toy/` has a class named `Player`,
  `Toolbelt`, `Handheld`, `Carryable`, `UseItem`, or similar as a runtime
  concept. The word "player" shows up only as: (a) references to `turing`'s
  AbstractUI actor/action system (`abstract_ui_actions.IssuedAction` has an
  `actor` field; `rig_editor.py`'s docstring says the validator rig's
  `place_validator_support` "installs a body/world support through the same
  actions as a player" -- meaning the design-time editor issues the same
  kind of document action a player would, not that there's an avatar), and
  (b) plain English ("a player deliberately baking more into one crate" in
  `ENGINE_TOY_ARCHITECTURE_NOTES.md`, meaning a human developer).

- **One existing "pull a real part out and it leaves with you" mechanic**,
  worth knowing about because it's conceptually the smallest possible
  precedent for "the player is now holding something that used to be part
  of the machine": `main_pygame.py` lines 381-388, bound to the `M` key.
  It toggles `sim.engine.exhaust_system.catalyst_fitted` (a plain bool on
  `engines.py` line 821) and logs *"catalytic converter PULLED -- in your
  hand: {displacement*0.8:.1f} L brick"*. Pulling it has real physics
  consequences (backpressure changes, `station_services.py`'s
  `ScrapClass` for `"catalytic-converter"` knows its precious-metal scrap
  value, `node_effects.py` re-routes exhaust behaviour). **But it is only a
  boolean flag with flavor text** -- there is no actual `InventoryItem`
  created, nothing is added to any inventory, and "in your hand" is a log
  string, not a data structure. It is the right shape of consequence
  (removing a part changes the machine's real physics) with none of the
  carrying infrastructure behind it.

- `bay_view.py`, `rig_editor.py`, `chassis_editor.py`, `control_panel.py`,
  `live_scene.py` are authoring/viewer tools (camera navigation, part
  selection, aesthetic dialogues) for a human developer building or
  inspecting a machine -- not player-in-world tool usage.

---

## 2. Electrical / power wiring for small tools

**`dc_power.py` is real and is the right reference, but nothing at
hand-tool scale exists anywhere.**

- `dc_power.py` docstring (lines 1-26) lays out exactly what the task
  description says: `Conductor` (copper of an AWG gauge and length, real
  resistance via resistivity + temperature coefficient, lines 44-72),
  `PowerPort` (a connector that may or may not be `present`, with its own
  `contact_resistance_ohm` and `corrosion`, lines 76-112), `Harness` (a
  source-to-load path through conductors and ports, `route()` picks
  exterior-vs-onboard based on which ports are actually fitted and
  healthy, lines 116-161+), and `DCBattery` (chemistry-aware capacity,
  internal resistance, usable depth of discharge, charge-below-freezing
  refusal for LiFePO4 -- class at line 233).
- **The `PowerPort.exterior` / `present` fields are the closest existing
  precedent for "plug in anywhere, generically."** The docstring is
  explicit about the intent (lines 79-89): *"A panel wired permanently
  into a vehicle's loom is part of the vehicle; a panel on a cord into an
  exterior receptacle is equipment, and can be moved, replaced, shared or
  left behind... a NATO slave receptacle."* This is a real, working
  concept for **optional equipment that plugs into a fixed exterior socket
  on a host machine** -- but it is still host-centric (the harness on the
  vehicle asks "is my exterior port fitted and healthy"), not tool-centric
  (there's no object representing the portable device itself that could be
  carried from host to host). `solar.py`'s `SolarKeepAlive` (lines
  302-349) is the one thing built on it, and even that is deliberately
  described as needing "no new harness, no exterior port and no
  supervisor" (line 315) -- i.e. it plugs in at the isolator, not through
  `PowerPort`.
- **Everything electrical in this repo is vehicle/plant/station scale.**
  Surveyed every battery/capacity model in the repo
  (`battery_bank.py:BatteryBank`, `dc_power.py:DCBattery`,
  `electrical_network.py:Battery`, `hcu.py` 500 Ah accessory bank,
  `plant.py:AccessoryBank` 180 Ah default, `station.py:EmergencyBattery`
  55 Ah, `station_powerplant.py` 200 Ah, `solar.py:SolarKeepAlive`'s own
  battery). The **smallest capacity anywhere in the codebase is 20 Ah**
  (`solar.py` line 349, the "keep-alive" battery -- sized for standby
  monitoring loads of a few watts, not a tool). There is nothing at
  mAh scale, nothing with a small brushed/BLDC hand-tool motor model, no
  existing "low-amperage, slow-but-portable" tradeoff anywhere. A grep for
  `cordless`, `hand-held`/`handheld tool`, `battery-powered`, `mAh`,
  `milliamp` across all of `engine_toy/` returned nothing.
- `compressors.py`'s `VacuumPump.motor_w` defaults to 750.0 W (a "small
  shop pump," per its own comment at line 428) -- still two to three
  orders of magnitude above what a genuinely handheld battery tool would
  draw. There's no smaller preset or "mini" variant declared anywhere.

---

## 3. Port-plugging mechanics

**Closures are declared statically at graph-build time. No runtime
remove/plug-in API exists for a `PartPort`/vessel-port closure. The
closest thing to "removable and player-attachable" is `WorldGun`'s
attachment slots (see item 1), which is a different object entirely.**

- `assembly_ports.py`'s `PartPort` (line 54) has a `closure: str = ""`
  field with a long, careful comment (lines 64-76) about exactly the
  concept the task is asking about: *"WHAT IS ACTUALLY IN THE HOLE when
  nothing is plumbed to it... Anything with a closure can have it removed,
  and then the hole behaves like any other hole: pressure behind it, fluid
  out of it."* This is currently used only for engine oil-gallery ports
  (screw-cap, gallery-plug, dipstick, drain-plug -- see `part_ports()`,
  lines 121-212) and the closure string is set once, in code, when the
  port list is built. **Nothing in `assembly_ports.py` reads or mutates
  `closure` after construction** -- there is no `remove_closure()`,
  `plug()`, or similar method. "Removed" is a concept the comment names but
  no code implements for this class.
- `cabinet.py`'s `Wiring.port()` (line 260) is the vessel-port helper used
  by `cabinet.py`, `autoclave.py`, and now `climate_separator_production.py`.
  It welds a `"vessel-port"` graph node with `closure=closure` as a
  **build-time keyword argument** (line 267): `built.port("ceiling",
  "purge", "purge-valve", 0.012)` in `climate_separator_production.py` line
  198 declares the purge port's closure as `"purge-valve"` once, forever,
  when the cabinet is authored. Same pattern for door closures
  (`f"{door.kind}-clamped-door"`, line 540) and glove-port closures
  (`"glove-sleeve"`, line 581). This is graph authorship, not runtime
  state -- there's no code path where a cabinet's already-built graph gets
  a port's closure swapped out during simulation.
- `climate_separator.py`'s `vacuum_pump: object | None = None` field
  (line 156, referenced in the task) is the runtime hook the physics layer
  actually reads (`purge_open` branch, lines 193-212): if a `VacuumPump` is
  present, the purge floor is `vacuum_pump.ultimate_pa()`; if not, it's
  plain atmosphere. This is a plain Python attribute assignment, not a
  port/closure at all -- whoever builds the `ClimateChamber` just sets
  `.vacuum_pump = compressors.VacuumPump(...)` or leaves it `None`. There
  is no notion of this being "plugged into" the `purge` port specifically;
  it's a sibling field on the chamber object that happens to gate the same
  purge behaviour.
- `bench.py`'s `Port` class (line 50) is a different, simpler thing:
  supply/return connections on a `FittedItem` (hydraulic or pneumatic),
  with `is_inlet`/`satisfied` (`connected_to is not None`). This is the
  mechanism `equipment.py`'s `EquipmentRig` uses to find
  `unsatisfied_inlets()` (line 94) so a bench supply can make up the
  shortfall automatically. It models "is something connected," not "what
  specifically is plugged in and can it be unplugged and carried away."
- `servicing.py` is exactly as relevant as the task suspected, but for a
  different reason than "removable closures": it models **an
  operator/technician's held interaction**, not a physical
  plug/attachment. `Need` (line 256) is "something, somewhere, wants
  something brought to it" (fuel, an element, regeneration, a charge) --
  and critically, `TouchJob` (line 390) is *"a held interaction at a
  place, that finishes after a delay"* with `.work(dt)` and `.interrupt()`
  (lines 412-425). The module's own comment (lines 367-386) is explicit
  that this is meant to be **the one shape every clearing/service
  interaction in the plant reduces to**, and that it is deliberately the
  full handoff to a future interaction/pathing system: *"This module does
  not schedule anybody -- it says what is owed."* No pathing, no crew AI,
  no player-execution of a `TouchJob` exists yet anywhere in the repo (a
  full-repo grep for consumers of `touch_jobs`/`TouchJob` outside
  `servicing.py` itself found none) -- it's declared infrastructure with
  zero runtime consumers today. This is genuinely the right conceptual
  cousin of "player walks up, plugs in a tool, and it takes time" (a
  `TouchJob`-shaped interaction fits a "hold the pump against the purge
  port for N seconds while it draws down" mechanic well), but it has never
  been connected to anything resembling a port/closure or a carried item.
- **Best existing precedent for "removable, player-attachable" full stop**:
  `world_gun.py`'s `WorldGun.attach()`/`.detach()` (lines 190-193, see item
  1). Slot-keyed, one item per slot, returns what was removed, and the
  removed/attached object is a first-class `Attachment` that can become a
  real `InventoryItem`. It attaches to a **gun**, not a **port on a
  machine**, but the shape (a dict of slot -> object, with attach/detach
  methods) is exactly transferable to "what's plugged into this vessel
  port right now."

---

## 4. Existing small/portable vacuum or suction devices

- `compressors.py:VacuumPump` (line 420, discussed in the task) is the only
  real staged vacuum pump in the repo: `displacement_m3_s`, `clearance_frac`,
  `stages`, `motor_w`, optional `gas_ballast`, with `ultimate_pa()`,
  `speed_m3_s()`, `pumpdown_s()` (numerically integrated over the real
  falling-speed curve, not the textbook constant-speed formula -- see the
  docstring at lines 473-482), and `power_w()`/`peak_power_w()` (power
  peaks partway down the pumpdown, not at the start -- lines 501-518).
  Default is explicitly "a small shop pump" (~14 m3/h) at 750 W, still far
  larger than anything handheld.
- No other vacuum/suction device exists anywhere in `engine_toy/`. Grepped
  for other pump/suction classes; nothing else evacuates a volume against
  atmosphere. `air_movers.py` and `air_separation.py` move or separate air
  at pressure, not below it.
- **Nearest "portable, small, carryable" precedent for equipment sizing in
  general** (not vacuum, not electrical): `air_separation.py`'s
  `CYLINDERS` dict (lines 456-469) declares three named gas-cylinder sizes
  including `"portable-2l"` (2 L, 200 bar, 3 kg, with the comment *"a few
  hundred litres of gas: minutes of breathing, or one small weld"* --
  explicitly sized around what "somebody has to carry"). That's a useful
  pattern reference for how this codebase already expresses "the same
  class of equipment, but a small carryable variant of it" as a sibling
  entry in a spec dict (`CylinderSpec`) rather than a new class -- exactly
  the move a handheld `VacuumPump` preset would want (a `"handheld"`
  entry alongside whatever named presets `VacuumPump` currently has none
  of -- it's just instantiated directly with kwargs today, no preset dict).

---

## 5. Rendering / visual / interaction layer

**A real interaction layer exists, but it's a pygame+OpenGL dashboard
viewer pointed at one loaded machine, not a first/third-person player-in-
world game. There is no player avatar, no camera-as-body, no walk/grab
loop anywhere in `engine_toy/`.**

- `main_pygame.py` -- the real, live, keyboard+mouse-driven frontend.
  `pygame.key.get_pressed()` per-frame input for throttle/brake/etc
  (docstring lines 1-65), a full help-text control scheme, and genuine
  mouse interaction: `MOUSEMOTION` hover-highlighting of parts
  (`_update_hover_part`), left-click part picking via real screen-ray
  casting through the live OpenGL camera (`view.pick_ray`,
  `engine_rays.RayMesh.pick`, lines 277-307), right-click firing a
  calibre-selected shot along the same ray (lines 308+), and an on-screen
  weapon-select button row (`WEAPON_BUTTONS`, `_weapon_hit`).
- `engine_gl_view.py` -- the real OpenGL rendering of the engine mesh
  (`EngineGLView`, referenced throughout `main_pygame.py`); `gl_compositor.py`,
  `gl_text.py`, `mesh_visualizer.py` round out a real, working small
  render stack (not a stub).
- `engine_rays.py` -- real ray/mesh intersection (`RayMesh`, `Ray`) used
  both for mouse-picking and for `world_gun.py`'s ballistic penetration.
- `bay_view.py`, `chassis_editor.py`, `rig_editor.py` -- separate, simpler
  camera-navigation viewers/editors for a human developer (arrow-key
  camera, `[`/`]` to cycle engines, construction-stage editing). These are
  authoring tools, explicitly documented as living inside "the rig" (a
  `turing`-side validator/staging concept), not player-in-world gameplay.
- `control_panel.py` -- enumerates an engine's real physical switches
  (guarded toggles, wire-sealed emergency switches) and binds them to
  getters/setters on the sim; it's a UI-adjacent module but still keyboard/
  programmatic, not spatial interaction.
- **What does not exist**: any avatar/player-body entity, first-/third-
  person camera-as-character, a walk-around-the-world loop, a "grab" or
  "use" verb applied to a nearby object by proximity, or any code that
  treats the player as something other than "whoever is driving the
  keyboard and mouse for the one loaded machine." The `AbstractUITool`/
  `EntityInventory` system in `turing/` (item 1) is clearly designed with
  a player-in-world game in mind (it has hotbars, equip, stackable items),
  but nothing in `engine_toy/`'s actual pygame runtime instantiates an
  `EntityInventory` or reads an `ActiveTool` -- it's available
  infrastructure, not yet consumed at the simulation-runtime layer the way
  it's consumed in `turing`'s own document/UI pipeline.

---

## RECOMMENDED PATH

This is a sketch, not code, and not a claim that any of it is quick.

**A. Represent the tool as data, using infrastructure that already exists.**
Build the handheld pump the same way `world_gun.py` built the ballistic
gun: a small `compressors.VacuumPump` instance (deliberately small
`displacement_m3_s`, low `motor_w` -- there's no preset for this today, so
add one, the way `air_separation.CYLINDERS["portable-2l"]` is a small
sibling entry rather than a new class) wrapped in an `AbstractUITool` (from
`turing.src.compiler.abstract_ui_tools`) with hooks like
`"connect-to-port"` / `"disconnect"`, and represented as a real
`InventoryItem` (`is_tool=True`) so it can live in an `EntityInventory` and
be the `ActiveTool`. This reuses `EntityInventory.equip()`/`.remove()`
rather than inventing a parallel carry-list.

**B. Give ports a real, removable closure -- the missing piece.** Today
`closure` on `PartPort` (`assembly_ports.py`) and on cabinet/autoclave
vessel-ports (`cabinet.py:Wiring.port()`) is a string baked in at graph-
build time with no runtime mutator. The smallest new infrastructure this
task needs is exactly that mutator: something like a `plugged: object |
None` companion to `closure` (or reuse `closure` itself as a live field
that can change from `"purge-valve"` to `"handheld-vacuum-pump:<id>"` and
back), plus `plug(port, item) -> bool` / `unplug(port) -> item` functions
that check compatibility (bore, kind, fluid) the way `mate_ports()` already
checks kind/tolerance/facing for permanent mating joints. `WorldGun.attach()`/
`.detach()` (`world_gun.py` lines 190-193) is the direct template: slot-
keyed dict, attach returns nothing/replaces, detach returns the removed
object. For the climate separator specifically, the natural target is the
`purge` vessel-port itself (`climate_separator_production.py` line 198)
rather than the `ClimateChamber.vacuum_pump` field, since the field is a
runtime physics shortcut and the port is the actual declared place in the
graph a "plugged in" object belongs.

**C. Wire the plug/unplug action through a `servicing.TouchJob`.**
`servicing.py`'s `TouchJob` (line 390) already models "somebody walks up,
holds an interaction at a place, and it resolves after a delay, and can be
interrupted" -- precisely the shape of "plug the pump in and wait for
`pumpdown_s()`." Rather than inventing a new held-interaction primitive,
build a `touch_job_for_plug(port, pump)`-style helper alongside the
existing `touch_job_for_task`/`touch_job_for_need` (lines 428-448), with
`seconds` driven by `VacuumPump.pumpdown_s(chamber_volume_m3)` (genuinely
slow for a small low-power pump, which is the whole point per the task).
This also means the "slow because it's small" tradeoff needs no new
physics -- `pumpdown_s()` already produces it once a small enough
`displacement_m3_s`/`motor_w` preset is declared.

**D. Electrical: decide if the low-amperage tradeoff needs a real circuit
or can stay abstract.** Nothing in the repo currently models a hand-tool-
scale battery/motor circuit (smallest existing battery is 20 Ah, smallest
motor 750 W). Two honest options: (1) skip real amperage modelling for v1
and let `motor_w`/`displacement_m3_s` alone encode "small and slow" (the
`VacuumPump` class doesn't require a battery at all -- `power_w()` is
already derived from the pumping physics, not drawn from a stored charge);
or (2) if battery life during use should matter, a genuinely new small
`DCBattery`/motor pairing would be needed -- `dc_power.py`'s `DCBattery`
class works fine at any capacity (nothing in it assumes vehicle scale), so
this is "declare a small one," not "build new machinery." Given the task's
framing (slow because small/low-power, not "runs out"), option 1 is
probably the right scope for a first pass.

**E. Interaction/render layer: there is no player avatar to hook into.**
Whatever "player walks up and plugs this in anywhere" ends up meaning at
runtime, it cannot reuse an existing walk/grab loop, because none exists.
The realistic near-term integration point is the same one `main_pygame.py`
already uses for its right-click gun: mouse-ray picking against the live
mesh (`engine_rays.RayMesh`/`view.pick_ray`) to select a port, then invoke
whatever plug/unplug + `TouchJob` machinery items B/C produce. A real
player avatar with proximity-based "use" is out of scope for a first
version and would be new infrastructure on top of everything above, not a
prerequisite for it -- the click-to-target pattern already proven in
`main_pygame.py` is sufficient to demonstrate the mechanic end to end.
