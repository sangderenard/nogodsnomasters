# Electrical hardware catalogue and game integration

**Status: integrated in `engine_toy`.** The original patch was treated as an
authoring brief. Its catalogue, construction evidence, tests and export data now
use the game's existing `Machine`, electrical-service, conductor, thermal-domain
and circuit-registration contracts. No competing physics/object engine was added.

Target reviewed: `sangderenard/nogodsnomasters`, branch `nogodsnomasters`, commit
`d2a66def49fc8f75ff2dcbdea56db0d7b6e5cddf` (23 September 2026).

## What is present

`hardware_construction.py` defines evidence-bearing manufacturing records:
individual cores, per-core insulation and plating, outer layers, hierarchical
strand/pair/overall/tape/reel lays, contacts, terminations and environmental seals.
The records serialize into ordinary graph attributes. They do not integrate time
or replace a material bank. The single-helix length helper is geometry, not a new
conductor or circuit solver; it is not applied to nested lays as if their lengths
could be multiplied indiscriminately.

`hardware_catalogue.py` contains **7 initial cable specimens and 6 connector
specimens**. The order is home, military, industrial. Some are published
construction references; flexible service cables and mating plug counterparts
are explicitly authored game specimens. Every parameter distinguishes published,
derived, authored, and unknown data.

`electrical_hardware.py` contains builders for the existing `Machine`,
`MachinePart`, `MachineLine` and `PartPort` classes: individual connectors, routed
cables, a duplex receptacle, breaker-panel interiors, a passive patch panel and a
composed service cord. Upstream `dc_power.Conductor` supplies the coarse supported-
AWG copper resistance. No copied conductor solver is included.

Power specimens select an explicit installed `ElectricalService`, kept separate
from the cable or connector's maximum rating. Supported power cables emit the
same conductor-row ABI consumed by the existing electrical and thermal engines.
Composed cords terminate on the connector contact bank's role-qualified buses;
the individual contact bodies retain the matching bus identity for inspection,
rendering and future mating interactions.

`examples/export_electrical_hardware.py` exports the construction catalogue without
runtime dependencies, or exports graph documents through the real game classes
when run in the user's complete workspace.

The JSON catalogue includes all provenance and an explicit list of unresolved
facts. It is generated from the Python definitions, not maintained separately.

## Verification

The focused authoring and real-workspace integration suite passes **81 tests**.
The tests use the actual `Machine`, `PartPort`, `Conductor`, electrical graph
audit and `ElectricalDeviceRegistry`; no replacement runtime is used. A complete
export produces the catalogue plus **19 real Machine graph documents**.

The passing tests cover evidence handling, JSON serialization, invalid inputs,
individual helix geometry, routed length, construction separation, source IDs,
contact identity/role matching, Class L contact sizes and engagement groups,
protection-conductor distinctions, installed-service separation, circuit
registration, panel slot/phase mapping and patch-channel mapping. They do not
certify the accuracy of every authored dimension.

## Running

```powershell
python -m pytest engine_toy/tests/test_hardware_construction.py engine_toy/tests/test_electrical_hardware_integration.py -q
python engine_toy/examples/export_electrical_hardware.py --output-dir hardware_exports --construction-only
python engine_toy/examples/export_electrical_hardware.py --output-dir hardware_exports
```

Example in the normal game import environment:

```python
from hardware_catalogue import cable_catalogue, connector_catalogue
from electrical_hardware import build_cord

cables = cable_catalogue()
connectors = connector_catalogue()
cord = build_cord(
    cables["military.flex.6-5"],
    connectors["military.class-l32-12.plug"],
    connectors["military.class-l32-12.receptacle"],
    identity="site.cable.001",
    points=((0, 0, 0), (2, 0, 0), (2, 0, 4)),
)
graph = cord.build_graph()  # Existing Machine serialization.
```

A `build_cord` result preserves both physical contact identities and the existing
runtime's role-qualified terminal buses. Its mechanical endpoint and connector
body nodes are not equipotential electrical junctions: roles remain separate in
the bus identity, so poles are not shorted by sharing a body.

## Fidelity boundaries — important

This is **not the finished home-to-military-to-industrial catalogue**, and is not a
ready-to-use, manufacturer-exact asset library. The current geometry is explicitly
coarse preview geometry. Socket openings, blade shapes, bayonet/thread profiles,
individual strands, panel cutouts, hinges and detailed constituent volumes are not
fully meshed. Part dimensions and masses that are authored preview choices say so.
The Class L wall-receptacle reference dimensions do not get assigned to its plug.

The selected Class L insert arrangement and contact size designations were read
from the manufacturer's drawing. Its master keyway and frequency insert rotation
remain unresolved: the catalogue must NOT print a complete MS part number or claim
interchangeability with actual field hardware until those are selected. The mating
gate checks explicit game interface keys and contact-role maps, not certification,
voltage compatibility, coupling travel, synchronization or permission to energize.

Breaker-panel internals include distinct handles, contacts, bimetal, magnetic trip,
latch and arc-chamber declarations; the main has declared contact paths as well.
There is no live trip curve, reset mechanism, contact opening event or arc solver
bound yet. Accordingly those unresolved paths are construction declarations, not
live `transport_domain="electrical"` edges; they cannot become accidental closed
circuits in the existing electrical solver. These are generic authored panels,
NOT accurate reproductions of a
particular residential load center, M200/PDISE panel or industrial switchboard.
Panel schedules come from the same slot declarations as their components. The
chosen bus pattern shares a phase across a row and alternates phases down rows.

The patch panel is a generic passive 8P8C feedthrough model with independent pins,
not an Ethernet switch, a completed IDC/PCB layout, an RF qualification, a PoE
certification or a signal simulation. Actual flexible patch cords, optical patching,
coaxial hardware, switchgear, transfer equipment and most domestic protection
variants remain additional catalogue work.

Seal boundaries and qualification evidence are present. Ingress, cap movement,
compression set, corrosion, water tracking and contact degradation are not live
new behaviors. A stored `IP67` or immersion-test record is not a permeability law.
The industrial receptacle's published qualifications are not inherited by its
unverified plug counterpart. Its tested assembly conditions remain unresolved.

Current power-cable reduction uses actual routed axial length and the existing
copper model; it does not silently claim to include unmeasured strand/pair excess
length, skin/proximity losses, dielectric loss or jacket thermal resistance. The
24 AWG data cable is outside the upstream AWG lookup reviewed, so its construction
exports but power reduction is explicitly unbound rather than falling back to a
wrong gauge. A published maximum DCR is retained as a bound, not treated as nominal.
Cable mechanical mass distribution and jacket/strand mechanics are not bound yet.

Unknown polymer/material grades are references awaiting research; there is no
new material-property table full of invented constants. Coarse parts declare
`thermal_domain="none"` until the relevant physical material and geometry are
resolved. The generic thermal capacity inherited from `Machine.build_graph()` is
explicitly marked unbound, not asserted as a measured property.

## Reference discipline

Sources are recorded in `hardware_catalogue.SOURCES` and the generated catalogue.
For example, the Southwire page's numeric table describes **12 AWG, four insulated
conductors plus ground**. Those dimensions are not copied into the 12/2 or 14/2
specimens. Belden's insulation is recorded as the stated **polyolefin**, not guessed
as HDPE. Its 1583A is horizontal solid cable, not a flexible patch cord.

This pack is for a game/simulation authoring workflow. It is not a wiring guide,
qualified-component database or electrical-safety certification.
