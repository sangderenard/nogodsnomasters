# Starter garage

`starter_garage_catalogue.py` is the authoring ledger: 162 inventory-entry
types, their independent starting quantities, and 40 major constituent
records. A set is only an organizing label; its sockets, bits, clamps, towels,
and other contents spawn as independent objects.

`starter_garage.py` is authoring and composition only. It returns the game's
existing `Machine` objects and one existing `AirVolume`; it has no garage
runtime, material ledger, pressure solver, interaction state, or snapshot
format. `MachineSim` and the managed engine systems own runtime state. For
current convenience, every loose tool and stock item is itself a Machine.

The existing `cabinet.py` washer/dryer production graphs supply the appliance
bodies. The fixed dryer model's service components and process-air circuit are
authored into `appliance_production.py`. One of exactly 47 fault combinations
is applied to those actual component nodes and edges once from the world seed.
The fixed spare carton remains independent Machines and never changes with the
selection.

The building exhaust is a separate Machine with one graph-discovered air
circuit. Its lint is an actual part carrying the existing `fouling.py`
vocabulary. Its numerical blockage/material calibration remains explicitly
unresolved because the brief supplied no calibrated coefficient; no second
pressure law or fan curve is layered beside the fluid engine.

The garage door likewise distinguishes three facts: the powered opener drive
is immobilized, the trolley begins coupled, and the healthy counterbalanced
door can be lifted only after the red release is pulled. The absent remote does
not explain the jam.

Regenerate the checked-in manifest with:

```powershell
python engine_toy/examples/export_starter_garage.py --seed starter
```

This writes the catalogue plus every ordinary Machine graph to
`starter_garage.json`.
