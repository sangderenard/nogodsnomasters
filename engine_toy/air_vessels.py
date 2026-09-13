"""Where the compressed air actually IS.

The pneumatic side of a machine is not one tank. It is a WET TANK that
takes the compressor's hot, wet discharge and lets the last of the water
fall out of it, a RESERVE TANK downstream of the treatment train, and --
behind a pressure-protection valve -- the PRIMARY and SECONDARY brake
reservoirs, which exist precisely so that a burst service line cannot
take the brakes with it.

`drivetrain_graph.FluidCircuit` deliberately lumps all of that into one
stored mass, and that lumping is right for what it does: charging, the
starter's draw and the idle-assist dump genuinely share one body of air,
and giving each vessel its own independent reservoir would have been
four rival ledgers of the same gas -- the exact mistake the fuel-leak
bug was.

So this is not a second ledger. The circuit's total remains
authoritative and this module only says WHERE that total is sitting: it
distributes the mass across the real declared vessels, moves it between
them through the real pipe conductances, and enforces the one real
behaviour the lump cannot express -- the protection valve, which lets
the brake reservoirs fill but will not let them empty backwards into a
service leak. The sum over the vessels always equals the circuit's mass
to within float error, and `verify_conservation` says so out loud.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ATM_PA = 101_325.0
# How fast a vessel-to-vessel pipe equalises. A real 8 mm airline
# between two truck reservoirs balances in a second or two, not
# instantly -- which is why a heavy brake application really does pull
# the primary down before the reserve catches it back up.
CONDUCTANCE_KG_S_PER_PA_PER_M2 = 4.0e-2


@dataclass
class AirVessel:
    identity: str
    label: str
    capacity_kg: float                 # mass held at its working pressure
    working_pressure_pa: float
    mass_kg: float = 0.0
    protected: bool = False            # sits behind the protection valve
    feeds_from: str | None = None      # the vessel immediately upstream
    bore_m: float = 0.008

    @property
    def fill_frac(self) -> float:
        return self.mass_kg / max(self.capacity_kg, 1e-9)

    @property
    def pressure_pa(self) -> float:
        """Isothermal ideal gas in a rigid vessel: pressure is linear in
        the mass held, from atmospheric when empty to the working
        pressure when full."""
        return ATM_PA + (self.working_pressure_pa - ATM_PA) * self.fill_frac


@dataclass
class AirVesselSet:
    vessels: list[AirVessel] = field(default_factory=list)
    protection_pressure_pa: float = 0.0
    charge_into: str | None = None     # the vessel the compressor discharges into
    draw_from: str | None = None       # the vessel service consumers tap

    # ---------------------------------------------------------------
    @classmethod
    def from_graph(cls, graph: dict, circuit) -> "AirVesselSet | None":
        """Read the real vessels off the graph the machine was built
        from: any node on this circuit that declares a `capacity_kg`,
        which is exactly how `FluidCircuit.bottle_capacity_kg` was
        summed in the first place. Nothing here is hand-listed, so an
        engine with one tank gets one tank and a truck airpack gets its
        whole set."""
        if graph is None or circuit is None:
            return None
        ids = set(getattr(circuit, "nodes", ()) or ())
        by_id = {n["identity"]: n for n in graph.get("nodes", [])}
        found: list[AirVessel] = []
        for ident in sorted(ids):
            n = by_id.get(ident)
            if n is None:
                continue
            cap = float(n.get("capacity_kg", 0.0) or 0.0)
            if cap <= 0.0:
                continue
            kind = str(n.get("tank_kind", "") or "")
            label = {"wet-tank": "wet-tk", "reserve-tank": "resv"}.get(
                kind, ident.split(".")[-1][:6])
            if kind == "brake-reservoir":
                label = "brk-1" if "primary" in ident else "brk-2"
            found.append(AirVessel(
                identity=ident, label=label, capacity_kg=cap,
                working_pressure_pa=float(n.get("working_pressure_pa", 0.0) or 827_000.0),
                mass_kg=cap, protected=(kind == "brake-reservoir")))
        if len(found) < 2:
            return None

        # the real pipe topology, so "which vessel feeds which" is read
        # off the machine rather than assumed
        adj: dict[str, set[str]] = {}
        for ed in graph.get("edges", []):
            a, b = ed.get("a"), ed.get("b")
            if a in ids and b in ids:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        vessel_ids = {v.identity for v in found}

        def chain_from(entry: str) -> dict[str, str]:
            """Walk the real pipes OUTWARD FROM THE COMPRESSOR'S vessel
            and record, for each vessel, which vessel the air reached it
            through.

            It has to be done in this direction. Asking each vessel
            independently for its "nearest other vessel" gives a wrong
            answer whenever two vessels are equidistant -- the reserve
            tank and a brake reservoir are both one valve apart, and a
            plain nearest-neighbour search happily decided the reserve
            was fed BY a brake reservoir, which is backwards through the
            protection valve and would have made the whole protected-
            side rule meaningless."""
            parent: dict[str, str] = {}
            seen = {entry}
            layer = [entry]
            while layer:
                reached: list[str] = []
                for source in layer:
                    frontier = [source]
                    local = {source}
                    while frontier:
                        step_out = []
                        for node in frontier:
                            for peer in adj.get(node, ()):
                                if peer in local or peer in seen:
                                    continue
                                local.add(peer)
                                if peer in vessel_ids:
                                    parent[peer] = source
                                    seen.add(peer)
                                    reached.append(peer)
                                else:
                                    step_out.append(peer)
                        frontier = step_out
                layer = reached
            return parent

        order = sorted(found, key=lambda v: (not v.identity.endswith("wet_tank"),
                                             v.protected, v.identity))
        parent = chain_from(order[0].identity)
        for v in order[1:]:
            v.feeds_from = parent.get(v.identity)
        out = cls(vessels=order,
                  protection_pressure_pa=float(getattr(circuit, "protection_pressure_pa", 0.0) or 0.0),
                  charge_into=order[0].identity)
        # service air is drawn from the biggest UNPROTECTED vessel --
        # which is the whole reason the reserve tank is the big one
        service = [v for v in order if not v.protected]
        out.draw_from = (max(service, key=lambda v: v.capacity_kg).identity
                         if service else order[0].identity)
        return out

    # ---------------------------------------------------------------
    @property
    def total_mass_kg(self) -> float:
        return sum(v.mass_kg for v in self.vessels)

    @property
    def total_capacity_kg(self) -> float:
        return sum(v.capacity_kg for v in self.vessels)

    def by_id(self, identity: str | None) -> AirVessel | None:
        for v in self.vessels:
            if v.identity == identity:
                return v
        return None

    def step(self, dt: float, circuit_mass_kg: float) -> None:
        """Take the circuit's authoritative total and say where it is.

        A rise goes in at the compressor's vessel and a fall comes out
        of the one the service side taps, because that is physically
        where air really enters and leaves. Everything after that is the
        pipes equalising, gated by the protection valve."""
        if not self.vessels or dt <= 0.0:
            return
        delta = circuit_mass_kg - self.total_mass_kg
        if delta > 0.0:
            entry = self.by_id(self.charge_into) or self.vessels[0]
            entry.mass_kg += delta
        elif delta < 0.0:
            self._draw(-delta)

        # the pipes equalise: one explicit, rate-limited pass
        for v in self.vessels:
            src = self.by_id(v.feeds_from)
            if src is None:
                continue
            dp = src.pressure_pa - v.pressure_pa
            if v.protected and dp < 0.0 and src.pressure_pa < self.protection_pressure_pa:
                # THE PROTECTION VALVE. Air may go into the brake
                # reservoirs and may not come back out below its
                # setting: with a hole in the service side this is the
                # line between "the service air is gone" and "the brakes
                # are gone too".
                continue
            area = 3.14159 * (v.bore_m * 0.5) ** 2
            move = dp * CONDUCTANCE_KG_S_PER_PA_PER_M2 * area * dt
            # never move more than either end actually has
            move = min(move, src.mass_kg) if move > 0.0 else max(move, -v.mass_kg)
            src.mass_kg -= move
            v.mass_kg += move
        # any float drift goes back where the circuit's total says it is
        self._reconcile(circuit_mass_kg)

    def _draw(self, amount_kg: float) -> None:
        """Consumption comes out of the service side first, and only
        reaches the protected reservoirs once the service side is
        genuinely empty."""
        order = [v for v in self.vessels if not v.protected] + [v for v in self.vessels if v.protected]
        tap = self.by_id(self.draw_from)
        if tap is not None and tap in order:
            order.remove(tap)
            order.insert(0, tap)
        for v in order:
            if amount_kg <= 1e-15:
                break
            take = min(v.mass_kg, amount_kg)
            v.mass_kg -= take
            amount_kg -= take

    def _reconcile(self, circuit_mass_kg: float) -> None:
        err = circuit_mass_kg - self.total_mass_kg
        if abs(err) < 1e-12:
            return
        if err > 0.0:
            entry = self.by_id(self.charge_into) or self.vessels[0]
            entry.mass_kg += err
        else:
            self._draw(-err)

    def verify_conservation(self, circuit_mass_kg: float) -> float:
        """How far this split has drifted from the circuit it describes.
        It should be float noise; anything else is a bug in here."""
        return abs(self.total_mass_kg - circuit_mass_kg)

    def gauges(self) -> list[tuple[str, float, str]]:
        """(label, fill fraction, reading) per vessel, for the HUD."""
        return [(v.label, v.fill_frac, f"{v.pressure_pa / 1e5:4.1f}b") for v in self.vessels]
