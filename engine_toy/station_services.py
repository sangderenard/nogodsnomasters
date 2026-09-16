"""What the station SELLS and BUYS, and what trading it costs.

A plant that separates air, cleans oil and freezes water is not only a
plant. Every one of those is somebody else's problem solved, and the kit
is already bought and already running -- the marginal cost of a service
is the fraction of a shift somebody spends filling a cylinder.

THE ECONOMICS THAT ARE ACTUALLY INTERESTING. Most of these are not sales
at all, they are DISPOSAL, and disposal is paid in the other direction.
Fouled oil is a liability to whoever has it: it has to go somewhere, and
somewhere legal costs money. A site with a centrifuge and a vacuum
dehydrator can take that liability, charge for taking it, clean it, and
sell it back. Being paid at both ends of the same litre is the best
business on this list, and the equipment for it was bought for the
station's own engines.

SCRAP WORKS THE SAME WAY AND IS NOT REALLY A BUSINESS AT ALL. Metal
bought by the kilogram is bought as FEEDSTOCK, not as stock in trade. A
station that can melt has no other source of bronze for a bushing, of
lead for a plate, of aluminium for a casting -- and the alternative to
buying scrap is a supply run, which costs more than the metal in every
currency including attention.

AND THE COST THAT IS NOT MONEY. Every trade brings somebody to the
position. That is the thing being traded and it is not on the invoice:
a station that deals with everyone is a station everyone can describe.
Each offer carries an EXPOSURE -- how much knowing about the place a
customer leaves with -- and the strategic question is never whether a
service is profitable but whether the profit is worth being known.

WHICH IS WHY FUEL IS THE MISTAKE. Fuel is the one thing here with no
by-product logic behind it: it is not something the station makes as a
consequence of doing something else, it is the station's own stock, sold
at the cost of its own endurance. It brings VEHICLES rather than people,
vehicles are logged and followed, and a fuel point is a thing worth
taking rather than a thing worth visiting. The offer exists so the
player can make the mistake, not because it is not one.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ServiceOffer:
    key: str
    label: str
    unit: str
    #: Positive: the customer pays the station. Negative: the station
    #: pays, which is what buying feedstock looks like.
    price_per_unit: float
    #: What has to exist and be working for this to be offerable.
    requires: tuple
    #: Exposure per VISIT, not per unit -- one customer buying ten
    #: cylinders knows exactly as much as one buying one.
    exposure_per_visit: float
    depletes: str = ""
    strategic: str = ""
    why: str = ""


SERVICES: dict[str, ServiceOffer] = {
    "fouled-fluid-intake": ServiceOffer(
        "fouled-fluid-intake", "take fouled fluid off their hands", "litre",
        price_per_unit=0.35, requires=("centrifuge", "sludge-receiver"),
        exposure_per_visit=0.8, depletes="sludge receiver capacity",
        strategic="the best offer on the list: paid to receive, paid again to return",
        why="used oil is a liability to whoever is holding it. Taking it is a service "
            "that is paid for, and it arrives as free feedstock for kit the station "
            "already runs for its own engines"),
    "clean-fluid-sale": ServiceOffer(
        "clean-fluid-sale", "sell reclaimed fluid", "litre",
        price_per_unit=1.10, requires=("centrifuge", "vacuum-dehydrator"),
        exposure_per_visit=0.8, depletes="filter elements, desiccant",
        strategic="sells the same litre that was paid for on the way in",
        why="a litre through a centrifuge and a vacuum dehydrator is not second-hand "
            "oil, it is oil -- water out, particulate out, and an analysis to prove it"),
    "fluid-analysis": ServiceOffer(
        "fluid-analysis", "oil analysis", "sample",
        price_per_unit=12.0, requires=("oil-analysis",),
        exposure_per_visit=0.3,
        strategic="the quietest offer here: a sample arrives, a number leaves",
        why="the station already runs spectrographic analysis on its own sumps; doing "
            "it for a stranger costs a vial and twenty minutes, and the customer never "
            "needs to see anything but a hand at a gate"),
    "dry-ice": ServiceOffer(
        "dry-ice", "dry ice", "kg",
        price_per_unit=2.40, requires=("front-end-purifier",),
        exposure_per_visit=0.6, depletes="sieve regeneration cycles",
        strategic="pure by-product: it is the contents of a filter that had to be "
                  "emptied anyway",
        why="what the molecular sieve caught on its way to protecting the cold box. "
            "Refrigeration that needs no vessel and no power, which is why it sells"),
    "oxygen-fill": ServiceOffer(
        "oxygen-fill", "oxygen cylinder fill", "m3",
        price_per_unit=9.00, requires=("air-separation", "cryo-boiler-coil"),
        exposure_per_visit=1.4, depletes="LOX stock",
        strategic="the highest-exposure sale here: medical and welding oxygen is "
                  "scarce and memorable, and people remember where they got it",
        why="the coil that freezes the ice bank vaporises by-product oxygen on its way "
            "to being sold, so the station is paid for gas it used as refrigeration "
            "first. The same kilogram does two jobs"),
    "nitrogen-fill": ServiceOffer(
        "nitrogen-fill", "nitrogen cylinder fill", "m3",
        price_per_unit=4.50, requires=("air-separation",),
        exposure_per_visit=0.9, depletes="LIN stock",
        strategic="lower value and lower exposure than oxygen; tyre fill and purging "
                  "are boring errands, and boring errands are not repeated as stories",
        why="the column makes far more nitrogen than the station can use, and the "
            "alternative to selling it is venting it"),
    "ice": ServiceOffer(
        "ice", "block ice", "kg",
        price_per_unit=0.20, requires=("ice-pit",),
        exposure_per_visit=0.5, depletes="the cold bank itself",
        strategic="cheap, heavy, and it sells the station's own thermal reserve. The "
                  "bank is what keeps the position dark on infrared; selling it is "
                  "selling concealment by the kilogram",
        why="the pit holds tonnes and ice is worth real money in a desert, which is "
            "exactly the temptation"),
    "refuelling": ServiceOffer(
        "refuelling", "refuelling", "litre",
        price_per_unit=1.60, requires=("fuel-storage",),
        exposure_per_visit=3.5, depletes="the station's own endurance",
        strategic="A MISTAKE, offered so it can be made. Everything else here is a "
                  "by-product of running the station; fuel is the station's own stock "
                  "sold at the price of its own range. It brings VEHICLES, which are "
                  "logged, followed and remembered, and it turns a place worth "
                  "visiting into a place worth taking",
        why="the margin is real and the exposure is four times anything else on the "
            "list. Selling fuel is how a quiet site stops being one"),
}


def service(key: str) -> ServiceOffer:
    s = SERVICES.get(str(key))
    if s is None:
        raise KeyError(f"unknown service {key!r}; declared: {', '.join(sorted(SERVICES))}")
    return s


# ---------------------------------------------------------------------
# SCRAP: bought by the kilogram, wanted as feedstock
# ---------------------------------------------------------------------
#
# These carry a RECOVERABLE FRACTION as well as a price, because the
# object is mostly not the metal. A starter motor is copper wrapped
# around a lot of steel; a battery is lead in a plastic box full of acid;
# a catalytic converter is a ceramic brick with a few grams of something
# absurd washed onto it.
#
# THE MELTING POINT IS THE WHOLE STORY OF WHAT A STATION CAN DO WITH IT.
# Lead runs out of a wood fire. Aluminium wants a real furnace but a
# small one. Copper and brass want a proper crucible and a forced draught
# and will eat the crucible while they do it. Which metals a site can
# process is set by the hottest thing it owns, and that is a better
# constraint than an inventory slot.

@dataclass(frozen=True)
class ScrapClass:
    key: str
    label: str
    price_per_kg: float           # paid to the seller, per kg of object
    recoverable_frac: float       # what of the object is the wanted metal
    melt_k: float                 # what it takes to do anything with it
    hazard: str = ""
    feeds: str = ""
    why: str = ""

    @property
    def price_per_recovered_kg(self) -> float:
        """What the metal actually costs once the rest is thrown away.

        The number to compare against buying the metal, and it is much
        worse than the sticker price for anything that is mostly not
        metal."""
        return self.price_per_kg / max(1e-6, self.recoverable_frac)


SCRAP: dict[str, ScrapClass] = {
    "copper": ScrapClass(
        "copper", "clean copper", 6.80, 0.97, 1357.8,
        feeds="windings, bearing shells, brazing, anything electrical",
        why="the most valuable common scrap and the one a station most needs itself. "
            "Every rewound starter, every alternator, every metre of cable is copper "
            "that does not have to be asked for"),
    "brass": ScrapClass(
        "brass", "yellow brass", 4.20, 0.95, 1203.0,
        feeds="bushings, fittings, valve seats",
        why="copper and zinc already alloyed into roughly what a bushing wants. It "
            "melts lower than copper and machines better, so a site that can only "
            "just melt should melt this"),
    "aluminium": ScrapClass(
        "aluminium", "aluminium", 1.30, 0.92, 933.5,
        feeds="castings, housings, heat sinks, patch plate",
        why="cheap per kilogram and the easiest real melt on the list -- a small "
            "furnace does it. Recycling it costs a twentieth of smelting it new, "
            "which is why aluminium scrap is never actually thrown away anywhere"),
    "lead": ScrapClass(
        "lead", "lead", 1.60, 0.98, 600.6,
        hazard="toxic: fume and dust, cumulative, and it does not leave the body",
        feeds="battery plates, ballast, radiation shielding, babbitt",
        why="melts at six hundred kelvin, which a wood fire reaches -- the one metal "
            "here that needs no furnace at all. That accessibility is exactly why "
            "lead poisoning has such a long occupational history"),
    "lead-acid-battery": ScrapClass(
        "lead-acid-battery", "scrap lead-acid battery", 0.85, 0.62, 600.6,
        hazard="sulfuric acid, and the case is full of it",
        feeds="lead stock, and a case that may be rebuildable",
        why="about sixty per cent lead by mass, and the most recycled product in the "
            "world by a distance -- the loop closes above ninety-five per cent because "
            "the economics work without anybody being virtuous about it. Worth taking "
            "in for the plates alone, and the acid is a reagent rather than a waste"),
    "catalytic-converter": ScrapClass(
        "catalytic-converter", "catalytic converter", 140.0, 0.002, 2041.0,
        hazard="worth stealing, which makes holding a pile of them a reason to be "
               "visited by people who are not customers",
        feeds="platinum-group catalyst, if there is chemistry to use it on",
        why="a ceramic honeycomb with two to seven GRAMS of platinum, palladium and "
            "rhodium washcoated onto it -- which is why the price is per unit rather "
            "than per kilogram, and why they get cut off vehicles in car parks. "
            "Recovering the metal needs chemistry a field station does not have, so in "
            "practice these are held and traded onward rather than processed"),
    "steel": ScrapClass(
        "steel", "mild steel", 0.18, 0.98, 1723.0,
        feeds="structure, plate, berm revetment, anything welded",
        why="worth almost nothing by weight and worth a great deal by being present. "
            "A station does not melt it, it CUTS and WELDS it, which needs no furnace "
            "-- the only scrap here that is useful without a crucible"),
}


def scrap_class(key: str) -> ScrapClass:
    s = SCRAP.get(str(key))
    if s is None:
        raise KeyError(f"unknown scrap {key!r}; declared: {', '.join(sorted(SCRAP))}")
    return s


def meltable_with(peak_k: float) -> tuple:
    """What a site can actually process, given the hottest thing it has.

    Steel is excluded deliberately: it is worked by cutting and welding
    rather than by melting, and a station that waits until it can melt
    steel before using steel has misunderstood the material."""
    return tuple(sorted(k for k, s in SCRAP.items()
                        if s.melt_k <= float(peak_k) and k != "steel"))


@dataclass
class Transaction:
    key: str
    quantity: float
    revenue: float          # negative when the station is buying
    exposure: float
    customer: str = "passing trade"


@dataclass
class ServiceLedger:
    """What the station has traded and who knows about it.

    Two running totals meant to be read together, because the second is
    the price of the first and it is not refundable."""
    identity: str = "station.services"
    log: list = field(default_factory=list)
    revenue: float = 0.0
    exposure: float = 0.0
    scrap_kg: dict = field(default_factory=dict)

    def offerable(self, key: str, capabilities) -> bool:
        """Whether the kit for it exists and is working.

        Capabilities are declared identities, never matched by name -- a
        station offers oxygen because it has a column and a coil, not
        because somebody wrote 'oxygen' somewhere."""
        have = set(capabilities)
        return all(r in have for r in service(key).requires)

    def missing(self, key: str, capabilities) -> tuple:
        have = set(capabilities)
        return tuple(r for r in service(key).requires if r not in have)

    def sell(self, key: str, quantity: float, customer: str = "passing trade") -> Transaction:
        s = service(key)
        qty = max(0.0, float(quantity))
        t = Transaction(s.key, qty, qty * s.price_per_unit, s.exposure_per_visit, customer)
        self.revenue += t.revenue
        self.exposure += t.exposure
        self.log.append(t)
        return t

    def buy_scrap(self, key: str, quantity: float,
                  customer: str = "passing trade") -> Transaction:
        """Money out, metal in. The revenue is negative and that is the
        point -- this is the only line here that is an investment."""
        s = scrap_class(key)
        qty = max(0.0, float(quantity))
        t = Transaction(s.key, qty, -qty * s.price_per_kg, 0.7, customer)
        self.revenue += t.revenue
        self.exposure += t.exposure
        self.scrap_kg[s.key] = self.scrap_kg.get(s.key, 0.0) + qty
        self.log.append(t)
        return t

    def recovered_kg(self, key: str) -> float:
        """Metal actually in hand, not weight of object taken in."""
        return self.scrap_kg.get(key, 0.0) * scrap_class(key).recoverable_frac

    @property
    def revenue_per_face(self) -> float:
        """The only ratio that matters. A high number is a station paid
        well for being known; a low one is a station giving itself away."""
        return 0.0 if self.exposure <= 0.0 else self.revenue / self.exposure

    def menu(self, capabilities) -> list:
        """What can be offered right now, best return per face first."""
        rows = []
        for k, s in SERVICES.items():
            ok = self.offerable(k, capabilities)
            rows.append({"key": k, "label": s.label, "unit": s.unit,
                         "price": s.price_per_unit, "exposure": s.exposure_per_visit,
                         "per_face": s.price_per_unit / max(0.01, s.exposure_per_visit),
                         "offerable": ok, "missing": self.missing(k, capabilities),
                         "strategic": s.strategic})
        rows.sort(key=lambda r: (not r["offerable"], -r["per_face"]))
        return rows


def reclaim_margin_per_litre() -> float:
    """Paid to take it and paid to return it.

    The number that explains why reclamation is the business and the
    cylinders are the sideline."""
    return (service("fouled-fluid-intake").price_per_unit
            + service("clean-fluid-sale").price_per_unit)
