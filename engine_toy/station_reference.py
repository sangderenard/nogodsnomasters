"""THE WHOLE STATION, ONE GRAPH, RUNNING.

    python station_reference.py            the spec, headless
    python station_reference.py --live     the window: fire it, watch it

WHAT IS IN IT. Not a drawing of a mechanism -- the mechanism, on the
machine, in the engine:

    the BOXED DRUM      `drum_reference.build`, the real one: two
                        annuli, cell walls, crossed shear diagonals,
                        six internal motors on their cradle, thrust
                        races, traction contacts and the inner lock-up
    the TWO ARCHES      standing on the drum's own top plate. Each foot
                        is BOLTED into the plate nodes nearest it, so
                        the load goes into the drum and the drum has to
                        answer for it -- an arch foot pinned to the
                        world would be a boundary condition and the
                        deck would never feel the shot
    the DANGLING PLATFORM, the GUN PLATFORM, the CRADLE and the GUN
                        `sled_reference.build` emitting into this same
                        graph rather than onto a stand-in ring

and then it is solved by the pieces the game already uses:

    live_scene.LiveStructure    FrameSolver + modal state, stepped at
                                the engine's own stability margin
    firing_frame_view.Trial     the baked mesh and the phong pass
    calibres.recoil_of          what firing actually puts back in --
                                projectile impulse plus gas impulse,
                                and the gas is a third of it

SPACE FIRES IT. The force that arrives is the one `calibres` computes
for the bore selected, applied at the breech along the bore, decaying
over the two milliseconds it really takes. It is not a number chosen to
make the picture move.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import drum_reference as dr
import sled_reference as sr
import stand


#: the drum's top plate ring -- what the arch feet bolt into
DECK_RING = "drum.upper.rim."
DECK_STATIONS = "drum.upper.station."

#: where the absorber pushes on the structure. Not the breech:
#: the breech is on a released freedom and just recoils.
RAIL_NODE = "mount.rail.aft"

#: THE TWO PRIME MOVERS, one to a bay. Their masses are read off the
#: catalogue rather than typed here: `engines.get(...).mass_kg` is what
#: the engine actually weighs, and a placeholder beside it would be a
#: second opinion about the same engine.
BAY_ENGINES = {"port": "ldt465-multifuel-deuce",
               "starboard": "agt1500-abrams-turbine"}
#: how far above the bay frame an engine's centre of mass sits
ENGINE_RISE_M = 0.520

#: below this a mode is a mechanism freedom, not an elastic one
ZERO_HZ = 0.05
#: how many elastic modes to keep ON TOP of the mechanism's freedoms
ELASTIC_MODES = 60
#: how deep to look when counting the freedoms
MODE_PROBE = 320


def build(fold: float = 0.0, **drum_overrides):
    """The drum, everything standing on it, and the stand under it."""
    g, box, drum, races, spec = dr.build(**drum_overrides)
    deck_y = float(spec["height_m"])
    deck_nodes = [n["identity"] for n in g.nodes
                  if n["identity"].startswith((DECK_RING, DECK_STATIONS))]
    _g, plat, d, u = sr.build(fold=fold, g=g, deck_y=deck_y,
                              deck_nodes=deck_nodes)

    # ---- THE SITE UNDER IT: work area, two engine bays, four legs ----
    # The drum's own static ring was pinned to the world, which is a
    # machine hanging in space. It stands on the stand, the stand stands
    # on the legs, and the legs stand on the ground -- so the only thing
    # still fixed to the world is a pad with dirt under it.
    import engines as engine_catalogue
    heaviest = max(float(engine_catalogue.get(i).mass_kg)
                   for i in BAY_ENGINES.values())
    st = stand.Stand(deck_y=0.0, engine_mass_kg=heaviest)
    site = stand.emit_stand(g, st)
    seats = [n for n in g.nodes if n["identity"].startswith("seat.")]
    pos = {n["identity"]: np.asarray(n["reference_position"], float)
           for n in g.nodes}
    frame = list(site["corners"].values())
    g.motion_group, g.assembly = "frame", "stand"
    for n in seats:
        n.pop("fixed_to", None)
        here = pos[n["identity"]]
        near = min(frame, key=lambda k: float(np.linalg.norm(
            pos[k][[0, 2]] - here[[0, 2]])))
        g.edge(f"stand.seat_post.{n['identity'].split('.')[-1]}",
               n["identity"], near, "rigid-distance", radius=0.070,
               alloy="4130n", palette="chassis-grey", beam_solvable=True,
               part_role="turret-post",
               load_path="the-turret-standing-on-the-work-area-frame")
    # ---- THE PRIME MOVERS, one in each bay ----
    # Mounted on the bay frame, not on the ground and not on the turret:
    # they ride up with the site when the legs stand it up, so nothing
    # they feed ever has to cross a gap that moves.
    g.motion_group, g.assembly = "frame", "powerplant"
    for side, identity in BAY_ENGINES.items():
        eng = engine_catalogue.get(identity)
        pts = site["bays"][side]
        centre = np.mean([pos[k] for k in pts.values()], axis=0)
        seat = centre + np.array([0.0, ENGINE_RISE_M, 0.0])
        node = f"engine.{side}"
        g.node(node, tuple(float(v) for v in seat),
               "load-bearing-structure", material="4130n",
               mass_in_total=False, mass_kg=float(eng.mass_kg),
               part_role="prime-mover", engine_identity=identity,
               engine_label=getattr(eng, "label", identity),
               displacement_l=float(getattr(eng, "displacement_l", 0.0)),
               bay=side, half_extent_m=(0.55, 0.45, 0.40))
        # FOUR MOUNTS, ONE PER BAY CORNER. An engine on fewer is an
        # engine that torques its own bay: the reaction to whatever it
        # drives comes out of the block, and it has to land somewhere.
        for tag, corner in pts.items():
            g.edge(f"engine.{side}.mount.{tag}", node, corner,
                   "engine-mount-isolator", radius=0.052, alloy="4130n",
                   palette="chassis-grey", beam_solvable=True,
                   part_role="engine-mount", bay=side,
                   load_path="the-engine-carried-by-its-own-bay-frame")
    return g, plat, box, spec, st, site


def spec_sheet(g, plat, box, spec, st=None) -> list:
    doc = g.as_document()
    from sled import describe
    import calibres
    recoiling = sum(float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
                    if x["identity"].startswith(("weapon.", "mount.", "gun.")))
    design_j = (calibres.recoil_of(calibres.cannon(120.0)).net_impulse_n_s
                ** 2) / (2.0 * recoiling)
    rot = sum(float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
              if not x["identity"].startswith("seat."))
    out = [f"THE STATION -- {len(doc['nodes'])} nodes,"
           f" {len(doc['edges'])} edges",
           f"  drum            {box.inner_radius_m:.3f} .."
           f" {box.outer_radius_m:.3f} m, {spec['height_m'] * 1000:.0f} mm"
           f" deep, deck at {spec['height_m']:.3f} m",
           f"  arch feet       bolted into {sr.PAD_BOLTS} plate nodes each,"
           f" 4 feet -- the load goes into the drum",
           f"  turning mass    {rot:.0f} kg",
           ""]
    out += describe(plat, design_j)
    return out


def paint(document: dict) -> dict:
    """Both palettes, one document: the drum's and the station's."""
    a = dr._paint(document)
    by = {n["identity"]: n for n in sr._paint(document)["nodes"]}
    be = {e["identity"]: e for e in sr._paint(document)["edges"]}
    keep = ("arch.", "dangling.", "gun.", "weapon.", "mount.", "station.")
    nodes = [by[n["identity"]] if n["identity"].startswith(keep) else n
             for n in a["nodes"]]
    edges = [be[e["identity"]] if e["identity"].startswith(keep) else e
             for e in a["edges"]]
    return dict(a, nodes=nodes, edges=edges)



# =====================================================================
#  FIRING IT THROUGH THE WHOLE CHAIN
# =====================================================================
def fire_through(g, plat, bores=(20.0, 120.0), gravity: bool = True) -> list:
    """A shot, with the force handed correctly between the sims.

    EACH SIM DOES ITS OWN JOB AND HANDS ON WHAT IT MADE. The mistake
    worth naming is putting the raw breech impulse straight onto the
    structure: that is fifteen meganewtons, it is what the mount would
    see if there were no recoil system at all, and a structure asked to
    carry it reports every member past yield. The recoil system exists
    precisely so the structure never sees it.

        calibres.recoil_of          the impulse -- projectile plus gas,
                                    and the gas is a third of it
        PlatformStage / the slide   what the absorber TRANSMITS out of
                                    that impulse. `AdaptiveRecoilDamper`
                                    inverts the energy balance for this
                                    round and reports the peak force it
                                    makes doing it
        graph_physics.solve_under_load
                                    that force as a real load: the
                                    frame solver turns it into
                                    displacements through the whole
                                    graph, and the game's own J2 member
                                    law turns the strains into stress,
                                    plastic flow and fracture demand

    The load is applied AT THE RAIL, because that is where the absorber
    pushes. Applying it at the breech would be loading the gun, which
    is on a released freedom and simply recoils.
    """
    import calibres
    import graph_physics as gp

    doc = g.as_document()
    recoiling = sum(float(x.get("mass_kg", 0.0)) for x in doc["nodes"]
                    if x["identity"].startswith(("weapon.", "mount.",
                                                 "gun.")))
    out = [f"FIRING THROUGH THE CHAIN -- {recoiling:.0f} kg recoiling",
           ""]
    for bore in bores:
        cal = calibres.cannon(bore)
        w = calibres.recoil_of(cal)
        imp = w.net_impulse_n_s
        sized = plat.slide_damper().orifice_for(imp, recoiling)
        transmitted = float(sized.get("peak_force_n", 0.0))
        out += [f"  {cal.name}   {cal.muzzle_energy_j / 1e6:.2f} MJ"
                f" at the muzzle"]
        out += ["  " + ln for ln in w.describe()]
        out += [f"    recoil velocity {imp / recoiling:7.2f} m/s"
                f"  ->  {0.5 * imp * imp / recoiling / 1000:8.1f} kJ"]
        if transmitted <= 0.0:
            out += [f"    the absorber's spring alone stops it --"
                    f" {sized.get('note', '')}", ""]
            continue
        out += [f"    absorber        aperture"
                f" {sized['orifice_mm']:5.1f} mm, transmits"
                f" {transmitted / 1000:7.0f} kN"
                f"   (raw: {w.peak_force_n() / 1000:.0f} kN)"]
        r = gp.solve_under_load(doc,
                                loads={RAIL_NODE: (0.0, 0.0, -transmitted)},
                                gravity=gravity)
        demand = np.abs(r["fracture_demand"])
        order = np.argsort(-demand)[:6]
        failed = int((r["failed"] > 0.5).sum())
        out += [f"    deflection      {r['max_deflection_m'] * 1000:7.2f} mm"
                f"   {failed} member(s) failed",
                "    worst members, by how close to fracture:"]
        for i in order:
            out.append(
                f"      demand {demand[i]:5.3f}   "
                f"axial {r['axial_stress_pa'][i] / 1e6:8.1f}   "
                f"bend {r['bending_stress_pa'][i] / 1e6:8.1f}   "
                f"shear {r['shear_stress_pa'][i] / 1e6:7.1f} MPa   "
                f"{r['identities'][i]}")
        out.append("")
    return out


# =====================================================================
#  THE WINDOW. The engine's own solver, stepped against the clock.
# =====================================================================
def live(g, plat, bore_mm: float = 120.0, W: int = 1440, H: int = 920):
    """The station, solved every tick, in a window.

    Every piece here is the game's own: `LiveStructure` is the frame
    solver plus modal state that `live_scene` runs, `Trial` is the
    baked mesh and the phong pass that `firing_frame_view` runs, and
    the force is whatever `calibres.recoil_of` says a shot of that
    bore puts back into the mount. Nothing in the loop is a number
    chosen to make the picture move."""
    import pygame
    import calibres
    import firing_frame_trial as fft
    import firing_frame_view as ffv
    from live_scene import LiveStructure
    from articulation import _flat

    doc = g.as_document()
    print("  assembling the structure ...", flush=True)
    t0 = time.perf_counter()
    # ---- HOW MANY MODES, AND WHY IT IS NOT A ROUND NUMBER ----------
    # This structure is a MECHANISM. The two parallelograms swing and
    # the gun slides, and those freedoms come out of the modal solve as
    # modes at zero frequency -- correctly, because a mechanism has
    # them. But `LiveStructure` retains the LOWEST modes, so asking for
    # forty gets forty mechanism freedoms and not one elastic mode: the
    # window would show the thing folding and never show it ringing.
    # So count the zero modes the structure actually has and retain
    # that many again plus the elastic ones we want.
    from frame_solver import FrameSolver
    probe = FrameSolver(document=doc, loads={}).modes(count=MODE_PROBE)
    hz = np.asarray(probe["all_frequencies_hz"], float)
    free = int((hz < ZERO_HZ).sum())
    want = free + ELASTIC_MODES
    print(f"  {free} mechanism freedoms at ~0 Hz -- the folding and the"
          f" slide. Retaining {want} modes to reach the elastic ones,"
          f" which start at {hz[free]:.2f} Hz", flush=True)
    st = LiveStructure(doc, modes=want)
    n60, h60 = st.substep_plan(1.0 / 60.0)
    print(f"  {len(st.members)} members, {len(st.omega)} modes,"
          f" up to {st.omega[-1] / (2 * math.pi):.1f} Hz"
          f"   ({time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"  stability margin {st.stability_margin} (the engine's own)"
          f" -> {n60} substeps of {h60 * 1000:.3f} ms in a 60 Hz frame",
          flush=True)

    # ---- WHAT FIRING PUTS IN, out of calibres and nothing else ------
    # These are IMPULSES in N.s -- `force_vector_n_s` is named for
    # what it is. The loop turns one into a force when it knows the
    # step it will be integrated over.
    shots = {}
    for bb in (20.0, 120.0):
        w = calibres.recoil_of(calibres.cannon(bb))
        shots[bb] = tuple(float(v) for v in w.force_vector_n_s)
        gas_pc = w.gas_impulse_n_s / (w.projectile_impulse_n_s
                                      + w.gas_impulse_n_s) * 100.0
        print(f"  {bb:5.0f} mm   {w.net_impulse_n_s:8.0f} N.s net"
              f"   ({gas_pc:.0f} % of it is gas)", flush=True)

    trial = ffv.Trial(doc, width=W, height=H, hidden=False)
    ranges = [trial.part_range.get("edge_" + _flat(i)) for i in st.identities]
    owner = np.concatenate([np.full(r[1] - r[0], i, np.int32)
                            for i, r in enumerate(ranges) if r])
    tris = np.concatenate([np.arange(r[0], r[1]) for r in ranges if r])
    bands = np.asarray([lim for lim, _c, _l in fft.UTILISATION_BANDS])
    band_mat = np.asarray([trial.band_ids[c]
                           for _l, c, _n in fft.UTILISATION_BANDS])
    base_mat = trial.rest_material_ids

    LOAD_NODE = "weapon.breech"
    if LOAD_NODE not in st.solver.index:
        raise KeyError(f"{LOAD_NODE} is not in the solve -- nothing to fire")

    clock = pygame.time.Clock()
    az, el, zoom = 0.62, 0.34, 1.0
    drag, running, frames = False, True, 0
    pending, bore, fired, peak = None, bore_mm, 0, 0.0
    print("  SPACE fire   B swap bore   R reset   G settle   "
          "drag orbit   wheel zoom   ESC quit", flush=True)
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    pending, fired = shots[bore], fired + 1
                elif ev.key == pygame.K_b:
                    bore = 20.0 if bore > 100.0 else 120.0
                    print(f"  bore now {bore:.0f} mm", flush=True)
                elif ev.key == pygame.K_g:
                    st.q[:] = st.shapes.T @ (st.mass * 0.02)
                    st.qd[:] = 0.0
                elif ev.key == pygame.K_r:
                    st.q[:] = 0.0
                    st.qd[:] = 0.0
                    st.clear()
                    peak = 0.0
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    drag = True
                elif ev.button in (4, 5):
                    zoom = max(0.25, min(4.0, zoom
                                         * (0.9 if ev.button == 4 else 1.1)))
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                drag = False
            elif ev.type == pygame.MOUSEMOTION and drag:
                az -= ev.rel[0] * 0.008
                el = max(-0.9, min(1.4, el + ev.rel[1] * 0.005))

        dt = min(clock.tick(60) / 1000.0, 0.05)
        # ---- DELIVER THE IMPULSE, NOT A FORCE FOR A WHILE ----------
        # The first version set a force on the keypress and then
        # cleared it before `step`, so the shot entered zero steps and
        # nothing moved: seven rounds fired and the peak stress never
        # left its static value. Reordering alone is not the fix
        # either. A shot is 2 ms and a frame here is 60, so holding the
        # firing force for a whole frame delivers thirty times the
        # impulse, and holding it for 2 ms of a 60 ms step delivers
        # none of it.
        #
        # What is actually true is that the impulse is INSTANTANEOUS on
        # this structure's terms: the fastest mode retained is 119 Hz,
        # an 8.4 ms period, so a 2 ms rise is not resolved by anything
        # in the model and pretending to resolve it is theatre. So the
        # whole impulse goes in over exactly one step, at F = I / dt.
        # F * dt is then the impulse `calibres` computed, whatever the
        # frame rate happens to be -- which is the only property that
        # matters and the one the old code did not have.
        if pending is not None:
            st.apply(LOAD_NODE, tuple(v / dt for v in pending))
            st.step(dt)
            st.clear()
            pending = None
        else:
            st.step(dt)

        util = st.utilisation()
        mat = base_mat.copy()
        if tris.size:
            band = np.searchsorted(bands, util)
            mat[tris] = band_mat[np.minimum(band[owner], len(band_mat) - 1)]
        trial.mesh.material_ids = mat
        trial.articulated.displace(st.positions())
        trial.view.restage_static_mesh()
        trial.view._elevation = el
        trial.view._zoom = zoom
        trial.present(az)
        pygame.display.flip()
        frames += 1
        peak = max(peak, float(util.max()))
        if frames % 60 == 0:
            hot = int((util > 0.85).sum())
            print(f"  {clock.get_fps():5.1f} fps   bore {bore:3.0f} mm"
                  f"   {fired} fired   now {util.max() * 100:6.2f}%"
                  f"   peak {peak * 100:6.2f}% of yield   {hot} hot",
                  flush=True)
    pygame.quit()


def main(argv) -> None:
    print("building the drum and everything on it ...", flush=True)
    g, plat, box, spec, st, site = build()
    print("\n".join(spec_sheet(g, plat, box, spec)), flush=True)
    problems = list(dict.fromkeys(g.check()))
    print(f"\n  check  {'PASS' if not problems else problems[:2]}", flush=True)
    print()
    print("\n".join(fire_through(g, plat)), flush=True)
    if "--live" in argv:
        live(g, plat)


if __name__ == "__main__":
    main(sys.argv[1:])
