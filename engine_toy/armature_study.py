"""Watch an armature move: a strip of frames through one motion.

    python armature_study.py hexapod --out hexapod_study.png
    python armature_study.py turret  --out turret_study.png

Made for studying rather than for playing. A live view shows you the
pose it is in now; a strip shows you the whole motion at once, which is
what you need to see HOW a linkage works -- which members change length,
which stay put, and where the geometry starts fighting you.

Every frame is a real solve. Nothing is interpolated between poses: the
activations are set, the structure is relaxed onto them, and that pose
is drawn.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))

TILE = 420


def _armature_graph(arm) -> dict:
    """The armature as a graph the renderer already understands."""
    nodes = []
    for n in arm.nodes:
        nodes.append({
            "identity": n.identity, "kind": "chassis-load-node",
            "reference_position": [float(v) for v in n.position],
            "body_half_extent_m": [0.030, 0.030, 0.030],
            "material": "steel-plate" if not n.anchored else "cast-iron",
            "mass_kg": n.mass_kg, "in_view": True,
        })
    edges = []
    for m in arm.members:
        edges.append({
            "identity": m.identity, "a": m.a, "b": m.b,
            "constraint": "linear-hydraulic-actuator" if m.actuated else "rigid-distance",
            "radius": 0.022 if m.actuated else 0.012,
            "material": "hardened-steel" if m.actuated else "steel-plate",
            "in_view": True,
        })
    return {"schema": "engine-toy-drivetrain-graph-v1", "identity": "armature",
            "machine": True, "nodes": nodes, "edges": edges}


def _hexapod():
    from armature import Armature, ArmatureNode, ArmatureMember, ElectricLinearActuator
    R, r, h = 0.32, 0.20, 0.36
    arm = Armature(identity="hexapod", end_effector="plat.0", effector_tip="plat.tip")
    for k in range(3):
        a = math.radians(90 + 120 * k)
        for sd in (-1, 1):
            ang = a + sd * math.radians(22)
            arm.nodes.append(ArmatureNode(
                f"base.{k}{'L' if sd < 0 else 'R'}",
                np.array([R * math.cos(ang), 0.0, R * math.sin(ang)]), anchored=True))
        arm.nodes.append(ArmatureNode(f"plat.{k}",
                                      np.array([r * math.cos(a), h, r * math.sin(a)]), mass_kg=4.0))
    arm.nodes.append(ArmatureNode("plat.tip", np.array([0.0, h + 0.50, 0.0]), mass_kg=2.0))
    pos = {n.identity: np.array(n.position, float) for n in arm.nodes}
    for k in range(3):
        j = (k + 1) % 3
        arm.members.append(ArmatureMember(
            f"rail.{k}", f"plat.{k}", f"plat.{j}",
            float(np.linalg.norm(pos[f"plat.{k}"] - pos[f"plat.{j}"]))))
        arm.members.append(ArmatureMember(
            f"tie.{k}", f"plat.{k}", "plat.tip",
            float(np.linalg.norm(pos[f"plat.{k}"] - pos["plat.tip"]))))
    for k in range(3):
        for sd in ("L", "R"):
            b = f"base.{k}{sd}"
            L = float(np.linalg.norm(pos[f"plat.{k}"] - pos[b]))
            arm.members.append(ArmatureMember(
                f"leg.{k}{sd}", b, f"plat.{k}", L,
                actuator=ElectricLinearActuator(identity=f"ram.{k}{sd}", screw_kind="acme"),
                minimum_m=L - 0.10, maximum_m=L + 0.10, activation_m=L))
    return arm


def study(kind: str, out: str, columns: int = 6) -> int:
    pygame.init()
    pygame.display.set_mode((64, 64), pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN)
    from engine_gl_view import EngineGLView
    from armature import PoseSolver

    arm = _hexapod()
    arm.solve_pose()
    # a sweep that shows the geometry working: around a cone, so every
    # leg takes its turn at full extension and full retraction
    goals = []
    for i in range(columns):
        a = 2.0 * math.pi * i / columns
        goals.append((math.sin(a) * 0.42, 1.0, math.cos(a) * 0.42))

    view = EngineGLView(width=TILE, height=TILE, covers_off=False,
                        animation_divisions=[0.0], detail=0.6)
    strip = pygame.Surface((TILE * columns, TILE + 26), pygame.SRCALPHA)
    strip.fill((16, 17, 20, 255))
    font = pygame.font.SysFont("consolas", 15)

    for i, goal in enumerate(goals):
        out_info = PoseSolver(armature=arm).solve("direction", goal)
        graph = _armature_graph(arm)
        view.set_graph(graph)
        while view.bake_next():
            pass
        view._angle_rad = 0.75
        view._elevation = 0.42
        view._zoom = 1.0
        rgba = view.render(crank_angle_deg=0.0, spin=False, dt=0.0, throttle_frac=0.0)
        frame = pygame.image.frombuffer(
            np.ascontiguousarray(rgba[:, :, :3]).tobytes(),
            (rgba.shape[1], rgba.shape[0]), "RGB")
        strip.blit(frame, (i * TILE, 26))
        legs = sorted(out_info["activations"].items())
        spread = max(v for _, v in legs) - min(v for _, v in legs)
        label = (f"aim {goal[0]:+.2f},{goal[2]:+.2f}   err {out_info['error']:.0e}   "
                 f"leg spread {spread * 1000:.0f} mm")
        strip.blit(font.render(label, True, (210, 216, 224)), (i * TILE + 10, 5))
    pygame.image.save(strip, out)
    print(f"{columns} poses -> {out}")
    pygame.quit()
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", nargs="?", default="hexapod")
    ap.add_argument("--out", default="armature_study.png")
    ap.add_argument("--columns", type=int, default=6)
    a = ap.parse_args()
    raise SystemExit(study(a.kind, a.out, a.columns))
