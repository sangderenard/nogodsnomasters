"""Run from engine_toy: python -m unittest discover -s tests -v."""
import math
import unittest

import numpy as np

import drum_reference
import structure_native as sn


def load_members(bank):
    values = dict(length_m=0.5, outer_radius_m=0.02, wall_m=0.003,
                  density_kg_m3=7850.0, youngs_cold_pa=2.05e11,
                  temperature_k=293.15, damping_ratio=0.005,
                  added_mass_per_m_kg=0.0, force_n=100.0)
    for name, value in values.items():
        bank.arrays[name][:] = value
    bank.arrays["length_m"][:] = np.linspace(0.2, 2.0, bank.n)
    return bank


class BeamBankTests(unittest.TestCase):
    def test_drum_frame_load_and_release_remain_bounded(self):
        graph, *_ = drum_reference.build()
        bank = sn.bank_for_target(graph.as_document(), target="python")
        self.assertEqual(bank.n, 232)
        driven = [i for i, ident in enumerate(bank.identities)
                  if ".track_hoop." in ident or ".rim_beam." in ident]
        initial = bank.step(0.0).copy()
        self.assertGreater(initial[:, 2].max(), 560.0)
        a = bank.arrays
        area = math.pi * (a["outer_radius_m"]**2
                          - (a["outer_radius_m"] - a["wall_m"])**2)
        mass = 0.25 * (area * a["density_kg_m3"]
                       + a["added_mass_per_m_kg"]) * a["length_m"]
        force = 14400.0 / len(driven)
        equilibrium = force / (mass[driven] * (2*math.pi*initial[driven, 2])**2)
        edges = {e["identity"]: e for e in graph.as_document()["edges"]}
        modulus = np.array([edges[i]["damage"]["youngs_modulus_pa"]
                            for i in bank.identities])
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for frame in range(600):
                # Default traction, clutch release, and reversed load;
                # variable frame times include the live window's 50 ms cap.
                command = (1.0, 0.0, -1.0, 0.0)[(frame // 150) % 4]
                a["force_n"][driven] = command * force
                dt = (1/60, 1/35, 0.05)[frame % 3]
                out = bank.step(dt, n_sub=8)
                stress = np.abs(out[:, 1]) * modulus * a["outer_radius_m"] / a["length_m"]
                self.assertTrue(np.isfinite(stress).all())
                self.assertTrue(np.isfinite(bank.qdot).all())
                self.assertTrue((np.abs(out[driven, 0]) < 4*equilibrium).all())
        self.assertLess(np.max(np.abs(bank.q[driven]) / equilibrium), 0.01)

    def test_vector_dispatch_matches_scalar_generated_source(self):
        bank = load_members(sn.PythonBank(8))
        bank.arrays["temperature_k"][:] = np.linspace(293.15, 1200.0, bank.n)
        bank.arrays["damping_ratio"][:] = np.linspace(0.0, 2.0, bank.n)
        bank.q[:] = np.linspace(-0.001, 0.001, bank.n)
        bank.qdot[:] = np.linspace(0.01, -0.02, bank.n)
        scope = {"max": max, "min": min, "abs": abs, "range": range}
        exec(compile(bank.source, "<scalar-beam-bank>", "exec"), scope)
        q, qdot, out = bank.q.copy(), bank.qdot.copy(), bank.out.copy()
        for dt in (0.0, 1/35, 0.05):
            count, h = bank.substep_plan(dt, 8)
            for _ in range(count):
                scope["beam_bank_run"](
                    bank.n, 1, h, *[bank.arrays[n] for n in sn.BANK_ARRAYS],
                    q, qdot, out)
            bank.step(dt, 8)
            np.testing.assert_allclose(bank.q, q, rtol=1e-11, atol=1e-14)
            np.testing.assert_allclose(bank.qdot, qdot, rtol=1e-11, atol=1e-14)
            np.testing.assert_allclose(bank.out, out, rtol=1e-11, atol=1e-14)

    def test_schedule_reacts_to_live_parameters_and_explicit_damping(self):
        bank = load_members(sn.PythonBank(3))
        dt = 0.05
        n_cold, h = bank.substep_plan(dt, 8)
        self.assertGreater(n_cold, 8)
        self.assertAlmostEqual(n_cold * h, dt)
        bank.arrays["temperature_k"][:] = 1200.0
        self.assertLess(bank.substep_plan(dt)[0], n_cold)
        bank.arrays["temperature_k"][:] = 293.15
        bank.arrays["added_mass_per_m_kg"][:] = 100.0
        self.assertLess(bank.substep_plan(dt)[0], n_cold)
        bank.arrays["added_mass_per_m_kg"][:] = 0.0
        bank.arrays["length_m"][:] *= 0.5
        self.assertGreater(bank.substep_plan(dt)[0], n_cold)
        bank.arrays["damping_ratio"][:] = 2.0
        count, h = bank.substep_plan(dt)
        omega = 2 * math.pi * bank.step(0.0)[:, 2]
        c = 2 * bank.arrays["damping_ratio"] * omega
        self.assertTrue((omega**2*h*h + 2*c*h < 4.0).all())
        self.assertEqual(bank.substep_plan(dt, count+7)[0], count+7)

    def test_zero_step_and_full_elapsed_time(self):
        bank = load_members(sn.PythonBank(2))
        bank.arrays["youngs_cold_pa"][:] = 0.0
        bank.arrays["force_n"][:] = 0.0
        bank.q[:] = (0.1, -0.1)
        bank.qdot[:] = (1.0, -2.0)
        before = bank.q.copy()
        bank.step(0.0)
        np.testing.assert_array_equal(bank.q, before)
        for dt in (1/60, 1/35, 0.05):
            bank.step(dt, n_sub=8)
        np.testing.assert_allclose(bank.q, before + bank.qdot*(1/60+1/35+0.05))

    def test_empty_bank_and_invalid_time(self):
        bank = sn.PythonBank(0)
        self.assertEqual(bank.step(0.05).shape, (0, 3))
        for dt in (-1.0, np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                bank.step(dt)


class NativeBeamBankTests(unittest.TestCase):
    def test_native_matches_python_with_adaptive_substeps(self):
        native = load_members(sn.native_bank()(4))
        reference = load_members(sn.PythonBank(4))
        self.assertEqual(native.source_digest, reference.source_digest)
        for bank in (native, reference):
            bank.q[:] = np.linspace(-0.001, 0.001, bank.n)
            bank.qdot[:] = np.linspace(0.01, -0.02, bank.n)
        for dt in (0.0, 1/35, 0.05):
            self.assertEqual(native.substep_plan(dt), reference.substep_plan(dt))
            actual = native.step(dt).copy()
            expected = reference.step(dt).copy()
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
            np.testing.assert_allclose(native.qdot, reference.qdot,
                                       rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
