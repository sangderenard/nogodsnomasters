import unittest

from demos.profiled_text_learning import run_profiled_demo


class ProfiledTextLearningDemoTests(unittest.TestCase):
    def test_demo_profiles_real_queue_and_nn_activity_without_false_native_claim(self):
        receipt = run_profiled_demo()
        self.assertNotEqual(receipt["before_token"], receipt["after_token"])
        self.assertEqual(receipt["vocabulary_epoch"], 2)
        self.assertEqual(receipt["embedding_epoch"], 2)
        self.assertGreaterEqual(receipt["optimizer_steps"], 6)
        self.assertIsNotNone(receipt["last_loss"])
        self.assertGreater(receipt["profile_calls"], 0)
        self.assertFalse(receipt["native_training"])
        self.assertIn("forward_vjp", receipt["native_blocker"])


if __name__ == "__main__":
    unittest.main()
