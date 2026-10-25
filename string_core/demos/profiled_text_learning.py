"""Profile exact token modification and the live AbstractNN learner.

This is the honest Python reference demonstration. It exercises the genuine
writer and learner queues and reports the current native-training blocker; it
does not claim that eager Adam is native.
"""

from __future__ import annotations

import argparse
import cProfile
from pathlib import Path
import pstats
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from token_orchard.contract import (
    DOMAIN_TEXT,
    OBSERVATION_TRANSFORM,
    OBSERVATION_WEIGHT_ONE,
    STATUS_OK,
)
from token_orchard.embedding import ContinuousEmbeddingLearner
from token_orchard.semantic import SemanticObservationBatch
from token_orchard.service import TokenService


NATIVE_TRAINING_BLOCKER = (
    "function_address: no source cells for generated forward_vjp"
)


def run_profiled_demo():
    learner = ContinuousEmbeddingLearner(
        dimension=8,
        hidden_dimension=16,
        learning_rate=0.01,
        steps_per_delta=3,
    )
    service = TokenService(embedding_sink=learner)
    profiler = cProfile.Profile()
    try:
        # Exact Unicode scalars for two successive spellings. No string object
        # enters Token Orchard's batch ABI.
        before = [99, 111, 108, 111, 114]
        after = [99, 111, 108, 111, 117, 114]
        values = before + after
        profiler.enable()
        status, ticket = service.submit_intern(
            DOMAIN_TEXT,
            values,
            [0, len(before), len(values)],
            block=True,
        )
        if status != STATUS_OK or ticket is None:
            raise RuntimeError(f"intern submission failed: {status}")
        batch_status, statuses, tokens, epoch = ticket.wait(5.0)
        if batch_status != STATUS_OK or statuses != [STATUS_OK, STATUS_OK]:
            raise RuntimeError((batch_status, statuses))
        if not learner.wait_until_epoch(epoch, 5.0):
            raise RuntimeError("learner did not consume graph delta")

        observation = SemanticObservationBatch.from_columns(
            0,
            [OBSERVATION_TRANSFORM],
            [tokens[0]],
            [tokens[1]],
            [0],
            [4],
            [1],
            [0],
            [OBSERVATION_WEIGHT_ONE],
        )
        if not learner.offer_observations(observation):
            raise RuntimeError("learner refused transform observation")
        if not learner.wait_until_observation(1, 5.0):
            raise RuntimeError("learner did not consume transform observation")
        profiler.disable()
        snapshot = learner.snapshot()
        stats = pstats.Stats(profiler)
        return {
            "before_token": tokens[0],
            "after_token": tokens[1],
            "vocabulary_epoch": epoch,
            "embedding_epoch": snapshot.epoch,
            "optimizer_steps": snapshot.optimizer_steps,
            "last_loss": snapshot.last_loss,
            "profile_calls": stats.total_calls,
            "profile_seconds": stats.total_tt,
            "native_training": False,
            "native_blocker": NATIVE_TRAINING_BLOCKER,
        }
    finally:
        profiler.disable()
        service.close(wait=True)
        learner.close(wait=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    arguments = parser.parse_args(argv)
    
    # Force native execution; remove the eager run_profiled_demo() branch
    from demos.native_text_learning import run_native_demo
    receipt = run_native_demo(steps=arguments.steps)
    
    for name, value in receipt.items():
        print(f"{name}: {value}")

if __name__ == "__main__":
    main()
