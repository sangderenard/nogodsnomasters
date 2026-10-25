"""Compile and execute one Token Orchard embedding-learning motion."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TURING_ROOT = PROJECT_ROOT.parent / "turing"
for root in (PROJECT_ROOT, TURING_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from token_orchard.contract import DOMAIN_TEXT, STATUS_OK
from token_orchard.semantic_tensor import embedding_position_batch
from token_orchard.service import TokenService


def _intern_demo_tokens():
    before = [99, 111, 108, 111, 114]
    after = [99, 111, 108, 111, 117, 114]
    service = TokenService()
    try:
        status, ticket = service.submit_intern(
            DOMAIN_TEXT,
            before + after,
            [0, len(before), len(before) + len(after)],
            block=True,
        )
        if status != STATUS_OK or ticket is None:
            raise RuntimeError(f"intern submission failed: {status}")
        batch_status, statuses, tokens, epoch = ticket.wait(5.0)
        if batch_status != STATUS_OK or statuses != [STATUS_OK, STATUS_OK]:
            raise RuntimeError((batch_status, statuses))
        return tokens, epoch
    finally:
        service.close(wait=True)


def run_native_demo(build_directory=None, *, steps=12):
    from src.common.tensors import AbstractTensor
    from src.common.tensors.abstract_nn import Identity, Linear, Model, Tanh
    from src.common.tensors.accelerator_backends.ssa_backend import (
        SSATensorOperations,
        SSATensorProgram,
    )
    from src.compiler.process_graph_autograd import (
        lower_training_motion_to_repository_ssa,
        obtain_graph_reverse,
    )
    from src.compiler.ssa_llvm_backend import (
        compile_artifact,
        emit_ssa_function_to_llvm,
        prepare_artifact_execution,
        with_native_adam_loop,
    )

    dimension = 8
    hidden_dimension = 16
    tokens, vocabulary_epoch = _intern_demo_tokens()
    with AbstractTensor.use_backend("numpy", "cpu"):
        positions = np.asarray(embedding_position_batch(
            AbstractTensor.tensor(tokens, dtype="int64"), dimension,
        ).data)
    source = positions[0:1]
    target = positions[1:2] - source

    program = SSATensorProgram("token_orchard_native_text_learning")
    shapes = (
        (1, dimension),
        (dimension, hidden_dimension),
        (1, hidden_dimension),
        (hidden_dimension, dimension),
        (1, dimension),
        (1, dimension),
    )
    x, w1, b1, w2, b2, target_tensor = [
        SSATensorOperations.input(program, shape) for shape in shapes
    ]
    first = Linear(dimension, hidden_dimension, like=x, init="xavier")
    first.W, first.b = w1, b1
    second = Linear(hidden_dimension, dimension, like=x, init="xavier")
    second.W, second.b = w2, b2
    prediction = Model(
        [first, second], [Tanh(), Identity()],
    ).forward(x)
    error = prediction - target_tensor
    loss = (error * error).sum()
    bindings = {
        "position": x,
        "first_weight": w1,
        "first_bias": b1,
        "second_weight": w2,
        "second_bias": b2,
        "target_residual": target_tensor,
    }
    value_ids = {
        name: int(value.data.value.id) for name, value in bindings.items()
    }
    parameter_ids = tuple(value_ids[name] for name in (
        "first_weight", "first_bias", "second_weight", "second_bias",
    ))
    product = obtain_graph_reverse(
        loss,
        bindings=bindings,
        wrt=parameter_ids,
        packaging="combined",
        unit_output_seed=True,
    )
    lowering = lower_training_motion_to_repository_ssa(
        product.motion,
        function_name="token_orchard_native_text_learning",
        observed_outputs={"prediction": int(prediction.data.value.id)},
    )
    if lowering.shortfalls:
        raise RuntimeError(f"repository SSA shortfalls: {lowering.shortfalls!r}")
    emitted = emit_ssa_function_to_llvm(
        lowering.module,
        lowering.function_name,
        entry_name=lowering.function_name,
    )
    if emitted.shortfalls:
        raise RuntimeError(f"LLVM shortfalls: {emitted.shortfalls!r}")
    wrapped = with_native_adam_loop(
        emitted,
        parameter_gradient_pairs=tuple(
            (parameter_id, product.motion.gradient_value_ids[parameter_id])
            for parameter_id in parameter_ids
        ),
        entry_name="token_orchard_native_text_learning_adam",
    )
    build_directory = Path(
        build_directory or PROJECT_ROOT / "build" / "native_text_learning"
    )
    artifact = compile_artifact(wrapped, directory=build_directory)

    rng = np.random.default_rng(7)
    scale = 0.15
    parameters = {
        value_ids["first_weight"]: rng.normal(0.0, scale, shapes[1]),
        value_ids["first_bias"]: np.zeros(shapes[2]),
        value_ids["second_weight"]: rng.normal(0.0, scale, shapes[3]),
        value_ids["second_bias"]: np.zeros(shapes[4]),
    }
    initial_prediction = (
        np.tanh(
            source @ parameters[value_ids["first_weight"]]
            + parameters[value_ids["first_bias"]]
        ) @ parameters[value_ids["second_weight"]]
        + parameters[value_ids["second_bias"]]
    )
    initial_loss = float(np.sum((initial_prediction - target) ** 2))
    state = artifact.optimizer_state_value_ids
    buffers = {
        value_ids["position"]: source,
        value_ids["target_residual"]: target,
        **{value_id: value.copy() for value_id, value in parameters.items()},
        state["steps"]: np.asarray(int(steps), dtype=np.int32),
        state["learning_rate"]: np.asarray(0.03),
        state["beta1"]: np.asarray(0.9),
        state["beta2"]: np.asarray(0.999),
        state["epsilon"]: np.asarray(1e-8),
        state["beta1_power"]: np.asarray(1.0),
        state["beta2_power"]: np.asarray(1.0),
        state["iteration"]: np.asarray(0, dtype=np.int32),
    }
    for parameter_id in parameter_ids:
        buffers[state["first_moment"][parameter_id]] = np.zeros_like(
            parameters[parameter_id]
        )
        buffers[state["second_moment"][parameter_id]] = np.zeros_like(
            parameters[parameter_id]
        )
    execution = prepare_artifact_execution(artifact, buffers).run()
    return {
        "before_token": tokens[0],
        "after_token": tokens[1],
        "vocabulary_epoch": vocabulary_epoch,
        "native_artifact": str(artifact.library_path),
        "optimizer": "adam",
        "optimizer_steps": int(execution.buffers[state["iteration"]]),
        "initial_loss": initial_loss,
        "final_loss": float(execution.buffers[lowering.outputs["loss_0"]]),
    }
