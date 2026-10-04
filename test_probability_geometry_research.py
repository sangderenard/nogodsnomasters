"""Analytical fixtures for complete-vocabulary probability measurements."""

import math

import pytest

from probability_geometry_research import (
    entropy_from_log_probs,
    jensen_shannon_distance,
    jensen_shannon_divergence,
    raw_top_logit_margin,
)


from src.common.tensors.numpy_backend import NumPyTensorOperations

def tensor(rows):
    # NumPy is the repository's canonical backend. Do not import a model or
    # install Torch merely to exercise small deterministic numeric fixtures.
    return NumPyTensorOperations.tensor(rows)


def log_probs(rows):
    return tensor([[math.log(p) if p else -math.inf for p in row] for row in rows])


def test_entropy_uniform_and_zero_support():
    p = log_probs([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    h = entropy_from_log_probs(p)
    assert h.shape == (2,)
    assert h.tolist() == pytest.approx([math.log(2), 0])


def test_raw_margin_preserves_batch_and_is_shift_invariant():
    logits = tensor([[4.0, 1.0, 2.0], [-9.0, -9.0, -12.0]])
    before = logits.tolist()
    assert raw_top_logit_margin(logits).tolist() == pytest.approx([2, 0])
    assert raw_top_logit_margin(logits + 1000).tolist() == pytest.approx([2, 0])
    assert logits.tolist() == before


def test_raw_margin_is_not_tempered_probability_gap():
    logits = tensor([[3.0, 1.0, -2.0]])
    assert raw_top_logit_margin(logits).tolist() == pytest.approx([2])
    assert raw_top_logit_margin(logits / 2).tolist() == pytest.approx([1])


def test_js_self_zero_symmetry_and_hand_value():
    p = log_probs([[0.75, 0.25, 0.0]])
    q = log_probs([[0.25, 0.75, 0.0]])
    expected = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert jensen_shannon_divergence(p, p).tolist() == pytest.approx([0], abs=1e-7)
    assert jensen_shannon_divergence(p, q).tolist() == pytest.approx([expected])
    assert jensen_shannon_divergence(q, p).tolist() == pytest.approx([expected])
    assert entropy_from_log_probs(p).tolist() == pytest.approx(entropy_from_log_probs(q).tolist())
    assert jensen_shannon_distance(p, q).tolist() == pytest.approx([math.sqrt(expected)])


def test_disjoint_support_and_shared_zero_are_finite():
    p = log_probs([[1, 0, 0], [0, 1, 0]])
    q = log_probs([[0, 1, 0], [0, 1, 0]])
    before = p.tolist()
    result = jensen_shannon_divergence(p, q)
    assert result.shape == (2,)
    assert result.tolist() == pytest.approx([math.log(2), 0], abs=1e-7)
    assert jensen_shannon_distance(p, q).tolist() == pytest.approx([math.sqrt(math.log(2)), 0])
    assert p.tolist() == before


def test_extreme_log_tail_does_not_require_log_of_underflowed_probability():
    p = tensor([[0.0, -1000.0, -math.inf]])
    q = tensor([[-1000.0, 0.0, -math.inf]])
    assert entropy_from_log_probs(p).tolist() == pytest.approx([0])
    assert jensen_shannon_divergence(p, q).tolist() == pytest.approx([math.log(2)])


@pytest.mark.parametrize("rows", [
    [[-math.inf, -math.inf]], [[math.nan, 0]], [[math.inf, 0]],
    [[0.1, -1]], [[math.log(0.2), math.log(0.3)]], [[0, 0]],
])
def test_rejects_invalid_or_incomplete_distributions(rows):
    with pytest.raises(ValueError):
        entropy_from_log_probs(tensor(rows))
    with pytest.raises(ValueError):
        jensen_shannon_divergence(tensor(rows), log_probs([[0.5, 0.5]]))


@pytest.mark.parametrize("rows", [[], [[]], [0.0, 0.0], [[[0.0, 0.0]]]])
def test_rejects_wrong_shapes(rows):
    with pytest.raises(ValueError):
        entropy_from_log_probs(tensor(rows))
    with pytest.raises(ValueError):
        raw_top_logit_margin(tensor(rows))


def test_no_pair_broadcasting():
    with pytest.raises(ValueError, match="identical shapes"):
        jensen_shannon_divergence(log_probs([[1, 0]]), log_probs([[1, 0], [0, 1]]))


@pytest.mark.parametrize("rows", [[[1.0]], [[0.0, -math.inf]], [[math.nan, 0.0]]])
def test_raw_margin_rejects_missing_runner_up_and_nonfinite_logits(rows):
    with pytest.raises(ValueError):
        raw_top_logit_margin(tensor(rows))


@pytest.mark.parametrize("atol", [-1, 1, math.inf, math.nan])
def test_rejects_invalid_normalization_tolerance(atol):
    with pytest.raises(ValueError):
        entropy_from_log_probs(log_probs([[1, 0]]), normalization_atol=atol)


def test_requires_abstract_tensor():
    with pytest.raises(TypeError):
        entropy_from_log_probs([[0.0]])


@pytest.mark.parametrize("dtype, rows", [
    ("int8", [[127, -128]]), ("int64", [[3, 1]]),
    ("complex64", [[3 + 1j, 1 + 2j]]),
])
def test_raw_margin_rejects_integer_overflow_and_complex_ordering(dtype, rows):
    with pytest.raises(TypeError, match="real floating-point"):
        raw_top_logit_margin(NumPyTensorOperations.tensor(rows, dtype=dtype))


def test_real_turing_class_identity_and_import_paths():
    import inspect
    import sys
    from pathlib import Path
    import probability_geometry_research as research
    from src.common.tensors import AbstractTensor
    from src.common.tensors.abstraction import AbstractTensor as CanonicalTensor
    from src.common.tensors.numpy_backend import AbstractTensor as BackendTensor

    turing_root = Path(__file__).resolve().parent / "turing"
    assert research.AbstractTensor is CanonicalTensor is AbstractTensor is BackendTensor
    assert Path(inspect.getfile(CanonicalTensor)).resolve() == (
        turing_root / "src/common/tensors/abstraction.py"
    ).resolve()
    assert Path(inspect.getfile(NumPyTensorOperations)).resolve() == (
        turing_root / "src/common/tensors/numpy_backend.py"
    ).resolve()
    assert type(tensor([[0.0]])) is NumPyTensorOperations
    assert "tensors" not in sys.modules
    print("AbstractTensor:", inspect.getfile(CanonicalTensor))
    print("NumPy backend:", inspect.getfile(NumPyTensorOperations))
    print("Canonical class identity:", research.AbstractTensor is BackendTensor)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_zero_support_and_extreme_tail_without_invalid_arithmetic(dtype):
    import numpy as np

    p = NumPyTensorOperations.tensor([[0.0, -1000.0, -math.inf]], dtype=dtype)
    q = NumPyTensorOperations.tensor([[-1000.0, 0.0, -math.inf]], dtype=dtype)
    # exp(-1000) necessarily underflows in these formats; NaN/overflow and
    # log(0) are not permitted. No runtime numerical warning is filtered.
    with np.errstate(invalid="raise", divide="raise", over="raise"):
        assert entropy_from_log_probs(p).tolist() == [0.0]
        assert jensen_shannon_divergence(p, q).tolist() == pytest.approx([math.log(2)])
        assert jensen_shannon_distance(p, q).tolist() == pytest.approx([math.sqrt(math.log(2))])
