#!/usr/bin/env python3
"""Full-vocabulary probability measurements, independent of graph topology.

Inputs are ``[batch, vocabulary]`` AbstractTensors; results have shape
``[batch]``. Log-probabilities must already be normalized over the complete,
identically ordered vocabulary, using natural logs. No top-k renormalization,
tail bin, policy transform, or backward candidate score is substituted here.
The caller owns context/model/vocabulary provenance and observation retention.

These functions do not attach observations to nodes or define similarity edges.
They perform no backend conversion, model evaluation, or input mutation.
"""
from __future__ import annotations

import math

import sys
from pathlib import Path

# Match the existing root workspace convention: turing is an independent
# sibling project beneath this coordination root, not Speak to Me's tensors.
_TURING_ROOT = Path(__file__).resolve().parent / "turing"
if str(_TURING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TURING_ROOT))

from src.common.tensors import AbstractTensor


def _check_rows(values: AbstractTensor) -> None:
    if not isinstance(values, AbstractTensor):
        raise TypeError("expected an AbstractTensor")
    if len(values.shape) != 2 or any(size == 0 for size in values.shape):
        raise ValueError("expected a nonempty [batch, vocabulary] tensor")


def _check_log_probabilities(
    log_probs: AbstractTensor, normalization_atol: float
) -> None:
    _check_rows(log_probs)
    if not math.isfinite(normalization_atol) or not 0 <= normalization_atol < 1:
        raise ValueError("normalization_atol must be finite and in [0, 1)")
    valid = log_probs.isfinite() | (log_probs == -math.inf)
    if not valid.all().item() or (log_probs > 0).any().item():
        raise ValueError("log-probabilities must be nonpositive, finite or -inf")
    mass = log_probs.exp().sum(dim=-1, keepdim=True)
    if ((mass - 1).abs() > normalization_atol).any().item():
        raise ValueError("each complete vocabulary row must sum to probability one")


def entropy_from_log_probs(
    log_probs: AbstractTensor, *, normalization_atol: float = 1e-5
) -> AbstractTensor:
    """Return full-distribution Shannon entropy in nats for each row.

    Exact zero probabilities are represented by ``-inf`` and contribute zero.
    The normalization tolerance validates input; it does not renormalize it.
    """
    _check_log_probabilities(log_probs, normalization_atol)
    safe_logs = AbstractTensor.where(log_probs == -math.inf, 0.0, log_probs)
    return -(log_probs.exp() * safe_logs).sum(dim=-1, keepdim=True).squeeze(-1)


def raw_top_logit_margin(logits: AbstractTensor) -> AbstractTensor:
    """Return largest minus second-largest *untempered, unmasked* raw logit.

    Supply at least two finite, real floating-point logits per row. A tie has margin
    zero. Passing log-probabilities scaled by a policy temperature would not
    measure this raw-model quantity; its provenance is the caller's contract.
    """
    _check_rows(logits)
    # AbstractTensor exposes dtype names, but no floating-point predicate.
    # Reject integer/complex arithmetic rather than overflow, rank complex
    # values, or silently change the caller's numeric precision.
    if "float" not in str(logits.get_dtype()).lower():
        raise TypeError("raw margin requires a real floating-point dtype")
    if logits.shape[-1] < 2 or not logits.isfinite().all().item():
        raise ValueError("raw margin requires at least two finite logits per row")
    values, _ = AbstractTensor.topk(logits, k=2, dim=-1)
    return values[:, 0] - values[:, 1]


def jensen_shannon_divergence(
    log_p: AbstractTensor,
    log_q: AbstractTensor,
    *,
    normalization_atol: float = 1e-5,
) -> AbstractTensor:
    """Return equal-weight full-vocabulary JS divergence in nats per row.

    Pair corresponding rows, with no broadcasting. Both inputs must use the
    same backend and device and the same vocabulary ordering. Compute the
    mixture in log space, including tokens with zero mass in both inputs.
    Roundoff can make a mathematically zero result slightly negative; clamp
    that lower bound before returning. This is a finite-precision measurement,
    not a coarsened/top-k approximation and not a semantic-equivalence claim.
    """
    _check_log_probabilities(log_p, normalization_atol)
    _check_log_probabilities(log_q, normalization_atol)
    if log_p.shape != log_q.shape:
        raise ValueError("paired distributions must have identical shapes")
    if type(log_p) is not type(log_q) or log_p.get_device() != log_q.get_device():
        raise ValueError("paired distributions must share a backend and device")

    maximum = log_p.maximum(log_q)
    both_zero = maximum == -math.inf
    # Replace both-zero entries before subtraction or log, not afterwards:
    # evaluating -inf - -inf or log(0) would already create NaNs/warnings.
    safe_maximum = AbstractTensor.where(both_zero, 0.0, maximum)
    mixture_sum = (log_p - safe_maximum).exp() + (log_q - safe_maximum).exp()
    safe_sum = AbstractTensor.where(both_zero, 1.0, mixture_sum)
    log_m = safe_maximum + safe_sum.log() - math.log(2.0)
    safe_p = AbstractTensor.where(log_p == -math.inf, log_m, log_p)
    safe_q = AbstractTensor.where(log_q == -math.inf, log_m, log_q)
    terms = log_p.exp() * (safe_p - log_m) + log_q.exp() * (safe_q - log_m)
    divergence = (0.5 * terms).sum(dim=-1, keepdim=True).squeeze(-1)
    return divergence.clamp_min(0.0)


def jensen_shannon_distance(
    log_p: AbstractTensor,
    log_q: AbstractTensor,
    *,
    normalization_atol: float = 1e-5,
) -> AbstractTensor:
    """Return sqrt(JS divergence), in square-root nats, for each row."""
    return jensen_shannon_divergence(
        log_p, log_q, normalization_atol=normalization_atol
    ).sqrt()
