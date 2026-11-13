# Root-owned probability geometry research

## Result

31 focused tests passed against the real Turing NumPy backend. This is a
root-level research prototype: entropy, raw top-logit margin, Jensen–Shannon
(JS) divergence and sqrt(JS) on complete vocabulary rows. It is not a model
experiment or retention integration.

## Ownership and provenance

- Root: `sangderenard/nogodsnomasters`, `nogodsnomasters`,
  `077c224ef0e58aa9970237ceccb9cd369e71159a`.
- Independent `turing/`: `sangderenard/turing`, `main`,
  `3559a8fdcae91ea773dbb433a23f9bad9b9c117a`.
- The import convention matches `engine_toy/graph_physics.py`: put the actual
  root-local `turing/` checkout on the import path. Import
  `src.common.tensors.AbstractTensor`; import its NumPy implementation from
  `src.common.tensors.numpy_backend`, since the package does not export backends.
- The identity fixture checks the resolved source files and proves the prototype,
  package export, abstraction module and backend refer to the same AbstractTensor
  class. The legacy top-level `tensors` package is absent from loaded modules.
- Speak to Me and its agent environment are unchanged. Turing is unchanged.
  No substitute tensor implementation, copied backend or fake dependency is used.
- The earlier unpublished Speak to Me prototype supplied the arithmetic design;
  it was never runtime verified there. This root experiment is separately tested.

## Reproduction

Use the two repositories at the revisions above, with `turing/` beneath this
root, as independent projects share the Windows workspace root. From this root:

```sh
python -m venv .research-venv
.research-venv/bin/python -m pip install --only-binary=:all: -r turing/requirements.txt pytest
.research-venv/bin/python -m pytest -q -s test_probability_geometry_research.py
.research-venv/bin/python -m pip check
```

Windows venv executables are under `.research-venv\Scripts\`. The execution
record here is Linux/Python 3.12. Turing publishes `requirements.txt` but has no
root setup script or packaging manifest; the tensor guidance's linked
`src/common/ENV_SETUP_OPTIONS.md` is absent. The isolated research environment
uses the entire published requirements file plus the explicitly additional
pytest test runner. It does not claim a minimal dependency closure and does not
reconfigure the Speak to Me environment. Only binary wheels were installed;
no Torch, models, GPU components or native build were requested.

Observed key versions: NumPy 2.5.3, NetworkX 3.7, SymPy 1.14.0, pytest 9.1.1.
Requirements are unpinned upstream; these versions are a run receipt, not a lock.

## Verification

- `python -m pytest -q -s test_probability_geometry_research.py` using the isolated
  environment: **31 passed, 1 warning in 0.33 s**.
- `python -m pip check`: **No broken requirements found**.
- The one import warning announces the absent optional Nodus tensor arena. The
  tested instances are explicitly the real NumPy backend. The warning was not
  hidden and no arena or native compiler result is claimed.
- Fixtures cover analytical entropy; raw margin, ties, shifts and temperature;
  self-zero JS, symmetry, a hand-computable value, disjoint support at ln(2),
  common zero support, batch shapes and no broadcasting; malformed inputs;
  integer/complex margin rejection; and finite extreme-tail arithmetic in
  float32/float64 with invalid/divide/overflow operations set to raise.
- Full root/Turing suites, compiler parity, autograd, other backends and real
  model integration were not run. Turing's test hazards explicitly caution
  against full-suite baselining. These are focused eager NumPy receipts only.

## Numerical and research contract

Rows are normalized natural-log probabilities over the same complete ordered
vocabulary. The input check cannot detect a renormalized shortlist; vocabulary,
model and exact evaluated-context provenance remain the caller's responsibility.
The normalization tolerance validates mass, never renormalizes the input.
JS uses a max-shifted log mixture, evaluates zero-mass terms by their limits,
and clamps its mathematical lower bound before sqrt. Tests do not enlarge
comparison tolerances to disguise substrate failures.

The source at Turing `numpy_backend.py:272` still computes log-softmax as
exp-normalize-log. This is a source observation, not a runtime regression result:
finite tiny tails can underflow before log. This prototype accepts log
probabilities supplied by callers and does not claim to repair upstream loss.

No graph, retention owner, allocator, backward-score provenance, physics or
production caller is changed. The next research increment must establish
full-distribution observation provenance before attaching measurements to
retained model observations.

## Prompt history

> it's almost better to keep the agent environment but do new testing in the root repo but this is kinda your thing i want, trying to clean the museum and bring back research departments

This follows the standing request to inspect the multirepo system, find bugs,
implement planned pieces and present them for review. It preserves the existing
agent environment and puts this bounded research increment in the root repository.
