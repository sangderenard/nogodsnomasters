def __init__(
        self,
        axiom: str,
        rules: Mapping[RuleKey, RuleResult | RuleCallable],
        *,
        parameters: Mapping[str, Any] | None = None,
        context_ignore: Iterable[str] = ("+", "-", "[", "]", "|"),
        actions: Mapping[str, str | Sequence[str] | ActionCallable] | None = None,
        angle_degrees: float = 90.0,
        step_length: float = 1.0,
        initial_position: Point = (0.0, 0.0),
        initial_heading_degrees: float = 0.0,
        width: float = 1.0,
        step_scale: float = 1.25,
        angle_scale: float = 1.25,
        width_scale: float = 1.25,
        angle_jitter_degrees: float = 0.0,
        length_jitter: float = 0.0,
        seed: int | str | bytes | None = None,
        max_symbols: int = 1_000_000,
        max_segments: int = 1_000_000,
        strict_branches: bool = True,
    ) -> None:
        if not isinstance(axiom, str) or not axiom:
            raise ValueError("axiom must be a non-empty string")
        if max_symbols < len(axiom) or max_symbols < 1:
            raise ValueError("max_symbols must accommodate the axiom")
        if max_segments < 0:
            raise ValueError("max_segments must be non-negative")
        self._require_positive("step_length", step_length)
        self._require_positive("width", width)
        self._require_positive("step_scale", step_scale)
        self._require_positive("angle_scale", angle_scale)
        self._require_positive("width_scale", width_scale)
        self._require_finite("angle_degrees", angle_degrees)
        self._require_finite("initial_heading_degrees", initial_heading_degrees)
        self._require_finite("angle_jitter_degrees", angle_jitter_degrees)
        self._require_finite("length_jitter", length_jitter)
        if angle_jitter_degrees < 0:
            raise ValueError("angle_jitter_degrees must be non-negative")
        if length_jitter < 0:
            raise ValueError("length_jitter must be non-negative")
        if len(initial_position) != 2 or not all(
            math.isfinite(float(value)) for value in initial_position
        ):
            raise ValueError("initial_position must contain two finite numbers")

        self.axiom = axiom
        self.rules = dict(rules)
        self.parameters = MappingProxyType(dict(parameters or {}))
        self.context_ignore = frozenset(context_ignore)
        self.actions = dict(self.DEFAULT_ACTIONS)
        if actions:
            self.actions.update(actions)
        self.angle_degrees = float(angle_degrees)
        self.step_length = float(step_length)
        self.initial_position = tuple(map(float, initial_position))
        self.initial_heading_degrees = float(initial_heading_degrees)
        self.width = float(width)
        self.step_scale = float(step_scale)
        self.angle_scale = float(angle_scale)
        self.width_scale = float(width_scale)
        self.angle_jitter_degrees = float(angle_jitter_degrees)
        self.length_jitter = float(length_jitter)
        self.seed = seed
        self.max_symbols = int(max_symbols)
        self.max_segments = int(max_segments)
        self.strict_branches = bool(strict_branches)

        self._validate_rules()
        self._validate_actions()