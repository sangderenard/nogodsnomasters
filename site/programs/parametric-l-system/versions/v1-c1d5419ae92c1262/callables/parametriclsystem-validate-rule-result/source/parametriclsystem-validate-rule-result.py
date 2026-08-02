def _validate_rule_result(self, result: RuleResult) -> None:
        if result is None or isinstance(result, str):
            return
        if isinstance(result, Mapping):
            choices = result.items()
        elif isinstance(result, Sequence):
            if not result:
                raise ValueError("a production alternative list cannot be empty")
            if all(isinstance(choice, str) for choice in result):
                return
            choices = result
        else:
            raise TypeError(f"unsupported production result: {result!r}")
        total = 0.0
        for item in choices:
            try:
                replacement, weight = item
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "weighted productions must contain (replacement, weight) pairs"
                ) from error
            if not isinstance(replacement, str):
                raise TypeError("a production replacement must be a string")
            self._require_finite("production weight", weight)
            if weight < 0:
                raise ValueError("production weights cannot be negative")
            total += float(weight)
        if total <= 0:
            raise ValueError("weighted productions need at least one positive weight")