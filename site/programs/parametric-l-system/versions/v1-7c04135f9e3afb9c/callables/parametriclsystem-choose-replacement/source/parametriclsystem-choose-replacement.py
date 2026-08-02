def _choose_replacement(
        self, result: RuleResult, *, symbol: str, rng: random.Random
    ) -> str:
        if result is None:
            return symbol
        self._validate_rule_result(result)
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping):
            population = tuple(result)
            weights = tuple(float(result[item]) for item in population)
        elif all(isinstance(choice, str) for choice in result):
            population = tuple(result)
            weights = None
        else:
            pairs = tuple(result)
            population = tuple(pair[0] for pair in pairs)
            weights = tuple(float(pair[1]) for pair in pairs)
        return rng.choices(population, weights=weights, k=1)[0]