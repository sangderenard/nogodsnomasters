def _validate_rules(self) -> None:
        for key, rule in self.rules.items():
            if isinstance(key, str):
                if len(key) != 1:
                    raise ValueError(f"rule symbol must be one character: {key!r}")
            elif isinstance(key, tuple) and len(key) == 3:
                left, symbol, right = key
                if len(symbol) != 1:
                    raise ValueError(f"rule symbol must be one character: {symbol!r}")
                if any(
                    neighbour is not None and len(neighbour) != 1
                    for neighbour in (left, right)
                ):
                    raise ValueError("context neighbours must be one character or None")
            else:
                raise TypeError(f"invalid rule key: {key!r}")
            if not callable(rule):
                self._validate_rule_result(rule)