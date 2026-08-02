def _validate_actions(self) -> None:
        for symbol, action in self.actions.items():
            if len(symbol) != 1:
                raise ValueError(f"action symbol must be one character: {symbol!r}")
            if callable(action):
                continue
            names = (action,) if isinstance(action, str) else tuple(action)
            unknown = set(names) - self.ACTION_NAMES
            if unknown:
                raise ValueError(f"unknown turtle action(s): {sorted(unknown)!r}")