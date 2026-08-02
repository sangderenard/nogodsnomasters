def _action_names(
        self,
        action: str | Sequence[str] | ActionCallable,
        context: TurtleContext,
    ) -> tuple[str, ...]:
        result = action(context) if callable(action) else action
        if result is None:
            return ()
        names = (result,) if isinstance(result, str) else tuple(result)
        unknown = set(names) - self.ACTION_NAMES
        if unknown:
            raise ValueError(f"unknown turtle action(s): {sorted(unknown)!r}")
        return names