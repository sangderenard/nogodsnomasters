def _interpretation_seed(self) -> int | str | bytes | None:
        if self.seed is None:
            return None
        # Give turtle jitter an independent deterministic stream. Using repr
        # also handles every seed type accepted by random.Random uniformly.
        return f"{self.seed!r}:turtle"