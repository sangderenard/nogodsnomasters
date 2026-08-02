def _require_positive(cls, name: str, value: float) -> None:
        cls._require_finite(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")