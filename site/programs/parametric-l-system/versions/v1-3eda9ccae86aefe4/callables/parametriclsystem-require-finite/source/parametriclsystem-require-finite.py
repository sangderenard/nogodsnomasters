def _require_finite(name: str, value: float) -> None:
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")