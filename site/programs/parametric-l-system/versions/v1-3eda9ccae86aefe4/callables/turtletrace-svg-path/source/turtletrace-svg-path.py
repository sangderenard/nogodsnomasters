def svg_path(self, *, precision: int = 6) -> str:
        """Return independent SVG move/line commands for all drawn edges."""

        if precision < 0:
            raise ValueError("precision must be non-negative")
        number = lambda value: format(value, f".{precision}f").rstrip("0").rstrip(".") or "0"
        return " ".join(
            f"M {number(edge.start[0])} {number(edge.start[1])} "
            f"L {number(edge.end[0])} {number(edge.end[1])}"
            for edge in self.segments
        )