def points(self) -> tuple[Point, ...]:
        """Segment endpoints, useful for plotting and simple serialization."""

        if not self.segments:
            return (self.final_state.position,)
        return tuple(
            point
            for segment in self.segments
            for point in (segment.start, segment.end)
        )