def __call__(self, iterations: int) -> TurtleTrace:
        """Shorthand for :meth:`trace`."""

        return self.trace(iterations)