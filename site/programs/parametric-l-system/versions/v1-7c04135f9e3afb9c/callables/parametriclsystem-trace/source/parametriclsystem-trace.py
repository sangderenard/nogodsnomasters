def trace(self, iterations: int) -> TurtleTrace:
        """Derive ``iterations`` generations and interpret the final word."""

        return self.interpret(self.derive(iterations))