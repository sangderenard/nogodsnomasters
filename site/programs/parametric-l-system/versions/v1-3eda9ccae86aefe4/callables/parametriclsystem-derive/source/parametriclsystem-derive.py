def derive(self, iterations: int) -> str:
        """Return the word after ``iterations`` parallel rewrites."""

        word = self.axiom
        for word in self.iter_derivation(iterations):
            pass
        return word