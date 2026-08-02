def iter_derivation(self, iterations: int) -> Iterator[str]:
        """Yield the axiom and then every complete parallel generation."""

        if iterations < 0:
            raise ValueError("iterations must be non-negative")
        word = self.axiom
        generator = random.Random(self.seed)
        yield word
        for generation in range(iterations):
            word = self.rewrite_once(word, generation=generation, rng=generator)
            yield word