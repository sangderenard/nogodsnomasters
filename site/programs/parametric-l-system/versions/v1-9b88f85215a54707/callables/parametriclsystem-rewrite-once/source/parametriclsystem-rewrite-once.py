def rewrite_once(
        self,
        word: str | None = None,
        *,
        generation: int = 0,
        rng: random.Random | None = None,
    ) -> str:
        """Apply one parallel production step to ``word``."""

        if generation < 0:
            raise ValueError("generation must be non-negative")
        source = self.axiom if word is None else word
        if not isinstance(source, str):
            raise TypeError("word must be a string")
        generator = rng or random.Random(self.seed)
        output: list[str] = []
        output_length = 0
        for index, symbol in enumerate(source):
            left = self._neighbour(source, index, -1)
            right = self._neighbour(source, index, 1)
            rule = self._matching_rule(symbol, left, right)
            if rule is None:
                replacement = symbol
            else:
                context = RewriteContext(
                    symbol=symbol,
                    index=index,
                    generation=generation,
                    left=left,
                    right=right,
                    word=source,
                    parameters=self.parameters,
                    rng=generator,
                )
                result = rule(context) if callable(rule) else rule
                replacement = self._choose_replacement(
                    result, symbol=symbol, rng=generator
                )
            output_length += len(replacement)
            if output_length > self.max_symbols:
                raise OverflowError(
                    f"generation {generation + 1} exceeded max_symbols="
                    f"{self.max_symbols}"
                )
            output.append(replacement)
        return "".join(output)