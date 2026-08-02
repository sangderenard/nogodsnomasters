def _neighbour(self, word: str, index: int, direction: int) -> str | None:
        cursor = index + direction
        while 0 <= cursor < len(word):
            candidate = word[cursor]
            if candidate not in self.context_ignore:
                return candidate
            cursor += direction
        return None