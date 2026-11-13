"""Token Orchard's replacement value class for authored Python ``str``.

The authoritative value is a text-domain token in one Vocabulary.  Host text
appears only in the explicit ingress/egress adapters.  Operator bodies use
integer atoms and the same vocabulary throughout.
"""

from __future__ import annotations

from .contract import DOMAIN_TEXT, STATUS_LIMIT_EXCEEDED, STATUS_OK, U64_MAX


class TokenOrchardStrError(ValueError):
    def __init__(self, status):
        self.status = int(status)
        super().__init__(f"Token Orchard string status {self.status}")


class TokenOrchardStr:
    """A text token and its vocabulary, with the supported ``str`` operators.

    The compiler may bind authored ``builtins.str`` values to this exact class.
    ``token`` is the compiled value; ``vocabulary`` is shared product state.
    """

    def __init__(self, vocabulary, token):
        if vocabulary.token_domain(token) != DOMAIN_TEXT:
            raise ValueError("TokenOrchardStr requires a text-domain token")
        self.vocabulary = vocabulary
        self.token = int(token)

    @classmethod
    def from_atoms(cls, vocabulary, atoms):
        status, token = vocabulary.intern(DOMAIN_TEXT, atoms)
        if status != STATUS_OK:
            raise TokenOrchardStrError(status)
        return cls(vocabulary, token)

    @classmethod
    def from_text(cls, vocabulary, text):
        """Explicit host-text ingress; operator bodies never use host text."""
        if not isinstance(text, str):
            raise TypeError("text ingress requires builtins.str")
        return cls.from_atoms(vocabulary, [ord(character) for character in text])

    def to_text(self):
        """Explicit host-text egress for the reference and tests."""
        return "".join(chr(atom) for atom in self._atoms())

    def _atoms(self):
        status, atoms = self.vocabulary.expand(self.token)
        if status != STATUS_OK:
            raise TokenOrchardStrError(status)
        return atoms

    def _result(self, result):
        status, token = result
        if status != STATUS_OK:
            raise TokenOrchardStrError(status)
        return TokenOrchardStr(self.vocabulary, token)

    def _same_vocabulary(self, other):
        if self.vocabulary is not other.vocabulary:
            raise ValueError("string operands belong to different vocabularies")

    def __add__(self, other):
        if not isinstance(other, TokenOrchardStr):
            return NotImplemented
        self._same_vocabulary(other)
        return self._result(self.vocabulary.concat(self.token, other.token))

    def __radd__(self, other):
        if not isinstance(other, TokenOrchardStr):
            return NotImplemented
        return other.__add__(self)

    def __len__(self):
        status, length = self.vocabulary.length(self.token)
        if status != STATUS_OK:
            raise TokenOrchardStrError(status)
        return length

    def __getitem__(self, key):
        length = len(self)
        if isinstance(key, slice):
            start, stop, step = key.indices(length)
            if step == 1:
                if stop < start:
                    stop = start
                return self._result(self.vocabulary.slice(self.token, start, stop))
            atoms = self._atoms()
            return TokenOrchardStr.from_atoms(
                self.vocabulary, [atoms[index] for index in range(start, stop, step)]
            )
        if not isinstance(key, int):
            raise TypeError("string indices must be integers or slices")
        index = int(key)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError("string index out of range")
        return self._result(self.vocabulary.atom_at(self.token, index))

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __mul__(self, count):
        if not isinstance(count, int):
            return NotImplemented
        copies = max(0, int(count))
        if len(self) * copies > self.vocabulary.max_sequence_atoms:
            raise TokenOrchardStrError(STATUS_LIMIT_EXCEEDED)
        return TokenOrchardStr.from_atoms(
            self.vocabulary, self._atoms() * copies
        )

    def __rmul__(self, count):
        return self.__mul__(count)

    def __contains__(self, needle):
        if not isinstance(needle, TokenOrchardStr):
            raise TypeError("string containment requires a text string")
        haystack_atoms = self._atoms()
        needle_atoms = needle._atoms()
        width = len(needle_atoms)
        for offset in range(len(haystack_atoms) - width + 1):
            if haystack_atoms[offset:offset + width] == needle_atoms:
                return True
        return False

    def __eq__(self, other):
        if not isinstance(other, TokenOrchardStr):
            return NotImplemented
        if self.vocabulary is other.vocabulary:
            return self.token == other.token
        return self._atoms() == other._atoms()

    def __ne__(self, other):
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    def _ordering(self, other, relation):
        if not isinstance(other, TokenOrchardStr):
            return NotImplemented
        left, right = self._atoms(), other._atoms()
        if relation == 0:
            return left < right
        if relation == 1:
            return left <= right
        if relation == 2:
            return left > right
        return left >= right

    def __lt__(self, other):
        return self._ordering(other, 0)

    def __le__(self, other):
        return self._ordering(other, 1)

    def __gt__(self, other):
        return self._ordering(other, 2)

    def __ge__(self, other):
        return self._ordering(other, 3)

    def __hash__(self):
        # Content hash, so equal values from distinct vocabularies agree.
        value = 14695981039346656037
        for atom in self._atoms():
            value = ((value ^ atom) * 1099511628211) & U64_MAX
        if value >= (1 << 63):
            value -= 1 << 64
        return -2 if value == -1 else value
