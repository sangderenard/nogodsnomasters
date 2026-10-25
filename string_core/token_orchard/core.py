"""Restricted-Python reference implementation of exact token interning."""

from array import array

from .kernel import fingerprint_lanes_span

from .contract import (
    AGGREGATE_BASE,
    BYTE_ATOM_BASE,
    BYTE_ATOM_LIMIT,
    BYTES_EMPTY_TOKEN,
    DEFAULT_MAX_AGGREGATES,
    DEFAULT_MAX_BATCH_ITEMS,
    DEFAULT_MAX_SEQUENCE_ATOMS,
    DOMAIN_BYTES,
    DOMAIN_TEXT,
    FINGERPRINT_CONSTANT,
    FINGERPRINT_FNV128,
    STATUS_INVALID_ATOM,
    STATUS_INVALID_DOMAIN,
    STATUS_INVALID_OFFSETS,
    STATUS_INVALID_SLICE,
    STATUS_INVALID_TOKEN,
    STATUS_LIMIT_EXCEEDED,
    STATUS_OK,
    SURROGATE_BEGIN,
    SURROGATE_COUNT,
    SURROGATE_END,
    TEXT_ATOM_BASE,
    TEXT_ATOM_LIMIT,
    TEXT_EMPTY_TOKEN,
    TOKEN_KIND_AGGREGATE,
    TOKEN_KIND_ATOM,
    TOKEN_KIND_EMPTY,
    TOKEN_KIND_INVALID,
    U64_MAX,
    U32_MAX,
    UNICODE_LIMIT,
)


def cover_candidate_is_better(
    candidate_count: int,
    candidate_atom_count: int,
    candidate_token: int,
    selected_count: int,
    selected_atom_count: int,
    selected_token: int,
) -> bool:
    """Compare deterministic cover costs without tuple-temporary state."""
    if candidate_count != selected_count:
        return candidate_count < selected_count
    if candidate_atom_count != selected_atom_count:
        return candidate_atom_count < selected_atom_count
    return candidate_token < selected_token


def choose_cover_candidate(
    candidate_count: int,
    candidate_atom_count: int,
    candidate_token: int,
    candidate_next: int,
    selected_count: int,
    selected_atom_count: int,
    selected_token: int,
    selected_next: int,
) -> tuple[int, int, int, int]:
    """Return the better candidate as one total scalar state transition."""
    if cover_candidate_is_better(
        candidate_count,
        candidate_atom_count,
        candidate_token,
        selected_count,
        selected_atom_count,
        selected_token,
    ):
        return (
            candidate_count,
            candidate_atom_count,
            candidate_token,
            candidate_next,
        )
    return (
        selected_count,
        selected_atom_count,
        selected_token,
        selected_next,
    )


class Vocabulary:
    """One deterministic dual-domain token vocabulary.

    Aggregate state is held in aligned structure-of-arrays fields. Dictionaries
    are derived lookup indexes only; exact atom rows remain the identity proof.
    """

    def __init__(
        self,
        fingerprint_mode: int = 0,
        max_sequence_atoms: int = DEFAULT_MAX_SEQUENCE_ATOMS,
        max_batch_items: int = DEFAULT_MAX_BATCH_ITEMS,
        max_aggregates: int = DEFAULT_MAX_AGGREGATES,
    ):
        if fingerprint_mode != FINGERPRINT_FNV128 and fingerprint_mode != FINGERPRINT_CONSTANT:
            raise ValueError("unsupported fingerprint mode")
        if (
            not isinstance(max_sequence_atoms, int)
            or isinstance(max_sequence_atoms, bool)
            or max_sequence_atoms < 0
            or max_sequence_atoms > U32_MAX
            or not isinstance(max_batch_items, int)
            or isinstance(max_batch_items, bool)
            or max_batch_items < 0
            or max_batch_items > U64_MAX
            or not isinstance(max_aggregates, int)
            or isinstance(max_aggregates, bool)
            or max_aggregates < 0
            or max_aggregates > U64_MAX - AGGREGATE_BASE + 1
        ):
            raise ValueError("invalid vocabulary limits")
        self.fingerprint_mode = fingerprint_mode
        self.max_sequence_atoms = max_sequence_atoms
        self.max_batch_items = max_batch_items
        self.max_aggregates = max_aggregates
        self.epoch = 0

        self.domain = array("B")
        self.atom_length = array("Q")
        self.fingerprint_hi = array("Q")
        self.fingerprint_lo = array("Q")
        self.exact_begin = array("Q")
        self.exact_count = array("I")
        self.cover_begin = array("Q")
        self.cover_count = array("I")
        self.created_epoch = array("Q")
        self.exact_atoms = array("I")
        self.cover_tokens = array("Q")

        self._candidates = {}
        self._prefix = {}

    def _valid_domain(self, domain: int) -> bool:
        return domain == DOMAIN_TEXT or domain == DOMAIN_BYTES

    def _valid_atom(self, domain: int, atom: int) -> bool:
        if not isinstance(atom, int) or isinstance(atom, bool):
            return False
        if domain == DOMAIN_BYTES:
            return 0 <= atom <= 255
        if domain == DOMAIN_TEXT:
            return 0 <= atom < UNICODE_LIMIT and not (
                SURROGATE_BEGIN <= atom < SURROGATE_END
            )
        return False

    def _empty_token(self, domain: int) -> int:
        if domain == DOMAIN_TEXT:
            return TEXT_EMPTY_TOKEN
        return BYTES_EMPTY_TOKEN

    def _atom_token(self, domain: int, atom: int) -> int:
        if domain == DOMAIN_BYTES:
            return BYTE_ATOM_BASE + atom
        rank = atom
        if atom >= SURROGATE_END:
            rank = atom - SURROGATE_COUNT
        return TEXT_ATOM_BASE + rank

    def _atom_from_token(self, token: int) -> tuple[int, int]:
        if TEXT_ATOM_BASE <= token < TEXT_ATOM_LIMIT:
            rank = token - TEXT_ATOM_BASE
            if rank >= SURROGATE_BEGIN:
                rank += SURROGATE_COUNT
            return DOMAIN_TEXT, rank
        if BYTE_ATOM_BASE <= token < BYTE_ATOM_LIMIT:
            return DOMAIN_BYTES, token - BYTE_ATOM_BASE
        return 0, 0

    def _row(self, token: int) -> int:
        return token - AGGREGATE_BASE

    def token_kind(self, token: int) -> int:
        if token == TEXT_EMPTY_TOKEN or token == BYTES_EMPTY_TOKEN:
            return TOKEN_KIND_EMPTY
        if TEXT_ATOM_BASE <= token < BYTE_ATOM_LIMIT:
            return TOKEN_KIND_ATOM
        row = self._row(token)
        if 0 <= row < len(self.domain):
            return TOKEN_KIND_AGGREGATE
        return TOKEN_KIND_INVALID

    def token_domain(self, token: int) -> int:
        if token == TEXT_EMPTY_TOKEN:
            return DOMAIN_TEXT
        if token == BYTES_EMPTY_TOKEN:
            return DOMAIN_BYTES
        domain, unused = self._atom_from_token(token)
        if domain != 0:
            return domain
        row = self._row(token)
        if 0 <= row < len(self.domain):
            return self.domain[row]
        return 0

    def _fingerprint(self, domain: int, atoms: list[int]) -> tuple[int, int]:
        return self._fingerprint_span(domain, atoms, 0, len(atoms))

    def _fingerprint_span(
        self, domain: int, atoms: list[int], begin: int, end: int
    ) -> tuple[int, int]:
        if self.fingerprint_mode == FINGERPRINT_CONSTANT:
            return 0, 0
        first_high, first_low, second_high, second_low = fingerprint_lanes_span(
            domain, atoms, begin, end
        )
        first = first_high * (1 << 32) + first_low
        second = second_high * (1 << 32) + second_low
        return first, second

    def _aggregate_atoms_equal(self, row: int, atoms: list[int]) -> bool:
        return self._aggregate_atoms_equal_span(row, atoms, 0, len(atoms))

    def _aggregate_atoms_equal_span(
        self, row: int, atoms: list[int], begin: int, end: int
    ) -> bool:
        count = end - begin
        if self.exact_count[row] != count:
            return False
        source = self.exact_begin[row]
        i = 0
        while i < count:
            if self.exact_atoms[source + i] != atoms[begin + i]:
                return False
            i += 1
        return True

    def _find_existing(
        self, domain: int, atoms: list[int], hi: int, lo: int
    ) -> int:
        return self._find_existing_span(domain, atoms, 0, len(atoms), hi, lo)

    def _find_existing_span(
        self,
        domain: int,
        atoms: list[int],
        begin: int,
        end: int,
        hi: int,
        lo: int,
    ) -> int:
        key = (domain, hi, lo, end - begin)
        candidates = self._candidates.get(key, ())
        for token in candidates:
            row = self._row(token)
            if self._aggregate_atoms_equal_span(row, atoms, begin, end):
                return token
        return 0

    def _aggregate_matches(
        self, token: int, atoms: list[int], begin: int
    ) -> bool:
        row = self._row(token)
        count = self.exact_count[row]
        if begin + count > len(atoms):
            return False
        source = self.exact_begin[row]
        i = 0
        while i < count:
            if self.exact_atoms[source + i] != atoms[begin + i]:
                return False
            i += 1
        return True

    def _cover(self, domain: int, atoms: list[int]) -> list[int]:
        return self._cover_span(domain, atoms, 0, len(atoms))

    def _cover_span(
        self, domain: int, atoms: list[int], begin: int, end: int
    ) -> list[int]:
        count = end - begin
        best_count = [count + 1] * (count + 1)
        best_atom_count = [0] * (count + 1)
        chosen_token = [0] * count
        chosen_next = [0] * count
        best_count[count] = 0

        i = count - 1
        while i >= 0:
            atom_index = begin + i
            atom_token = self._atom_token(domain, atoms[atom_index])
            selected_token = atom_token
            selected_next = i + 1
            selected_count = 1 + best_count[selected_next]
            selected_atom_count = 1 + best_atom_count[selected_next]
            prefix_candidates = self._prefix.get((domain, atoms[atom_index]), ())
            for token in prefix_candidates:
                row = self._row(token)
                length = self.exact_count[row]
                if self._aggregate_matches(token, atoms, atom_index):
                    next_index = i + length
                    candidate_count = 1 + best_count[next_index]
                    candidate_atom_count = best_atom_count[next_index]
                    (
                        selected_count,
                        selected_atom_count,
                        selected_token,
                        selected_next,
                    ) = choose_cover_candidate(
                        candidate_count,
                        candidate_atom_count,
                        token,
                        next_index,
                        selected_count,
                        selected_atom_count,
                        selected_token,
                        selected_next,
                    )
            best_count[i] = selected_count
            best_atom_count[i] = selected_atom_count
            chosen_token[i] = selected_token
            chosen_next[i] = selected_next
            i -= 1

        cover = [0] * count
        cover_count = 0
        i = 0
        while i < count:
            cover[cover_count] = chosen_token[i]
            cover_count += 1
            i = chosen_next[i]
        return cover[:cover_count]

    def _validate_atoms(self, domain: int, atoms: list[int]) -> int:
        return self._validate_atom_span(domain, atoms, 0, len(atoms))

    def _validate_atom_span(
        self, domain: int, atoms: list[int], begin: int, end: int
    ) -> int:
        if not self._valid_domain(domain):
            return STATUS_INVALID_DOMAIN
        if end - begin > self.max_sequence_atoms:
            return STATUS_LIMIT_EXCEEDED
        index = begin
        while index < end:
            if not self._valid_atom(domain, atoms[index]):
                return STATUS_INVALID_ATOM
            index += 1
        return STATUS_OK

    def _intern_validated(
        self,
        domain: int,
        atoms: list[int],
        fingerprint: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        return self._intern_validated_span(
            domain, atoms, 0, len(atoms), fingerprint
        )

    def _intern_validated_span(
        self,
        domain: int,
        atoms: list[int],
        begin: int,
        end: int,
        fingerprint: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        atom_count = end - begin
        if atom_count == 0:
            return STATUS_OK, self._empty_token(domain)
        if atom_count == 1:
            return STATUS_OK, self._atom_token(domain, atoms[begin])

        if fingerprint is None:
            hi, lo = self._fingerprint_span(domain, atoms, begin, end)
        else:
            hi, lo = fingerprint
        existing = self._find_existing_span(
            domain, atoms, begin, end, hi, lo
        )
        if existing != 0:
            return STATUS_OK, existing
        if len(self.domain) >= self.max_aggregates:
            return STATUS_LIMIT_EXCEEDED, 0
        token = AGGREGATE_BASE + len(self.domain)
        if token > U64_MAX:
            return STATUS_LIMIT_EXCEEDED, 0

        cover = self._cover_span(domain, atoms, begin, end)
        self.epoch += 1
        row = len(self.domain)
        self.domain.append(domain)
        self.atom_length.append(atom_count)
        self.fingerprint_hi.append(hi)
        self.fingerprint_lo.append(lo)
        self.exact_begin.append(len(self.exact_atoms))
        self.exact_count.append(atom_count)
        self.cover_begin.append(len(self.cover_tokens))
        self.cover_count.append(len(cover))
        self.created_epoch.append(self.epoch)
        atom_index = begin
        while atom_index < end:
            self.exact_atoms.append(atoms[atom_index])
            atom_index += 1
        self.cover_tokens.extend(cover)

        key = (domain, hi, lo, atom_count)
        self._candidates.setdefault(key, []).append(token)
        self._prefix.setdefault((domain, atoms[begin]), []).append(token)
        return STATUS_OK, AGGREGATE_BASE + row

    def validate_batch_envelope(
        self, domain: int, values: list[int], offsets: list[int]
    ) -> tuple[int, int]:
        """Validate batch structure without inspecting atoms or mutating state."""
        if not self._valid_domain(domain):
            return STATUS_INVALID_DOMAIN, 0
        if len(offsets) == 0 or offsets[0] != 0:
            return STATUS_INVALID_OFFSETS, 0
        item_count = len(offsets) - 1
        if item_count > self.max_batch_items:
            return STATUS_LIMIT_EXCEEDED, 0
        previous = 0
        for offset in offsets:
            if not isinstance(offset, int) or isinstance(offset, bool):
                return STATUS_INVALID_OFFSETS, 0
            if offset < previous or offset > len(values):
                return STATUS_INVALID_OFFSETS, 0
            previous = offset
        if offsets[-1] != len(values):
            return STATUS_INVALID_OFFSETS, 0
        return STATUS_OK, item_count

    def intern_batch(
        self, domain: int, values: list[int], offsets: list[int]
    ) -> tuple[int, list[int], list[int]]:
        envelope_status, item_count = self.validate_batch_envelope(
            domain, values, offsets
        )
        statuses = [STATUS_OK] * item_count
        tokens = [0] * item_count
        if envelope_status == STATUS_OK:
            i = 0
            while i < item_count:
                begin = offsets[i]
                end = offsets[i + 1]
                status = self._validate_atom_span(domain, values, begin, end)
                if status == STATUS_OK:
                    status, token = self._intern_validated_span(
                        domain, values, begin, end
                    )
                else:
                    token = 0
                statuses[i] = status
                tokens[i] = token
                i += 1
        return envelope_status, statuses, tokens

    def intern_preflighted_batch(
        self,
        domain: int,
        values: list[int],
        offsets: list[int],
        item_statuses: list[int],
        fingerprints: list[tuple[int, int] | None],
    ) -> tuple[int, list[int], list[int]]:
        """Commit a tensor-screened batch in exact FIFO token order."""
        envelope_status, item_count = self.validate_batch_envelope(
            domain, values, offsets
        )
        statuses = [STATUS_OK] * item_count
        tokens = [0] * item_count
        if envelope_status != STATUS_OK:
            return envelope_status, statuses, tokens
        if len(item_statuses) != item_count or len(fingerprints) != item_count:
            raise ValueError("preflight result does not match batch item count")

        item_index = 0
        while item_index < item_count:
            status = item_statuses[item_index]
            if status == STATUS_OK:
                begin = offsets[item_index]
                end = offsets[item_index + 1]
                fingerprint = fingerprints[item_index]
                if self.fingerprint_mode == FINGERPRINT_CONSTANT:
                    fingerprint = (0, 0)
                if fingerprint is None or len(fingerprint) != 2:
                    raise ValueError("invalid preflight fingerprint")
                fingerprint_hi = fingerprint[0]
                fingerprint_lo = fingerprint[1]
                if (
                    not isinstance(fingerprint_hi, int)
                    or isinstance(fingerprint_hi, bool)
                    or fingerprint_hi < 0
                    or fingerprint_hi > U64_MAX
                    or not isinstance(fingerprint_lo, int)
                    or isinstance(fingerprint_lo, bool)
                    or fingerprint_lo < 0
                    or fingerprint_lo > U64_MAX
                ):
                    raise ValueError("invalid preflight fingerprint")
                status, token = self._intern_validated_span(
                    domain,
                    values,
                    begin,
                    end,
                    (fingerprint_hi, fingerprint_lo),
                )
            else:
                token = 0
            statuses[item_index] = status
            tokens[item_index] = token
            item_index += 1
        return envelope_status, statuses, tokens

    def intern(self, domain, atoms):
        batch_status, statuses, tokens = self.intern_batch(
            domain, atoms, [0, len(atoms)]
        )
        if batch_status != STATUS_OK:
            return batch_status, 0
        return statuses[0], tokens[0]

    def graph_delta_since_row(self, first_row):
        """Copy newly minted graph rows as flat integer cover topology.

        This is an advisory attachment boundary. The returned data cannot
        mutate vocabulary state and carries no authority over token identity.
        """
        row_count = len(self.domain)
        if (
            not isinstance(first_row, int)
            or isinstance(first_row, bool)
            or first_row < 0
            or first_row > row_count
        ):
            raise ValueError("invalid first graph-delta row")
        begin_epoch = (
            self.epoch if first_row == row_count
            else int(self.created_epoch[first_row]) - 1
        )
        parent_count = row_count - first_row
        parent_tokens = [0] * parent_count
        cover_offsets = [0] * (parent_count + 1)
        cover_value_count = 0
        row = first_row
        while row < row_count:
            cover_value_count += int(self.cover_count[row])
            cover_offsets[row - first_row + 1] = cover_value_count
            row += 1
        cover_values = [0] * cover_value_count
        value_index = 0
        row = first_row
        while row < row_count:
            parent_tokens[row - first_row] = AGGREGATE_BASE + row
            source = int(self.cover_begin[row])
            source_end = source + int(self.cover_count[row])
            while source < source_end:
                cover_values[value_index] = int(self.cover_tokens[source])
                value_index += 1
                source += 1
            row += 1
        return (
            begin_epoch,
            self.epoch,
            parent_tokens,
            cover_values,
            cover_offsets,
        )

    def expand(self, token):
        kind = self.token_kind(token)
        if kind == TOKEN_KIND_INVALID:
            return STATUS_INVALID_TOKEN, []
        if kind == TOKEN_KIND_EMPTY:
            return STATUS_OK, []
        if kind == TOKEN_KIND_ATOM:
            unused, atom = self._atom_from_token(token)
            return STATUS_OK, [atom]
        row = self._row(token)
        begin = self.exact_begin[row]
        count = self.exact_count[row]
        return STATUS_OK, list(self.exact_atoms[begin : begin + count])

    def cover(self, token):
        kind = self.token_kind(token)
        if kind == TOKEN_KIND_INVALID:
            return STATUS_INVALID_TOKEN, []
        if kind == TOKEN_KIND_EMPTY:
            return STATUS_OK, []
        if kind == TOKEN_KIND_ATOM:
            return STATUS_OK, [token]
        row = self._row(token)
        begin = self.cover_begin[row]
        count = self.cover_count[row]
        return STATUS_OK, list(self.cover_tokens[begin : begin + count])

    def length(self, token):
        kind = self.token_kind(token)
        if kind == TOKEN_KIND_INVALID:
            return STATUS_INVALID_TOKEN, 0
        if kind == TOKEN_KIND_EMPTY:
            return STATUS_OK, 0
        if kind == TOKEN_KIND_ATOM:
            return STATUS_OK, 1
        return STATUS_OK, self.atom_length[self._row(token)]

    def concat(self, left, right):
        left_domain = self.token_domain(left)
        if left_domain == 0 or left_domain != self.token_domain(right):
            return STATUS_INVALID_TOKEN, 0
        left_status, left_atoms = self.expand(left)
        right_status, right_atoms = self.expand(right)
        if left_status != STATUS_OK or right_status != STATUS_OK:
            return STATUS_INVALID_TOKEN, 0
        return self.intern(left_domain, left_atoms + right_atoms)

    def slice(self, token, begin, end):
        domain = self.token_domain(token)
        if domain == 0:
            return STATUS_INVALID_TOKEN, 0
        status, atoms = self.expand(token)
        if status != STATUS_OK:
            return status, 0
        if (
            not isinstance(begin, int)
            or isinstance(begin, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or begin < 0
            or end < begin
            or end > len(atoms)
        ):
            return STATUS_INVALID_SLICE, 0
        return self.intern(domain, atoms[begin:end])

    def atom_at(self, token, index):
        domain = self.token_domain(token)
        if domain == 0:
            return STATUS_INVALID_TOKEN, 0
        status, atoms = self.expand(token)
        if status != STATUS_OK:
            return status, 0
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(atoms)
        ):
            return STATUS_INVALID_SLICE, 0
        return STATUS_OK, self._atom_token(domain, atoms[index])

    def state_tuple(self):
        """Return deterministic authoritative state for tests and receipts."""
        return (
            self.epoch,
            tuple(self.domain),
            tuple(self.atom_length),
            tuple(self.fingerprint_hi),
            tuple(self.fingerprint_lo),
            tuple(self.exact_begin),
            tuple(self.exact_count),
            tuple(self.cover_begin),
            tuple(self.cover_count),
            tuple(self.created_epoch),
            tuple(self.exact_atoms),
            tuple(self.cover_tokens),
        )
