# Key change protocol: forcing key and crypto changes across the site safely

Owner, 2026-11-21: "establish protocol for forcing key changes across the site safely, any time we need to change our
crypto". This is that protocol. It covers two kinds of change with one procedure:

- **Key rotation**: the same algorithms, new keys (routine, on a schedule, or because a key may be exposed).
- **Crypto change**: new algorithms (a suite is weakened, a standard changes, a quantum-capable attacker becomes real).

Everything below is written so a change can be *forced*: carried out against a holder that is offline, slow or
compromised, without any window in which nothing is trusted and without trusting what the change is replacing.

## 1. Principles

1. **Every format names its suite.** No signature, sealed blob or key file is "just bytes": each carries a suite id,
   and every verifier holds an allowlist of suites it accepts. A crypto change is then adding a suite, running both,
   and removing the old one from allowlists, never a flag day that breaks parsing.
2. **Every key has an epoch.** A node's identity is its *lineage* (the kid of its first key, stable for life); its
   current key is lineage + epoch. Whoever checks "is this the writer" compares lineages, so rotation never breaks the
   one-writer rule.
3. **Hybrid until proven otherwise.** Signatures are classical + post-quantum together (both must verify); key
   agreement is classical + post-quantum combined through HKDF (both must be broken). Today: Ed25519 + ML-DSA-65,
   X25519 + ML-KEM-768, content ChaCha20-Poly1305.
4. **Planned changes are cross-signed.** A new key is announced in a record signed by the old key *and* the new key:
   the old proves continuity, the new proves possession. Peers accept it only from a key they already trust.
5. **Forced changes come from a root kept offline.** If a key may be compromised, its own signature proves nothing.
   A site root key, held offline by the owner (like `owner-recovery.key`), signs revocations and replacements that
   every holder obeys at once.
6. **Overlap, then cutover, then retire.** Old and new are both accepted for a bounded grace period; after cutover only
   the new is accepted; after retirement the old key is destroyed. A forced change has no grace period.
7. **Nothing in the clear, nothing replayed.** Records carry generations or times and are refused if older than what
   the holder already has. Private keys never leave the machine that made them; only public blocks travel.
8. **Every change is recorded.** Each rotation, revocation and suite change is an entry in the service SQL (the one
   identity authority) and an experience report, with the old and new kids, the reason and who ordered it.

## 2. Inventory: what holds keys, and how each changes

Paths are under `srv/state/secrets/` unless stated. Classes: **A** asymmetric identity (rotate by this protocol's
records), **S** symmetric service secret (rotate by dual-key overlap), **T** third-party token (rotate at the provider),
**X** TLS.

| Key | Class | Algorithm today | Used by | How it changes |
|---|---|---|---|---|
| Mesh node keys (`NODUS_MESH_KEYS` folder) | A | Ed25519+ML-DSA-65 / X25519+ML-KEM-768 | nodus `mesh_crypto.py` rows | §4 planned rotation record; §5 forced by site root |
| Region keys (`region/sign.key`, `box.key`) | A | Ed25519 / X25519 | turing `state_broadcast.py` | §4, once state_broadcast carries suites and epochs (§7); until then re-enroll |
| Owner recovery key (`C:\Users\alber\speaktome-recovery\owner-recovery.key`) | A | X25519 | state_broadcast recipients | re-enroll a new recovery key, re-pack, destroy the old; owner only |
| Site root key (new, offline) | A | Ed25519+ML-DSA-65 | revocations, forced replacements | owner makes a new root; every node gets the new `root.pub` signed by the old root |
| `records.key` | S | records cipher (`secure_records.py`) | sealed logs and retained records | §6: new key beside the old, re-seal, drop the old |
| `private_hash.key` | S | keyed hash | redaction aliases | §6; aliases change, so redacted exports are re-made |
| `nodus_link.key` | S | HMAC-SHA256 tickets | link tickets (`link_tickets.py`, link hosts) | §6: hosts accept both keys for the ticket lifetime, then the old is dropped |
| `admin_token*.txt`, `service_tokens/` | S | bearer tokens | admin bootstrap, services | issue new, restart holders, revoke old |
| `operator_credentials/*.secret`, `user_credentials/` | S | account secrets | operator and agent accounts | per account through the service (the account's own reset path) |
| `cloudflare/`, `modal/`, `github/`, `mcp_credentials/`, Alpaca | T | provider tokens | DNS, Modal front, landing, Suno, paper trading | issue a new token at the provider, install, revoke the old there |
| `tls/` (local CA), `tls-public/`, Let's Encrypt on Oracle | X | RSA/ECDSA certificates | HTTPS | Let's Encrypt renews itself; the local CA is re-issued and re-trusted; a forced change revokes at the CA |

The inventory is part of the protocol: a new key anywhere on the site is added to this table in the same change that
creates it.

## 3. Suites

A suite id is a small integer every format carries. Today:

| Suite | Signatures | Key agreement | Content | Status |
|---|---|---|---|---|
| 1 | Ed25519 + ML-DSA-65 | X25519 + ML-KEM-768, HKDF-SHA256 | ChaCha20-Poly1305 | current |

A crypto change adds suite *n*+1 to the code, then: (1) every holder learns to *verify* it (allowlist gains it);
(2) writers start *producing* it; (3) once every reader has moved, allowlists drop the old suite; (4) old keys of that
suite are retired. A suite is never removed from an allowlist while any holder still produces it, unless it is being
forced out (§5), in which case refusal is the point.

## 4. Planned rotation (the old key is trusted)

1. **Prepare.** The holder makes a new key set (epoch *e*+1) beside the current one. Nothing changes yet.
2. **Announce.** It publishes a rotation record: lineage, old kid, new public block, effective time, grace period,
   signed by the old key and the new key (both hybrids). The record travels the way the holder's data travels (on the
   mesh, the canvas row's state path; for regions, the broadcast) and is also delivered by orchestration.
3. **Accept.** Each peer checks both signatures, that the old kid is the lineage's current trusted key, and that the
   record is newer than any it holds; then installs the new key as the lineage's next epoch.
4. **Overlap.** Until the effective time plus grace, the peer accepts either epoch; the holder signs with the new one.
5. **Cutover.** After the grace period the old epoch is refused for that lineage.
6. **Retire.** The holder destroys the old private keys; the record is kept as history.

## 5. Forced change (the old key is not trusted)

Used when a key may be exposed, a suite is broken, or a holder is unreachable.

1. The owner, offline, signs a force record with the site root: the revoked kids (or a whole suite), the replacement
   public block if there is one, and the time. No grace period.
2. Orchestration delivers it to every holder (and it travels the mesh like any record).
3. Each holder checks the root signature against its `root.pub` and the record's newness, then at once refuses the
   revoked kids or suite and installs the replacement as the lineage's next epoch.
4. Everything the revoked key ever sealed is treated as readable by an attacker: the owner of that data re-seals or
   re-issues it under the replacement (for regions, a fresh generation packed to the new keys).
5. The root key itself is changed the same way: a new root public block signed by the old root; if the old root is
   compromised, `root.pub` is replaced by hand on every machine by the owner.

## 6. Symmetric secrets (class S)

There is no peer to cross-sign for a shared secret, so the overlap is in the holder: install the new secret beside the
old (`<name>.next`), accept both, write with the new, re-seal or re-issue what was made with the old, then drop the old.
A forced change drops the old at once and accepts the breakage (sessions and tickets made with it stop working).

## 7. Where each part stands (2026-11-21)

- **Mesh node keys** (nodus `module_library/source/tools/mesh_crypto.py`, branch `claude/link-bulk-files`): done for
  suite 1. Public blocks carry suite, epoch and lineage; envelopes carry their suite and each holder keeps an
  allowlist (`suites`); `rotate`, `accept`, `root-keygen` and `force` are the operator commands, and the verify row
  applies rotation and force records that arrive on the mesh. Drilled 2026-11-21 on Windows and on the Oracle A1 host
  (rotation with overlap and cutover, a reader's rotation with its retired epoch still opening old data, forced
  revocation at once, replay and forgery refused, a suite removed from the allowlist refused), and end to end across
  processes in `tests/mesh_secure_test.cpp`: after A rotates, B still takes A's canvas as A; after the root revokes
  A's key, B takes nothing more from it.
- **Region keys** (turing `state_broadcast.py`): Ed25519 / X25519, no suite or epoch yet. Next: carry suite ids and
  epochs, move to suite 1 (the same hybrid keys), and accept rotation and force records from the same root.
- **Site root**: to be made by the owner offline; until it exists, forced changes are done by hand (replace keys and
  peer files on every machine).
- **Class S, T, X**: by the steps above, run manually today; the service SQL record of each change is to be added.

## 8. Drills

A rotation is only trusted once it has been practised. Each suite and each key class gets a drill: rotate on a test
mesh (`tests/mesh_secure_test.cpp` style, one process per node), force-revoke a key and check that its holder's data is
refused at once, and run a suite change end to end. A drill that fails blocks the change it was rehearsing.
