#!/usr/bin/env python3
"""Build and validate exact Git commit SHA-1 search jobs.

The message-trailer carrier puts a five-byte raw nonce at offsets 48..52 of
the final padded SHA-1 block. The custom-header carrier uses the same mutable
block layout and records the fixed SHA-1 blocks following it. Candidate
integers use a single, explicit mapping::

    candidate 0xAABBCCDDEE -> nonce bytes AA BB CC DD EE
    W12 = 0xAABBCCDD
    W13[31:24] = 0xEE

The other bytes in W13, SHA-1 padding, object length, and all prefix blocks are
fixed.  This module deliberately contains a small independent SHA-1
implementation in addition to the hashlib oracle; generated-kernel validation
must not depend on the same optimized implementation as the device code.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


MASK32 = 0xFFFF_FFFF
SHA1_IV = (0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0)
SHA1_K = (0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6)
RAW_NONCE_BYTES = 5
RAW_NONCE_BITS = RAW_NONCE_BYTES * 8
RAW_NONCE_BLOCK_OFFSET = 48
MASKED_NONCE_BYTES = 8
MASKED_NONCE_BITS = 40
MASKED_NONCE_BLOCK_OFFSET = 44


def rol32(value: int, amount: int) -> int:
    """Rotate a 32-bit integer left."""

    return ((value << amount) | (value >> (32 - amount))) & MASK32


def git_object_header(object_type: str, payload_length: int) -> bytes:
    """Return Git's exact loose-object hashing header (without compression)."""

    if not object_type or any(ch.isspace() or ch == "\0" for ch in object_type):
        raise ValueError("object type must be a non-empty token")
    if payload_length < 0:
        raise ValueError("payload length cannot be negative")
    try:
        encoded_type = object_type.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("object type must be ASCII") from exc
    return encoded_type + b" " + str(payload_length).encode("ascii") + b"\0"


def serialize_git_commit(payload: bytes) -> bytes:
    """Return the exact byte string Git hashes for a commit payload."""

    raw_payload = bytes(payload)
    return git_object_header("commit", len(raw_payload)) + raw_payload


def sha1_pad(message: bytes) -> bytes:
    """Apply SHA-1 padding to a byte string."""

    bit_length = len(message) * 8
    padded = bytearray(message)
    padded.append(0x80)
    padded.extend(b"\0" * ((56 - len(padded) % 64) % 64))
    padded.extend(bit_length.to_bytes(8, "big"))
    return bytes(padded)


def block_words(block: bytes) -> tuple[int, ...]:
    """Decode one SHA-1 block into its sixteen big-endian input words."""

    if len(block) != 64:
        raise ValueError("a SHA-1 block must contain exactly 64 bytes")
    return tuple(int.from_bytes(block[offset : offset + 4], "big") for offset in range(0, 64, 4))


def expand_schedule(block_or_words: bytes | Sequence[int]) -> tuple[int, ...]:
    """Expand a 64-byte block or sixteen words into SHA-1 W[0..79]."""

    if isinstance(block_or_words, bytes):
        initial = list(block_words(block_or_words))
    else:
        if len(block_or_words) != 16:
            raise ValueError("the initial SHA-1 schedule must contain sixteen words")
        initial = [word & MASK32 for word in block_or_words]
    schedule = initial + [0] * 64
    for index in range(16, 80):
        schedule[index] = rol32(
            schedule[index - 3]
            ^ schedule[index - 8]
            ^ schedule[index - 14]
            ^ schedule[index - 16],
            1,
        )
    return tuple(schedule)


def sha1_working_after_rounds(
    state: Sequence[int], block: bytes, rounds: int
) -> tuple[int, ...]:
    """Return SHA-1 working registers after ``rounds`` compression rounds."""

    if len(state) != 5:
        raise ValueError("SHA-1 state must contain five words")
    if not 0 <= rounds <= 80:
        raise ValueError("SHA-1 round count must be between 0 and 80")
    schedule = expand_schedule(block)
    a, b, c, d, e = (word & MASK32 for word in state)
    for index, word in enumerate(schedule[:rounds]):
        if index < 20:
            function = d ^ (b & (c ^ d))
            constant = SHA1_K[0]
        elif index < 40:
            function = b ^ c ^ d
            constant = SHA1_K[1]
        elif index < 60:
            function = (b & c) | (d & (b | c))
            constant = SHA1_K[2]
        else:
            function = b ^ c ^ d
            constant = SHA1_K[3]
        temp = (rol32(a, 5) + function + e + constant + word) & MASK32
        a, b, c, d, e = temp, a, rol32(b, 30), c, d
    return a, b, c, d, e


def sha1_compress_working(state: Sequence[int], block: bytes) -> tuple[int, ...]:
    """Return working registers (a..e) after 80 rounds, before feed-forward."""

    return sha1_working_after_rounds(state, block, 80)


def sha1_compress(state: Sequence[int], block: bytes) -> tuple[int, ...]:
    """Compress one block and return the feed-forward SHA-1 state."""

    working = sha1_compress_working(state, block)
    return tuple(((left & MASK32) + right) & MASK32 for left, right in zip(state, working))


def sha1_digest_independent(message: bytes) -> bytes:
    """Compute SHA-1 without hashlib, for differential validation."""

    state: tuple[int, ...] = SHA1_IV
    padded = sha1_pad(message)
    for offset in range(0, len(padded), 64):
        state = sha1_compress(state, padded[offset : offset + 64])
    return b"".join(word.to_bytes(4, "big") for word in state)


def sha1_prestate(padded_message: bytes, block_index: int) -> tuple[int, ...]:
    """Hash complete blocks before ``block_index`` and return their state."""

    if len(padded_message) % 64:
        raise ValueError("padded message length must be a multiple of 64")
    block_count = len(padded_message) // 64
    if not 0 <= block_index <= block_count:
        raise ValueError("block index is outside the padded message")
    state: tuple[int, ...] = SHA1_IV
    for index in range(block_index):
        start = index * 64
        state = sha1_compress(state, padded_message[start : start + 64])
    return state


@dataclasses.dataclass(frozen=True)
class TargetPrefix:
    """A most-significant-bit SHA-1 target prefix.

    ``value`` stores exactly ``bits`` significant bits, right-aligned as an
    integer.  ``aligned_value`` moves them to the top of a 160-bit digest.
    """

    bits: int
    value: int

    def __post_init__(self) -> None:
        if not 1 <= self.bits <= 160:
            raise ValueError("target width must be between 1 and 160 bits")
        if not 0 <= self.value < (1 << self.bits):
            raise ValueError("target value does not fit its bit width")

    @classmethod
    def from_hex(cls, text: str, bits: int | None = None) -> "TargetPrefix":
        """Parse hexadecimal prefix digits, optionally selecting their first bits.

        With ``bits=5``, ``a8`` means the first five bits of the written bit
        string (``10101``), not the low five bits of the integer.
        """

        digits = text.strip().lower()
        if digits.startswith("0x"):
            digits = digits[2:]
        if not digits or len(digits) > 40 or any(ch not in "0123456789abcdef" for ch in digits):
            raise ValueError("target must contain 1..40 hexadecimal digits")
        supplied_bits = len(digits) * 4
        selected_bits = supplied_bits if bits is None else bits
        if not 1 <= selected_bits <= supplied_bits or selected_bits > 160:
            raise ValueError("target bits must select 1..160 supplied leading bits")
        value = int(digits, 16) >> (supplied_bits - selected_bits)
        return cls(selected_bits, value)

    @property
    def aligned_value(self) -> int:
        return self.value << (160 - self.bits)

    @property
    def aligned_bytes(self) -> bytes:
        return self.aligned_value.to_bytes(20, "big")

    @property
    def words(self) -> tuple[int, ...]:
        raw = self.aligned_bytes
        return tuple(int.from_bytes(raw[offset : offset + 4], "big") for offset in range(0, 20, 4))

    @property
    def masks(self) -> tuple[int, ...]:
        remaining = self.bits
        masks: list[int] = []
        for _ in range(5):
            used = min(32, max(0, remaining))
            masks.append(0 if used == 0 else (MASK32 << (32 - used)) & MASK32)
            remaining -= used
        return tuple(masks)

    def matches(self, digest: bytes) -> bool:
        if len(digest) != 20:
            raise ValueError("a SHA-1 digest must contain exactly 20 bytes")
        return int.from_bytes(digest, "big") >> (160 - self.bits) == self.value

    def display_hex(self) -> str:
        """Return a canonical, zero-padded representation of selected bits."""

        digits = (self.bits + 3) // 4
        shifted = self.value << (digits * 4 - self.bits)
        return f"{shifted:0{digits}x}"


def h0_gate_parameters(prestate_h0: int, target: TargetPrefix) -> tuple[int, int]:
    """Return modular interval ``base, span`` for the final working ``a``.

    A kernel can reject when ``uint32_t(a - base) >= span``.  For prefixes of
    32 or more bits, ``span`` is one and this reduces to exact equality.
    """

    used = min(32, target.bits)
    span = 1 << (32 - used)
    digest_base = target.words[0] & ((MASK32 << (32 - used)) & MASK32)
    return (digest_base - prestate_h0) & MASK32, span


def working_state_matches_prefix(
    working: Sequence[int], prestate: Sequence[int], target: TargetPrefix
) -> bool:
    """Apply the kernel-style prefix gate to final-block working registers."""

    if len(working) != 5 or len(prestate) != 5:
        raise ValueError("working state and prestate must each contain five words")
    base, span = h0_gate_parameters(prestate[0], target)
    if ((working[0] - base) & MASK32) >= span:
        return False
    digest_words = tuple((working[index] + prestate[index]) & MASK32 for index in range(5))
    return all((word & mask) == (wanted & mask) for word, wanted, mask in zip(digest_words, target.words, target.masks))


def candidate_nonce(candidate: int) -> bytes:
    """Map a 40-bit candidate integer to its five big-endian nonce bytes."""

    if not 0 <= candidate < (1 << RAW_NONCE_BITS):
        raise ValueError("candidate must be an unsigned 40-bit integer")
    return candidate.to_bytes(RAW_NONCE_BYTES, "big")


def candidate_words(candidate: int) -> tuple[int, int]:
    """Return direct candidate contributions to W12 and W13."""

    candidate_nonce(candidate)  # range check
    return candidate >> 8, (candidate & 0xFF) << 24


def nonce_is_log_safe(nonce: bytes) -> bool:
    """Return whether raw nonce bytes avoid Git's troublesome NUL byte."""

    if len(nonce) != RAW_NONCE_BYTES:
        raise ValueError("raw nonce must contain exactly five bytes")
    return b"\0" not in nonce


def nonce_is_header_safe(nonce: bytes) -> bool:
    """Return whether raw nonce bytes can remain on one commit-header line."""

    if len(nonce) != RAW_NONCE_BYTES:
        raise ValueError("raw nonce must contain exactly five bytes")
    return b"\0" not in nonce and b"\n" not in nonce


def masked_candidate_nonce(candidate: int) -> bytes:
    """Map 40 candidate bits to eight printable 0x20..0x3f bytes."""

    if not 0 <= candidate < (1 << MASKED_NONCE_BITS):
        raise ValueError("candidate must be an unsigned 40-bit integer")
    return bytes(0x20 | ((candidate >> shift) & 0x1F) for shift in range(35, -1, -5))


@dataclasses.dataclass(frozen=True)
class RawTailJob:
    """Complete host-side description of a final-one-block raw nonce job."""

    source_payload: bytes
    payload_template: bytes
    object_template: bytes
    nonce_payload_offset: int
    nonce_object_offset: int
    mutable_block: int
    filler_bytes: int
    source_newline_appended: bool
    final_block_template: bytes
    prestate: tuple[int, ...]
    target: TargetPrefix
    label: bytes
    filler_byte: bytes
    placeholder: bytes

    def __post_init__(self) -> None:
        if len(self.final_block_template) != 64:
            raise ValueError("final block template must be exactly one block")
        if self.nonce_object_offset % 64 != RAW_NONCE_BLOCK_OFFSET:
            raise ValueError("raw nonce must begin at final-block offset 48")
        if len(self.prestate) != 5:
            raise ValueError("prestate must contain five words")

    @property
    def base_words(self) -> tuple[int, ...]:
        return block_words(self.final_block_template)

    @property
    def padded_blocks(self) -> int:
        return self.mutable_block + 1

    def materialize_payload(self, candidate: int) -> bytes:
        result = bytearray(self.payload_template)
        result[self.nonce_payload_offset : self.nonce_payload_offset + RAW_NONCE_BYTES] = candidate_nonce(candidate)
        return bytes(result)

    def materialize_object(self, candidate: int) -> bytes:
        return serialize_git_commit(self.materialize_payload(candidate))

    def materialize_final_block(self, candidate: int) -> bytes:
        result = bytearray(self.final_block_template)
        result[RAW_NONCE_BLOCK_OFFSET : RAW_NONCE_BLOCK_OFFSET + RAW_NONCE_BYTES] = candidate_nonce(candidate)
        return bytes(result)

    def digest(self, candidate: int) -> bytes:
        """Authoritative candidate digest, computed with hashlib."""

        return hashlib.sha1(self.materialize_object(candidate)).digest()

    def digest_from_prestate(self, candidate: int) -> bytes:
        """Independent candidate digest using fixed prestate plus final block."""

        state = sha1_compress(self.prestate, self.materialize_final_block(candidate))
        return b"".join(word.to_bytes(4, "big") for word in state)

    def matches_target(self, candidate: int) -> bool:
        return self.target.matches(self.digest(candidate))

    def verify_candidate(self, candidate: int, *, require_log_safe: bool = True) -> bytes:
        """Independently validate a candidate returned by a search kernel.

        The full serialized object is hashed with hashlib, then cross-checked
        against the fixed-prestate compression path.  A returned digest is
        therefore safe to compare with Git; policy or target failures raise a
        ``ValueError`` instead of silently accepting a device false positive.
        """

        nonce = candidate_nonce(candidate)
        if require_log_safe and not nonce_is_log_safe(nonce):
            raise ValueError("candidate nonce contains NUL and violates the winner policy")
        oracle = self.digest(candidate)
        independent = self.digest_from_prestate(candidate)
        if oracle != independent:
            raise RuntimeError("hashlib and independent fixed-prestate SHA-1 calculations disagree")
        if not self.target.matches(oracle):
            raise ValueError("candidate digest does not match the requested prefix")
        return oracle

    def manifest(self) -> dict[str, object]:
        base, span = h0_gate_parameters(self.prestate[0], self.target)
        object_header_bytes = len(self.object_template) - len(self.payload_template)
        legal_fraction = (255 / 256) ** RAW_NONCE_BYTES
        return {
            "schema": "git-sha1-raw-tail-job-v1",
            "hash": "sha1",
            "object_type": "commit",
            "layout": "raw-binary-tail-4+1",
            "endianness": "sha1-words-and-candidate-are-big-endian",
            "source_payload_bytes": len(self.source_payload),
            "source_newline_appended": self.source_newline_appended,
            "payload_bytes": len(self.payload_template),
            "git_object_header_bytes": object_header_bytes,
            "git_object_bytes": len(self.object_template),
            "padded_bytes": self.padded_blocks * 64,
            "padded_blocks": self.padded_blocks,
            "fixed_prefix_blocks": self.mutable_block,
            "mutable_block": self.mutable_block,
            "candidate_dependent_blocks": 1,
            "data_bytes_in_final_block": len(self.object_template) % 64,
            "filler_bytes": self.filler_bytes,
            "label_hex": self.label.hex(),
            "filler_byte_hex": self.filler_byte.hex(),
            "placeholder_hex": self.placeholder.hex(),
            "nonce": {
                "bytes": RAW_NONCE_BYTES,
                "bits": RAW_NONCE_BITS,
                "payload_offset": self.nonce_payload_offset,
                "object_offset": self.nonce_object_offset,
                "block_offset": RAW_NONCE_BLOCK_OFFSET,
                "first_word": 12,
                "last_word": 13,
                "mapping": "candidate.to_bytes(5, 'big')",
                "w12": "candidate[39:8]",
                "w13_mask": "0xff000000",
                "w13": "candidate[7:0] << 24",
                "candidate_min": 0,
                "candidate_max": (1 << RAW_NONCE_BITS) - 1,
                "winner_policy": "reject nonce containing NUL before writing commit",
                "log_safe_fraction": legal_fraction,
                "log_safe_search_bits": math.log2((255**RAW_NONCE_BYTES)),
            },
            "target": {
                "bits": self.target.bits,
                "hex": self.target.display_hex(),
                "aligned_words": [f"{word:08x}" for word in self.target.words],
                "word_masks": [f"{word:08x}" for word in self.target.masks],
                "working_a_gate_base": f"{base:08x}",
                "working_a_gate_span": span,
            },
            "prestate": [f"{word:08x}" for word in self.prestate],
            "base_words": [f"{word:08x}" for word in self.base_words],
            "final_block_template_hex": self.final_block_template.hex(),
            "template_object_sha1": hashlib.sha1(self.object_template).hexdigest(),
        }


@dataclasses.dataclass(frozen=True)
class HeaderJob:
    """A commit job whose nonce is carried by a custom header.

    The nonce retains the production kernel's W12/W13 layout in its mutable
    block.  Complete padded blocks after that block are fixed suffix blocks
    which must be compressed before checking the target digest.
    """

    source_payload: bytes
    payload_template: bytes
    object_template: bytes
    nonce_payload_offset: int
    nonce_object_offset: int
    mutable_block: int
    filler_bytes: int
    mutable_block_template: bytes
    suffix_blocks: tuple[bytes, ...]
    prestate: tuple[int, ...]
    target: TargetPrefix
    header_name: bytes
    value_prefix: bytes
    filler_byte: bytes
    placeholder: bytes

    def __post_init__(self) -> None:
        if len(self.mutable_block_template) != 64:
            raise ValueError("mutable block template must be exactly one block")
        if self.nonce_object_offset % 64 != RAW_NONCE_BLOCK_OFFSET:
            raise ValueError("raw nonce must begin at mutable-block offset 48")
        if len(self.prestate) != 5:
            raise ValueError("prestate must contain five words")
        if any(len(block) != 64 for block in self.suffix_blocks):
            raise ValueError("header-job suffix blocks must be complete SHA-1 blocks")

    @property
    def base_words(self) -> tuple[int, ...]:
        return block_words(self.mutable_block_template)

    @property
    def suffix_words(self) -> tuple[tuple[int, ...], ...]:
        return tuple(block_words(block) for block in self.suffix_blocks)

    def materialize_payload(self, candidate: int) -> bytes:
        result = bytearray(self.payload_template)
        result[self.nonce_payload_offset : self.nonce_payload_offset + RAW_NONCE_BYTES] = candidate_nonce(candidate)
        return bytes(result)

    def materialize_object(self, candidate: int) -> bytes:
        return serialize_git_commit(self.materialize_payload(candidate))

    def materialize_mutable_block(self, candidate: int) -> bytes:
        result = bytearray(self.mutable_block_template)
        result[RAW_NONCE_BLOCK_OFFSET : RAW_NONCE_BLOCK_OFFSET + RAW_NONCE_BYTES] = candidate_nonce(candidate)
        return bytes(result)

    def digest(self, candidate: int) -> bytes:
        return hashlib.sha1(self.materialize_object(candidate)).digest()

    def digest_from_prestate(self, candidate: int) -> bytes:
        state = sha1_compress(self.prestate, self.materialize_mutable_block(candidate))
        for block in self.suffix_blocks:
            state = sha1_compress(state, block)
        return b"".join(word.to_bytes(4, "big") for word in state)

    def matches_target(self, candidate: int) -> bool:
        return self.target.matches(self.digest(candidate))

    def verify_candidate(self, candidate: int, *, require_header_safe: bool = True) -> bytes:
        nonce = candidate_nonce(candidate)
        if require_header_safe and not nonce_is_header_safe(nonce):
            raise ValueError("candidate nonce contains NUL or newline and is not header-safe")
        oracle = self.digest(candidate)
        independent = self.digest_from_prestate(candidate)
        if oracle != independent:
            raise RuntimeError("hashlib and independent suffix-job SHA-1 calculations disagree")
        if not self.target.matches(oracle):
            raise ValueError("candidate digest does not match the requested prefix")
        return oracle

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "git-sha1-header-job-v1",
            "hash": "sha1",
            "object_type": "commit",
            "layout": "custom-header-raw-4+1-with-fixed-suffix",
            "source_payload_bytes": len(self.source_payload),
            "payload_bytes": len(self.payload_template),
            "git_object_bytes": len(self.object_template),
            "padded_blocks": self.mutable_block + 1 + len(self.suffix_blocks),
            "fixed_prefix_blocks": self.mutable_block,
            "mutable_block": self.mutable_block,
            "fixed_suffix_blocks": len(self.suffix_blocks),
            "filler_bytes": self.filler_bytes,
            "header_name": self.header_name.decode("ascii"),
            "value_prefix_hex": self.value_prefix.hex(),
            "nonce": {
                "bytes": RAW_NONCE_BYTES,
                "bits": RAW_NONCE_BITS,
                "payload_offset": self.nonce_payload_offset,
                "object_offset": self.nonce_object_offset,
                "block_offset": RAW_NONCE_BLOCK_OFFSET,
                "mapping": "candidate.to_bytes(5, 'big')",
                "winner_policy": "reject nonce containing NUL or newline",
            },
            "target": {
                "bits": self.target.bits,
                "hex": self.target.display_hex(),
                "aligned_words": [f"{word:08x}" for word in self.target.words],
                "word_masks": [f"{word:08x}" for word in self.target.masks],
            },
            "prestate": [f"{word:08x}" for word in self.prestate],
            "base_words": [f"{word:08x}" for word in self.base_words],
            "suffix_words": [
                [f"{word:08x}" for word in words] for words in self.suffix_words
            ],
            "template_object_sha1": hashlib.sha1(self.object_template).hexdigest(),
        }


@dataclasses.dataclass(frozen=True)
class MaskedHeaderJob:
    """A custom-header job with eight printable bytes carrying 40 bits."""

    source_payload: bytes
    payload_template: bytes
    object_template: bytes
    nonce_payload_offset: int
    nonce_object_offset: int
    mutable_block: int
    filler_bytes: int
    mutable_block_template: bytes
    suffix_blocks: tuple[bytes, ...]
    prestate: tuple[int, ...]
    target: TargetPrefix
    header_name: bytes
    value_prefix: bytes
    filler_byte: bytes
    placeholder: bytes

    def __post_init__(self) -> None:
        if len(self.mutable_block_template) != 64:
            raise ValueError("mutable block template must be exactly one block")
        if self.nonce_object_offset % 64 != MASKED_NONCE_BLOCK_OFFSET:
            raise ValueError("masked nonce must begin at mutable-block offset 44")
        if len(self.prestate) != 5:
            raise ValueError("prestate must contain five words")
        if any(len(block) != 64 for block in self.suffix_blocks):
            raise ValueError("masked-header suffix blocks must be complete SHA-1 blocks")

    @property
    def base_words(self) -> tuple[int, ...]:
        return block_words(self.mutable_block_template)

    @property
    def suffix_words(self) -> tuple[tuple[int, ...], ...]:
        return tuple(block_words(block) for block in self.suffix_blocks)

    def materialize_payload(self, candidate: int) -> bytes:
        result = bytearray(self.payload_template)
        result[self.nonce_payload_offset : self.nonce_payload_offset + MASKED_NONCE_BYTES] = masked_candidate_nonce(candidate)
        return bytes(result)

    def materialize_object(self, candidate: int) -> bytes:
        return serialize_git_commit(self.materialize_payload(candidate))

    def materialize_mutable_block(self, candidate: int) -> bytes:
        result = bytearray(self.mutable_block_template)
        result[
            MASKED_NONCE_BLOCK_OFFSET : MASKED_NONCE_BLOCK_OFFSET + MASKED_NONCE_BYTES
        ] = masked_candidate_nonce(candidate)
        return bytes(result)

    def digest(self, candidate: int) -> bytes:
        return hashlib.sha1(self.materialize_object(candidate)).digest()

    def digest_from_prestate(self, candidate: int) -> bytes:
        state = sha1_compress(self.prestate, self.materialize_mutable_block(candidate))
        for block in self.suffix_blocks:
            state = sha1_compress(state, block)
        return b"".join(word.to_bytes(4, "big") for word in state)

    def verify_candidate(self, candidate: int) -> bytes:
        oracle = self.digest(candidate)
        independent = self.digest_from_prestate(candidate)
        if oracle != independent:
            raise RuntimeError("hashlib and independent masked-header SHA-1 calculations disagree")
        if not self.target.matches(oracle):
            raise ValueError("candidate digest does not match the requested prefix")
        return oracle

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "git-sha1-masked-header-job-v1",
            "hash": "sha1",
            "object_type": "commit",
            "layout": "custom-header-printable-mask-8x5",
            "source_payload_bytes": len(self.source_payload),
            "payload_bytes": len(self.payload_template),
            "git_object_bytes": len(self.object_template),
            "padded_blocks": self.mutable_block + 1 + len(self.suffix_blocks),
            "fixed_prefix_blocks": self.mutable_block,
            "mutable_block": self.mutable_block,
            "fixed_suffix_blocks": len(self.suffix_blocks),
            "filler_bytes": self.filler_bytes,
            "header_name": self.header_name.decode("ascii"),
            "value_prefix_hex": self.value_prefix.hex(),
            "nonce": {
                "bytes": MASKED_NONCE_BYTES,
                "candidate_bits": MASKED_NONCE_BITS,
                "payload_offset": self.nonce_payload_offset,
                "object_offset": self.nonce_object_offset,
                "block_offset": MASKED_NONCE_BLOCK_OFFSET,
                "first_word": 11,
                "last_word": 12,
                "mapping": "byte[i] = 0x20 | candidate[39-5*i:35-5*i]",
                "alphabet": "0x20..0x3f",
            },
            "target": {
                "bits": self.target.bits,
                "hex": self.target.display_hex(),
                "aligned_words": [f"{word:08x}" for word in self.target.words],
                "word_masks": [f"{word:08x}" for word in self.target.masks],
            },
            "prestate": [f"{word:08x}" for word in self.prestate],
            "base_words": [f"{word:08x}" for word in self.base_words],
            "suffix_words": [
                [f"{word:08x}" for word in words] for words in self.suffix_words
            ],
            "template_object_sha1": hashlib.sha1(self.object_template).hexdigest(),
        }


def build_raw_tail_job(
    payload: bytes,
    target: TargetPrefix,
    *,
    label: bytes = b"X: ",
    filler_byte: bytes = b" ",
    placeholder: bytes = b"PPPPP",
    max_filler: int = 4096,
) -> RawTailJob:
    """Append and align a raw 4+1-byte nonce in the final padded block."""

    source = bytes(payload)
    if not label or b"\0" in label or b"\n" in label:
        raise ValueError("label must be non-empty and cannot contain NUL or newline")
    if len(filler_byte) != 1 or filler_byte in (b"\0", b"\n"):
        raise ValueError("filler byte must be one non-NUL, non-newline byte")
    if len(placeholder) != RAW_NONCE_BYTES:
        raise ValueError("placeholder must contain exactly five bytes")
    if max_filler <= 0:
        raise ValueError("max_filler must be positive")

    source_newline_appended = not source.endswith(b"\n")
    base = source + (b"\n" if source_newline_appended else b"")
    for filler_bytes in range(max_filler):
        nonce_payload_offset = len(base) + len(label) + filler_bytes
        candidate_payload = base + label + filler_byte * filler_bytes + placeholder + b"\n"
        candidate_object = serialize_git_commit(candidate_payload)
        header_bytes = len(candidate_object) - len(candidate_payload)
        nonce_object_offset = header_bytes + nonce_payload_offset
        if nonce_object_offset % 64 != RAW_NONCE_BLOCK_OFFSET:
            continue
        padded = sha1_pad(candidate_object)
        mutable_block = nonce_object_offset // 64
        if mutable_block != len(padded) // 64 - 1:
            continue
        if len(candidate_object) % 64 != 54:
            continue

        final_start = mutable_block * 64
        final_template = bytearray(padded[final_start : final_start + 64])
        final_template[RAW_NONCE_BLOCK_OFFSET : RAW_NONCE_BLOCK_OFFSET + RAW_NONCE_BYTES] = b"\0" * RAW_NONCE_BYTES
        prestate = sha1_prestate(padded, mutable_block)
        return RawTailJob(
            source_payload=source,
            payload_template=candidate_payload,
            object_template=candidate_object,
            nonce_payload_offset=nonce_payload_offset,
            nonce_object_offset=nonce_object_offset,
            mutable_block=mutable_block,
            filler_bytes=filler_bytes,
            source_newline_appended=source_newline_appended,
            final_block_template=bytes(final_template),
            prestate=prestate,
            target=target,
            label=label,
            filler_byte=filler_byte,
            placeholder=placeholder,
        )
    raise RuntimeError(f"could not find final-block raw nonce alignment using fewer than {max_filler} filler bytes")


def build_header_job(
    payload: bytes,
    target: TargetPrefix,
    *,
    header_name: bytes = b"x",
    value_prefix: bytes = b"",
    filler_byte: bytes = b" ",
    placeholder: bytes = b"PPPPP",
    max_filler: int = 4096,
) -> HeaderJob:
    """Insert and align a custom commit header for fixed-suffix searching."""

    source = bytes(payload)
    separator = source.find(b"\n\n")
    if separator < 0:
        raise ValueError("commit payload must contain a blank line between headers and message")
    if not header_name or any(byte <= 0x20 or byte >= 0x7F for byte in header_name):
        raise ValueError("header name must be a non-empty printable ASCII token")
    if b"\0" in value_prefix or b"\n" in value_prefix:
        raise ValueError("header value prefix cannot contain NUL or newline")
    if len(filler_byte) != 1 or filler_byte in (b"\0", b"\n"):
        raise ValueError("filler byte must be one non-NUL, non-newline byte")
    if len(placeholder) != RAW_NONCE_BYTES:
        raise ValueError("placeholder must contain exactly five bytes")
    if max_filler <= 0:
        raise ValueError("max_filler must be positive")

    headers = source[:separator]
    message = source[separator + 2 :]
    header_prefix = headers + b"\n" + header_name + b" " + value_prefix
    for filler_bytes in range(max_filler):
        nonce_payload_offset = len(header_prefix) + filler_bytes
        candidate_payload = (
            header_prefix
            + filler_byte * filler_bytes
            + placeholder
            + b"\n\n"
            + message
        )
        candidate_object = serialize_git_commit(candidate_payload)
        object_header_bytes = len(candidate_object) - len(candidate_payload)
        nonce_object_offset = object_header_bytes + nonce_payload_offset
        if nonce_object_offset % 64 != RAW_NONCE_BLOCK_OFFSET:
            continue

        padded = sha1_pad(candidate_object)
        mutable_block = nonce_object_offset // 64
        suffix_start = (mutable_block + 1) * 64
        mutable_start = mutable_block * 64
        mutable_template = bytearray(padded[mutable_start:suffix_start])
        mutable_template[RAW_NONCE_BLOCK_OFFSET : RAW_NONCE_BLOCK_OFFSET + RAW_NONCE_BYTES] = b"\0" * RAW_NONCE_BYTES
        suffix_blocks = tuple(
            padded[offset : offset + 64] for offset in range(suffix_start, len(padded), 64)
        )
        return HeaderJob(
            source_payload=source,
            payload_template=candidate_payload,
            object_template=candidate_object,
            nonce_payload_offset=nonce_payload_offset,
            nonce_object_offset=nonce_object_offset,
            mutable_block=mutable_block,
            filler_bytes=filler_bytes,
            mutable_block_template=bytes(mutable_template),
            suffix_blocks=suffix_blocks,
            prestate=sha1_prestate(padded, mutable_block),
            target=target,
            header_name=header_name,
            value_prefix=value_prefix,
            filler_byte=filler_byte,
            placeholder=placeholder,
        )
    raise RuntimeError(f"could not align custom-header nonce using fewer than {max_filler} filler bytes")


def build_masked_header_job(
    payload: bytes,
    target: TargetPrefix,
    *,
    header_name: bytes = b"x",
    value_prefix: bytes = b"",
    filler_byte: bytes = b" ",
    placeholder: bytes = b"PPPPPPPP",
    max_filler: int = 4096,
) -> MaskedHeaderJob:
    """Insert an aligned eight-character, 40-bit printable custom header."""

    source = bytes(payload)
    separator = source.find(b"\n\n")
    if separator < 0:
        raise ValueError("commit payload must contain a blank line between headers and message")
    if not header_name or any(byte <= 0x20 or byte >= 0x7F for byte in header_name):
        raise ValueError("header name must be a non-empty printable ASCII token")
    if b"\0" in value_prefix or b"\n" in value_prefix:
        raise ValueError("header value prefix cannot contain NUL or newline")
    if len(filler_byte) != 1 or filler_byte in (b"\0", b"\n"):
        raise ValueError("filler byte must be one non-NUL, non-newline byte")
    if len(placeholder) != MASKED_NONCE_BYTES:
        raise ValueError("placeholder must contain exactly eight bytes")
    if max_filler <= 0:
        raise ValueError("max_filler must be positive")

    headers = source[:separator]
    message = source[separator + 2 :]
    header_prefix = headers + b"\n" + header_name + b" " + value_prefix
    for filler_bytes in range(max_filler):
        nonce_payload_offset = len(header_prefix) + filler_bytes
        candidate_payload = (
            header_prefix
            + filler_byte * filler_bytes
            + placeholder
            + b"\n\n"
            + message
        )
        candidate_object = serialize_git_commit(candidate_payload)
        object_header_bytes = len(candidate_object) - len(candidate_payload)
        nonce_object_offset = object_header_bytes + nonce_payload_offset
        if nonce_object_offset % 64 != MASKED_NONCE_BLOCK_OFFSET:
            continue

        padded = sha1_pad(candidate_object)
        mutable_block = nonce_object_offset // 64
        mutable_start = mutable_block * 64
        suffix_start = mutable_start + 64
        if suffix_start > len(padded):
            continue
        mutable_template = bytearray(padded[mutable_start:suffix_start])
        mutable_template[
            MASKED_NONCE_BLOCK_OFFSET : MASKED_NONCE_BLOCK_OFFSET + MASKED_NONCE_BYTES
        ] = b"\0" * MASKED_NONCE_BYTES
        suffix_blocks = tuple(
            padded[offset : offset + 64] for offset in range(suffix_start, len(padded), 64)
        )
        return MaskedHeaderJob(
            source_payload=source,
            payload_template=candidate_payload,
            object_template=candidate_object,
            nonce_payload_offset=nonce_payload_offset,
            nonce_object_offset=nonce_object_offset,
            mutable_block=mutable_block,
            filler_bytes=filler_bytes,
            mutable_block_template=bytes(mutable_template),
            suffix_blocks=suffix_blocks,
            prestate=sha1_prestate(padded, mutable_block),
            target=target,
            header_name=header_name,
            value_prefix=value_prefix,
            filler_byte=filler_byte,
            placeholder=placeholder,
        )
    raise RuntimeError(
        f"could not align masked custom-header nonce using fewer than {max_filler} filler bytes"
    )


def _c_u32_array(name: str, words: Iterable[int]) -> str:
    values = tuple(words)
    encoded = ", ".join(f"0x{word:08x}u" for word in values)
    return f"static constexpr uint32_t {name}[{len(values)}] = {{{encoded}}};\n"


def render_cuda_header(job: RawTailJob) -> str:
    """Render the compact constants contract consumed by CUDA experiments."""

    base, span = h0_gate_parameters(job.prestate[0], job.target)
    pre12 = sha1_working_after_rounds(job.prestate, job.final_block_template, 12)
    lines = [
        "// Generated by tools/git_sha1_job.py; do not edit.\n",
        "#pragma once\n",
        "#include <stdint.h>\n",
        "\n",
        _c_u32_array("JOB_HIN", job.prestate),
        _c_u32_array("JOB_PRE12", pre12),
        _c_u32_array("JOB_BASE16", job.base_words),
        _c_u32_array("JOB_TARGET_WORDS", job.target.words),
        _c_u32_array("JOB_TARGET_MASKS", job.target.masks),
        f"static constexpr uint32_t JOB_TARGET_BITS = {job.target.bits}u;\n",
        f"static constexpr uint32_t JOB_H0_GATE_BASE = 0x{base:08x}u;\n",
        f"static constexpr uint32_t JOB_H0_GATE_SPAN = {span}u;\n",
        f"static constexpr uint32_t JOB_MUTABLE_BLOCK = {job.mutable_block}u;\n",
        f"static constexpr uint64_t JOB_NONCE_OBJECT_OFFSET = {job.nonce_object_offset}ull;\n",
        f"static constexpr uint64_t JOB_NONCE_PAYLOAD_OFFSET = {job.nonce_payload_offset}ull;\n",
        "static constexpr uint32_t JOB_NONCE_BLOCK_OFFSET = 48u;\n",
        "static constexpr uint32_t JOB_NONCE_BYTES = 5u;\n",
        "// Mapping: W12 = candidate >> 8; W13 |= (candidate & 0xff) << 24.\n",
    ]
    return "".join(lines)


def write_job(job: RawTailJob, output_dir: Path) -> None:
    """Write binary templates, manifest, and CUDA constants atomically enough for local use."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "payload-template.bin").write_bytes(job.payload_template)
    (output_dir / "object-template.bin").write_bytes(job.object_template)
    (output_dir / "final-block-template.bin").write_bytes(job.final_block_template)
    (output_dir / "job.json").write_text(json.dumps(job.manifest(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "job_constants.cuh").write_text(render_cuda_header(job), encoding="utf-8")


def write_header_job(job: HeaderJob, output_dir: Path) -> None:
    """Write custom-header templates and fixed suffix blocks."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "payload-template.bin").write_bytes(job.payload_template)
    (output_dir / "object-template.bin").write_bytes(job.object_template)
    (output_dir / "mutable-block-template.bin").write_bytes(job.mutable_block_template)
    (output_dir / "suffix-blocks.bin").write_bytes(b"".join(job.suffix_blocks))
    (output_dir / "job.json").write_text(json.dumps(job.manifest(), indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="raw commit payload (not the Git object header)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output job directory")
    parser.add_argument("--target", required=True, help="1..40 leading SHA-1 hexadecimal digits")
    parser.add_argument(
        "--target-bits",
        type=int,
        help="use this many leading bits of --target (default: all supplied nibbles)",
    )
    parser.add_argument(
        "--carrier",
        choices=("trailer", "header"),
        default="trailer",
        help="location used to carry the five candidate bytes",
    )
    parser.add_argument("--label", default="X: ", help="ASCII message-trailer label")
    parser.add_argument("--header-name", default="x", help="ASCII custom-header name")
    parser.add_argument(
        "--value-prefix",
        default="",
        help="fixed ASCII bytes before a custom-header candidate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target = TargetPrefix.from_hex(args.target, args.target_bits)
    try:
        label = args.label.encode("ascii")
        header_name = args.header_name.encode("ascii")
        value_prefix = args.value_prefix.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SystemExit("--label, --header-name, and --value-prefix must be ASCII") from exc
    if args.carrier == "header":
        job = build_header_job(
            args.payload.read_bytes(),
            target,
            header_name=header_name,
            value_prefix=value_prefix,
        )
        write_header_job(job, args.output)
    else:
        job = build_raw_tail_job(args.payload.read_bytes(), target, label=label)
        write_job(job, args.output)
    print(json.dumps(job.manifest(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
