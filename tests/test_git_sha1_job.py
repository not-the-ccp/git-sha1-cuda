from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest

from tools.git_sha1_job import (
    MASK32,
    RAW_NONCE_BLOCK_OFFSET,
    MASKED_NONCE_BLOCK_OFFSET,
    TargetPrefix,
    block_words,
    build_header_job,
    build_masked_header_job,
    build_raw_tail_job,
    candidate_nonce,
    candidate_words,
    h0_gate_parameters,
    nonce_is_header_safe,
    nonce_is_log_safe,
    masked_candidate_nonce,
    render_cuda_header,
    serialize_git_commit,
    sha1_compress,
    sha1_compress_working,
    sha1_digest_independent,
    sha1_pad,
    sha1_working_after_rounds,
    working_state_matches_prefix,
    write_header_job,
    write_job,
)


class GitSerializationTests(unittest.TestCase):
    def test_exact_serialization(self) -> None:
        payload = b"tree 0123456789012345678901234567890123456789\n\nmessage\n"
        expected = b"commit 55\0" + payload
        self.assertEqual(serialize_git_commit(payload), expected)

    def test_hashlib_matches_git_hash_object(self) -> None:
        rng = random.Random(0x6174_6F62)
        for size in (0, 1, 9, 10, 55, 56, 63, 64, 99, 100, 255):
            payload = rng.randbytes(size)
            expected = hashlib.sha1(serialize_git_commit(payload)).hexdigest()
            result = subprocess.run(
                ["git", "hash-object", "-t", "commit", "--stdin", "--literally"],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            self.assertEqual(result.stdout.strip().decode("ascii"), expected)


class IndependentSha1Tests(unittest.TestCase):
    def test_padding_boundaries_and_random_messages(self) -> None:
        rng = random.Random(0x5A1_0A11)
        sizes = list(range(0, 80)) + [111, 112, 119, 120, 127, 128, 255, 1024]
        sizes.extend(rng.randrange(0, 4096) for _ in range(100))
        for size in sizes:
            message = rng.randbytes(size)
            with self.subTest(size=size):
                self.assertEqual(sha1_digest_independent(message), hashlib.sha1(message).digest())
                padded = sha1_pad(message)
                self.assertEqual(len(padded) % 64, 0)
                self.assertEqual(int.from_bytes(padded[-8:], "big"), size * 8)

    def test_partial_round_state_reaches_full_compression(self) -> None:
        rng = random.Random(0x12_80_5A1)
        for _ in range(40):
            state = tuple(rng.getrandbits(32) for _ in range(5))
            block = rng.randbytes(64)
            self.assertEqual(sha1_working_after_rounds(state, block, 0), state)
            self.assertEqual(
                sha1_working_after_rounds(state, block, 80),
                sha1_compress_working(state, block),
            )


class TargetPrefixTests(unittest.TestCase):
    def test_partial_hex_selects_leading_bits(self) -> None:
        target = TargetPrefix.from_hex("a8", 5)
        self.assertEqual(target.value, 0b10101)
        self.assertEqual(target.display_hex(), "a8")
        self.assertEqual(target.words[0], 0xA800_0000)
        self.assertEqual(target.masks[0], 0xF800_0000)

    def test_prefix_matching_and_working_gate_all_widths(self) -> None:
        rng = random.Random(0xC0FF_EE51)
        for _ in range(60):
            prestate = tuple(rng.getrandbits(32) for _ in range(5))
            block = rng.randbytes(64)
            working = sha1_compress_working(prestate, block)
            digest_words = sha1_compress(prestate, block)
            digest = b"".join(word.to_bytes(4, "big") for word in digest_words)
            digest_int = int.from_bytes(digest, "big")
            for bits in range(1, 161):
                value = digest_int >> (160 - bits)
                target = TargetPrefix(bits, value)
                self.assertTrue(target.matches(digest))
                self.assertTrue(working_state_matches_prefix(working, prestate, target))

                bad_target = TargetPrefix(bits, value ^ 1)
                self.assertFalse(bad_target.matches(digest))
                self.assertFalse(working_state_matches_prefix(working, prestate, bad_target))

    def test_h0_modular_interval_wraps(self) -> None:
        target = TargetPrefix(3, 0b001)
        prestate_h0 = 0xF000_0000
        base, span = h0_gate_parameters(prestate_h0, target)
        self.assertEqual(base, 0x3000_0000)
        self.assertEqual(span, 1 << 29)
        for offset in (0, span - 1):
            digest_h0 = (prestate_h0 + base + offset) & MASK32
            self.assertEqual(digest_h0 >> 29, target.value)


class RawTailJobTests(unittest.TestCase):
    def test_layout_invariants_across_source_lengths(self) -> None:
        rng = random.Random(0xB10C_4A11)
        target = TargetPrefix.from_hex("deadbeef")
        lengths = list(range(0, 150)) + [rng.randrange(0, 8000) for _ in range(150)]
        for size in lengths:
            source = rng.randbytes(size)
            job = build_raw_tail_job(source, target)
            with self.subTest(size=size):
                padded = sha1_pad(job.object_template)
                self.assertEqual(job.nonce_object_offset % 64, RAW_NONCE_BLOCK_OFFSET)
                self.assertEqual(job.mutable_block, len(padded) // 64 - 1)
                self.assertEqual(len(job.object_template) % 64, 54)
                self.assertEqual(job.manifest()["candidate_dependent_blocks"], 1)
                self.assertEqual(job.final_block_template[48:53], b"\0" * 5)
                self.assertEqual(job.final_block_template[53:55], b"\n\x80")
                self.assertLess(job.filler_bytes, 128)

    def test_candidate_mapping_digest_and_nonce_policy(self) -> None:
        rng = random.Random(0x4B1_4E5A)
        job = build_raw_tail_job(b"tree deadbeef\n\nexample", TargetPrefix.from_hex("12345"))
        candidates = [0, 1, 0xAABBCCDDEE, (1 << 40) - 1]
        candidates.extend(rng.randrange(1 << 40) for _ in range(300))
        for candidate in candidates:
            nonce = candidate_nonce(candidate)
            self.assertEqual(nonce, candidate.to_bytes(5, "big"))
            self.assertEqual(candidate_words(candidate), (candidate >> 8, (candidate & 0xFF) << 24))
            payload = job.materialize_payload(candidate)
            obj = job.materialize_object(candidate)
            self.assertEqual(payload[job.nonce_payload_offset : job.nonce_payload_offset + 5], nonce)
            self.assertEqual(obj[job.nonce_object_offset : job.nonce_object_offset + 5], nonce)
            words = block_words(job.materialize_final_block(candidate))
            self.assertEqual(words[12], candidate >> 8)
            self.assertEqual(words[13] & 0xFF00_0000, (candidate & 0xFF) << 24)
            self.assertEqual(job.digest_from_prestate(candidate), hashlib.sha1(obj).digest())

        self.assertFalse(nonce_is_log_safe(b"\x01\x02\0\x03\x04"))
        self.assertTrue(nonce_is_log_safe(b"\x01\x02\x03\x04\xff"))

    def test_target_match_uses_full_materialized_object(self) -> None:
        job = build_raw_tail_job(b"tree x\n\nmessage\n", TargetPrefix(1, 0))
        for candidate in range(100):
            digest = hashlib.sha1(job.materialize_object(candidate)).digest()
            self.assertEqual(job.matches_target(candidate), digest[0] < 0x80)

    def test_candidate_verifier_checks_policy_hash_and_target(self) -> None:
        source = b"tree x\n\nmessage\n"
        planning_job = build_raw_tail_job(source, TargetPrefix(1, 0))
        candidate = 0x0102_0304_05
        digest = planning_job.digest(candidate)
        target = TargetPrefix(160, int.from_bytes(digest, "big"))
        exact_job = build_raw_tail_job(source, target)
        self.assertEqual(exact_job.verify_candidate(candidate), digest)

        wrong_job = build_raw_tail_job(source, TargetPrefix(160, int.from_bytes(digest, "big") ^ 1))
        with self.assertRaisesRegex(ValueError, "does not match"):
            wrong_job.verify_candidate(candidate)
        with self.assertRaisesRegex(ValueError, "contains NUL"):
            exact_job.verify_candidate(0)

    def test_generated_artifacts_are_self_describing(self) -> None:
        job = build_raw_tail_job(b"tree x\n\nmessage", TargetPrefix.from_hex("abcdef", 21))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_job(job, output)
            manifest = json.loads((output / "job.json").read_text(encoding="utf-8"))
            self.assertEqual((output / "payload-template.bin").read_bytes(), job.payload_template)
            self.assertEqual((output / "object-template.bin").read_bytes(), job.object_template)
            self.assertEqual((output / "final-block-template.bin").read_bytes(), job.final_block_template)
            self.assertEqual(manifest["schema"], "git-sha1-raw-tail-job-v1")
            self.assertEqual(manifest["nonce"]["mapping"], "candidate.to_bytes(5, 'big')")
            self.assertEqual(manifest["target"]["bits"], 21)
            self.assertEqual(manifest["template_object_sha1"], hashlib.sha1(job.object_template).hexdigest())
            header = (output / "job_constants.cuh").read_text(encoding="utf-8")
            self.assertEqual(header, render_cuda_header(job))
            self.assertIn("JOB_HIN", header)
            self.assertIn("JOB_PRE12", header)
            self.assertIn("JOB_BASE16", header)
            self.assertIn("W12 = candidate >> 8", header)

    def test_newline_mutation_is_explicit(self) -> None:
        with_newline = build_raw_tail_job(b"message\n", TargetPrefix(1, 0))
        without_newline = build_raw_tail_job(b"message", TargetPrefix(1, 0))
        self.assertFalse(with_newline.source_newline_appended)
        self.assertTrue(without_newline.source_newline_appended)
        self.assertEqual(without_newline.payload_template[:8], b"message\n")


class HeaderJobTests(unittest.TestCase):
    def test_layout_and_digest_across_message_lengths(self) -> None:
        rng = random.Random(0x4845_4144)
        for size in list(range(0, 180)) + [511, 1024, 4097]:
            message = rng.randbytes(size)
            source = b"tree deadbeef\nauthor A <a@localhost> 0 +0000\ncommitter A <a@localhost> 0 +0000\n\n" + message
            job = build_header_job(source, TargetPrefix.from_hex("12345678"))
            candidate = 0x6162_6364_65
            with self.subTest(size=size):
                self.assertEqual(job.nonce_object_offset % 64, RAW_NONCE_BLOCK_OFFSET)
                self.assertGreaterEqual(len(job.suffix_blocks), int(bool(message)))
                self.assertEqual(job.mutable_block_template[48:53], b"\0" * 5)
                self.assertEqual(job.digest_from_prestate(candidate), job.digest(candidate))
                payload = job.materialize_payload(candidate)
                separator = payload.index(b"\n\n")
                self.assertIn(b"\nx ", payload[:separator])
                self.assertEqual(payload[separator + 2 :], message)

    def test_header_nonce_policy(self) -> None:
        self.assertTrue(nonce_is_header_safe(b"abcde"))
        self.assertFalse(nonce_is_header_safe(b"ab\0de"))
        self.assertFalse(nonce_is_header_safe(b"ab\nde"))

    def test_git_accepts_header_without_exposing_it_as_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "init", "-q", directory], check=True)
            tree = subprocess.run(
                ["git", "-C", directory, "mktree"],
                input=b"",
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.strip()
            source = (
                b"tree "
                + tree
                + b"\nauthor Test <test@localhost> 0 +0000"
                + b"\ncommitter Test <test@localhost> 0 +0000"
                + b"\n\nvisible subject\n"
            )
            job = build_header_job(source, TargetPrefix(1, 0))
            payload = job.materialize_payload(0x6162_6364_65)
            oid = subprocess.run(
                ["git", "-C", directory, "hash-object", "-t", "commit", "-w", "--stdin"],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            ).stdout.strip()
            stored = subprocess.run(
                ["git", "-C", directory, "cat-file", "commit", oid],
                stdout=subprocess.PIPE,
                check=True,
            ).stdout
            subject = subprocess.run(
                ["git", "-C", directory, "show", "-s", "--format=%s", oid],
                stdout=subprocess.PIPE,
                check=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "-C", directory, "fsck", "--strict", "--no-dangling"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            self.assertEqual(stored, payload)
            self.assertEqual(subject, b"visible subject")
            self.assertIn(b"\nx ", stored[: stored.index(b"\n\n")])

    def test_generated_header_artifacts_include_suffix(self) -> None:
        source = b"tree x\nauthor A <a@localhost> 0 +0000\ncommitter A <a@localhost> 0 +0000\n\nmessage\n"
        job = build_header_job(source, TargetPrefix.from_hex("abcdef"))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_header_job(job, output)
            manifest = json.loads((output / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], "git-sha1-header-job-v1")
            self.assertEqual(
                (output / "mutable-block-template.bin").read_bytes(),
                job.mutable_block_template,
            )
            self.assertEqual(
                (output / "suffix-blocks.bin").read_bytes(),
                b"".join(job.suffix_blocks),
            )

    def test_fixed_value_prefix_creates_new_search_domain(self) -> None:
        source = b"tree x\nauthor A <a@localhost> 0 +0000\ncommitter A <a@localhost> 0 +0000\n\nmessage\n"
        target = TargetPrefix.from_hex("0123456789a")
        first = build_header_job(source, target, value_prefix=b"0000000000000000 ")
        second = build_header_job(source, target, value_prefix=b"0000000000000001 ")
        candidate = 0x6162_6364_65
        self.assertEqual(first.nonce_object_offset % 64, RAW_NONCE_BLOCK_OFFSET)
        self.assertEqual(second.nonce_object_offset % 64, RAW_NONCE_BLOCK_OFFSET)
        self.assertNotEqual(first.object_template, second.object_template)
        self.assertNotEqual(first.digest(candidate), second.digest(candidate))


class MaskedHeaderJobTests(unittest.TestCase):
    def test_candidate_mapping_is_printable_and_bijective(self) -> None:
        candidates = (0, 1, 0x1234_5678_9A, (1 << 40) - 1)
        encoded = [masked_candidate_nonce(candidate) for candidate in candidates]
        self.assertEqual(encoded[0], b" " * 8)
        self.assertEqual(encoded[-1], b"?" * 8)
        self.assertEqual(len(set(encoded)), len(candidates))
        for nonce in encoded:
            self.assertTrue(all(0x20 <= byte <= 0x3F for byte in nonce))

    def test_layout_and_digest_across_message_lengths(self) -> None:
        rng = random.Random(0x384D_4153)
        for size in (0, 1, 31, 32, 63, 64, 255, 1024):
            message = rng.randbytes(size)
            source = b"tree deadbeef\nauthor A <a@localhost> 0 +0000\ncommitter A <a@localhost> 0 +0000\n\n" + message
            job = build_masked_header_job(source, TargetPrefix.from_hex("12345678"))
            candidate = 0x1234_5678_9A
            with self.subTest(size=size):
                self.assertEqual(job.nonce_object_offset % 64, MASKED_NONCE_BLOCK_OFFSET)
                self.assertEqual(job.mutable_block_template[44:52], b"\0" * 8)
                self.assertEqual(job.base_words[11:13], (0, 0))
                self.assertEqual(job.digest_from_prestate(candidate), job.digest(candidate))
                self.assertIn(masked_candidate_nonce(candidate), job.materialize_payload(candidate))


if __name__ == "__main__":
    unittest.main()
