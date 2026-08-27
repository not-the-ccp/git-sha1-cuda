# Designing a Git commit for fast SHA-1 vanity search

This guide is about **layout**, not brute-force mechanics. The single largest performance decision is where the mutable nonce lands in the exact byte string Git hashes.

## 1. Hash the object Git actually hashes

For a SHA-1 repository, a commit object is hashed as:

```text
"commit " + decimal(payload_length) + NUL + raw_commit_payload
```

So byte alignment must include the object header. Changing the payload length can also change the decimal length field and therefore shift every payload byte by one when a digit boundary is crossed.

Use the planner rather than counting visible characters by hand:

```bash
python3 searchable_commit.py --git-ref HEAD --prefix deadbeef
```

For an existing signed PGP commit whose signature must remain valid:

```bash
python3 searchable_commit.py --git-ref HEAD --prefix deadbeef --preserve-signature
```

To materialize the recommended template and a machine-readable plan:

```bash
python3 searchable_commit.py --git-ref HEAD --prefix deadbeef --emit job/
```

## 2. Primary optimization objective: minimize candidate-dependent blocks

SHA-1 processes 64-byte blocks serially. If the first mutable byte is in block `k`, every block before `k` can be prehashed once. The nonce-containing block and every later block must be processed per candidate.

Therefore, all else equal:

```text
nonce in final padded block  >>  nonce in penultimate block  >>  nonce near the start
```

A micro-optimization that saves several instructions cannot compensate for forcing hundreds of extra SHA-1 compressions per candidate.

For an unsigned commit, or a commit whose contents are still free to change, **put the nonce at the tail of the commit message**.

## 3. Best printable layout

A practical default is a base64url nonce in a final trailer-like line, aligned so its first byte is at final-block offset 48:

```text
...
Vanity: XXXXXX
```

Six base64url characters provide 36 search bits. For a 32-bit target, the one-pass success probability is approximately:

```text
1 - exp(-2^(36-32)) ~= 0.999999887
```

This is enough for a normal 32-bit vanity prefix without making the hot kernel unnecessarily large.

Why offset 48? It begins at SHA-1 word W12, leaving rounds 0–11 candidate-invariant and giving a particularly useful W12/W13 specialization while still leaving room for the line terminator and SHA-1 padding in the same block.

The exact number of alignment filler bytes depends on the entire serialized object, so calculate it; do not assume a fixed count.

## 4. Maximum-throughput binary layout

If human-readable message bytes are not important, the fastest architecture currently under development uses five raw bytes at final-block offsets 48–52:

```text
W12 = four-byte outer counter
W13 high byte = inner candidate
```

That provides 40 raw search bits and removes base64 character lookup/packing from the outer schedule. It also gives 256 inner candidates per outer value instead of 64.

Git commit messages can contain non-NUL bytes, but arbitrary binary bytes can make normal log/display tooling unpleasant. Treat this as an **opt-in throughput mode**.

The kernel may hash all 256 byte values directly and reject a found candidate if any of its five bytes is NUL. The probability that all five bytes are non-NUL is `(255/256)^5 ~= 0.98062`, so expected search overhead is only about 1.98%. Because validity is checked only after a rare hash hit, the hot hashing path stays branch-free.

## 5. Search-space size and epochs

You do not need a nonce with vastly more entropy than the requested prefix.

For target width `b` and search-space width `s`, a useful approximation is:

```text
P(success in one complete space) = 1 - exp(-2^(s-b))
```

Examples for a 32-bit prefix:

```text
5 base64url chars = 30 bits -> about 22.1% one-pass success
6 base64url chars = 36 bits -> about 99.9999887%
5 raw bytes       = ~40 bits -> effectively certain for a 32-bit target
```

A smaller hot nonce can also be paired with a fixed **epoch/salt**. If one nonce space is exhausted, change the epoch, regenerate the fixed job constants, and search again. Expected total work to find a random `b`-bit prefix remains about `2^b` candidates; the epoch simply partitions that work into convenient kernel-sized spaces.

## 6. Signed commits are different

Adding or changing ordinary commit-message bytes changes the payload covered by a commit signature. Signing *after* finding a vanity ID also changes the commit object and therefore destroys that ID.

For an existing PGP-signed commit, there is a useful exception: the PGP signature is stored in Git's `gpgsig` header, and ASCII-armored PGP permits a `Comment:` armor header. A nonce can be placed in that armor metadata while keeping the underlying detached signature packet unchanged. The project has verified this end-to-end with `git verify-commit` and byte-identical dearmored signature packets.

Example physical lines inside the multi-line `gpgsig` header:

```text
gpgsig -----BEGIN PGP SIGNATURE-----
 Comment: Vanity-XXXXXX
 ...
 -----END PGP SIGNATURE-----
```

The leading space is required by Git's multi-line header representation.

This preserves the signed payload, but it can be **much slower** if the `gpgsig` occurs early and a long commit message follows it. All suffix SHA-1 blocks after the nonce remain candidate-dependent. Always run the planner and inspect `candidate_dependent_blocks` before choosing this strategy.

Do not assume the same armor-metadata trick applies to SSH or X.509 signatures; the current implementation only recommends it for PGP.

## 7. Long signed commits: optimize the suffix, not the head

For a PGP armor nonce followed by hundreds of fixed blocks, almost all work is in the fixed suffix. The current architecture therefore:

1. prehashes every complete fixed block before the nonce;
2. computes the nonce-containing head block once per candidate;
3. pre-expands the 80-word schedule for every fixed suffix block once;
4. reuses each suffix schedule load across a small vector of candidate states;
5. keeps head temporary state out of the long-lived suffix register frame.

For a 355-block fixed suffix, spending extra registers to make the one mutable block a few percent faster can reduce total throughput. Optimize the dominant suffix path first.

## 8. Prefix comparison can also be specialized

For a 32-bit H0 target, do not compute `H0 = final_a + initial_H0` for every candidate and then compare. Pre-adjust the target once:

```text
final_a == target_H0 - initial_H0  (mod 2^32)
```

For prefixes longer than 32 bits, H0 remains the hot gate. Only after an H0 match do you reconstruct/check H1–H4. Since an H0 match occurs roughly once per `2^32` candidates, the extra digest-word work is effectively off the hot path.

## 9. What the planner reports

`searchable_commit.py` evaluates strategies using:

- exact Git object-header length;
- nonce absolute byte offset;
- nonce SHA-1 block and within-block offset;
- first directly affected SHA-1 word;
- number of candidate-dependent compression blocks;
- effective search bits and one-pass success probability;
- whether the strategy preserves or invalidates an existing signature;
- message cleanliness (printable vs binary).

The top-ranked result is a recommendation, not an immutable rule. For example, a binary-tail strategy can be fastest but undesirable for a public commit message; in that case choose the printable-tail result immediately below it.

## 10. Practical recipes

### Unsigned, normal public commit

Use a six-character base64url tail nonce aligned near final-block offset 48. This is the default balance of interoperability and throughput.

### Unsigned, maximum throughput / controlled repository

Use the raw 4+1-byte binary tail specialization, aligned at offsets 48–52. Materialize and independently verify the winning object before writing it.

### Existing PGP-signed commit, signature must survive

Use a `Comment:` nonce in the PGP armor and optimize the resulting fixed suffix. Expect it to be much slower than a final-block nonce if the commit message after `gpgsig` is long.

### Existing SSH/X.509-signed commit

Do not mutate ordinary message bytes unless you are prepared to invalidate/recreate the signature. The current tooling does not claim a signature-preserving metadata nonce for these formats.

## 11. Always verify the final object

A search result is not done until the CPU independently reconstructs the exact payload and checks it through Git. The safe finalization path is:

1. reconstruct nonce bytes from the GPU candidate ID;
2. reject illegal/message-policy bytes;
3. compute the full Git object SHA-1 independently on CPU;
4. verify the requested prefix;
5. write with `git hash-object -t commit -w --stdin`;
6. require Git's returned object ID to equal the independently calculated ID;
7. round-trip with `git cat-file commit <id>` and require byte-for-byte payload equality;
8. for signed commits, also run `git verify-commit`.

Never trust the GPU's hot prefix filter as the sole correctness check.
