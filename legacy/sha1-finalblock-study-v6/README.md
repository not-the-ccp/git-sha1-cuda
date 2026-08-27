# SHA-1 Git final-block CUDA study — v5.1

Repair and continuation of Round 5.

Run:

```bash
./run_study.sh --fresh
```

The default `research` profile performs:

1. CPU/harness validation.
2. 230-variant GPU correctness + survey at offsets `0,4,8,12,16,20,24,28,31,32,36,40,44,48`.
3. Repeated confirmation of the top 24 on the already-built binaries.
4. Top-six coverage over every `NONCE_OFF=0..48`, compiling only those finalists.
5. Register-cap experiments (`default, 48, 56, 64, 72, 80`) on the top three at offsets 16/32/48.
6. SASS/resource/ptxas logs and ranked CSV summaries.

If an experimental variant has a genuine hash mismatch, it is recorded and quarantined rather than poisoning the rest of the run. Unexpected CUDA/compiler/harness errors still fail the campaign and create `results-v5.1.partial.tar.xz` containing completed evidence.

Successful runs create `results-v5.1.tar.xz`.

See `BUGFIX.md` for the Round-5 failure analysis.
