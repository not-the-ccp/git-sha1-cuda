# Custom-header suffix benchmark

The custom-header carrier aligns its five candidate bytes at W12 and the high
byte of W13 in a mutable SHA-1 block. Padded blocks following that block are
fixed. Their 80-word schedules are expanded on the host, uploaded with the
job, and compressed for every candidate before the target comparison.

`tests/c_api_smoke.cpp` contains an exact one-suffix-block commit fixture. Its
GPU digest is checked against the complete CPU digest before benchmarking. The
multi-block measurements repeat the fixed suffix schedule to isolate the cost
of additional compression blocks.

Run a billion-candidate sample with:

```sh
./build/gsv_c_api_smoke 4194304 7 32 header 1
```

The final argument is the fixed suffix-block count. Omitting `header` selects
the message-trailer kernel.

## RTX 4060 results

Environment: GeForce RTX 4060, `sm_89`, CUDA 13.3.1. Each launch evaluates
1,073,741,824 candidates. Reported values are representative steady-state
measurements; the shared GPU can introduce short clock variations.

| Layout | Fixed suffix blocks | SHA-1 blocks per candidate | Throughput |
|---|---:|---:|---:|
| Message trailer | 0 | 1 | 10.95 GH/s |
| Custom header | 0 | 1 | 10.69 GH/s |
| Custom header | 1 | 2 | 5.42 GH/s |
| Custom header | 2 | 3 | 3.60 GH/s |
| Custom header | 4 | 5 | 2.14 GH/s |
| Custom header | 8 | 9 | 1.18 GH/s |

The header search kernels use 87 registers for all target widths, with no
stack, local-memory, or register-spill traffic. The diagnostic header kernel
uses 80 registers. Dynamic shared memory remains 33,280 bytes per block.
