# CpuPi

CpuPi computes π on the CPU with all arithmetic and decimal conversion in RAM. The digit count is the number of digits **after the decimal point**. You can benchmark the calculation, stream the full decimal result to stdout, or write it to a file.

## Build and run

Build on the machine that will run CpuPi. The release targets Linux x86-64 with AVX-512 IFMA, F, DQ, BW and VL. It requires GCC with C++20 support, GNU Make, POSIX threads and OpenSSL's libcrypto development headers. On Debian or Ubuntu, the build packages are `g++`, `make` and `libssl-dev`.

```sh
make -j2
./pi_cpu 100
./pi_cpu 1000000000 --benchmark --threads 32
./pi_cpu 1234567 --stdout > pi.txt
./pi_cpu 5000000000 --output pi-5b.txt
```

`make` builds a native, profile-guided executable. It trains once on 60 million digits. The optional thread count defaults to the number of CPUs in the process affinity mask, up to 32. Use `--threads N` (or a trailing `N`) to select 1–256 threads.

| Mode | Output |
|---|---|
| `--benchmark`, `-b` | Last 32 fractional digits (or all available), SHA-256, and computation time |
| `--stdout` (default) | Full decimal text followed by a newline |
| `--output FILE`, `-o FILE` | The same text written to a file |

The timer covers worker startup, arithmetic and decimal formatting. It stops before hashing or writing output. SHA-256 covers the ASCII decimal text including `3.` and excluding the newline. Zero fractional digits produces `3`. The program accepts arbitrary digit counts that fit the address space and working RAM. Peak memory depends on the multiplication plans, thread count and available memory. `PI_TRANSFORM_MIB` can limit the principal transform workspace.

## Performance

Measured on an AMD Ryzen 9 9950X3D with 16 cores, 32 threads and 124 GiB of RAM under Linux/WSL2. CpuPi used GCC 11.4 and 32 workers. CpuPi times are three-run medians for 1B–3B and individual runs for 4B and 5B. Peak resident memory (RSS) is the maximum from the three-run sweep at each size. The y-cruncher column gives the median from an earlier three-run comparison on the same machine using Komari 0.8.7.9547. All CpuPi runs were checked against the SHA-256 of the full y-cruncher decimal output.

| Fractional digits | CpuPi compute (s) | CpuPi peak RSS (GiB) | y-cruncher compute (s) |
|---:|---:|---:|---:|
| 1 billion | 14.33 | 7.76 | 15.76 |
| 2 billion | 31.54 | 14.90 | 33.85 |
| 3 billion | 51.07 | 23.51 | 53.95 |
| 4 billion | 70.49 | 29.91 | 74.61 |
| 5 billion | 90.19 | 36.20 | 96.76 |

[Per-run measurements and SHA-256 digests](benchmarks/ryzen9950x3d.csv)

The release also matched the reference at 10 billion digits by full SHA-256 and at 987,654,321 digits by direct byte comparison.

Run `python3 benchmarks/verify.py --threads 32` to reproduce the five-size sweep (Python 3 and GNU time). Run `make check` for decimal, CLI and integer arithmetic tests, then `./check-integer --large` for the larger multiplication cases.

## Algorithms

CpuPi sums the Chudnovsky series with binary splitting. It cancels common prime factors before large products and retains the precision needed for the requested digits. A lower-memory splitting path handles problems beyond the factor sieve's range.

Integers use base 10⁸ limbs. Multiplication selects schoolbook, Karatsuba, bounded floating-point FFT or exact multi-prime number-theoretic transforms with Chinese remainder reconstruction. Newton iteration computes the reciprocal and inverse square root. The final limbs format directly as decimal digits.
