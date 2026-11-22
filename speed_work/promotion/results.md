# CPU π tuning results

The release keeps base-10⁸ limbs from the Chudnovsky leaves through decimal output. Independent binary-limb tests with GMP and FLINT had an advantage on short products, but did not establish a faster full decimal pipeline. In a single-worker MPFR comparison at 50 million digits, converting the binary result to decimal took longer than the entire decimal implementation at that precision.

## Where time went

Four instrumented 1B runs of the optimized standard schedule averaged 14.592 s on the Ryzen 9 9950X3D. The series, including its sieve and overlapped square-root preparation, took 12.761 s (87.5%). Division took 1.231 s, final scaling and correction took 0.555 s, and formatting and cleanup took 0.043 s. Worker setup accounted for the remainder. The inverse square root overlapped the last subtree by 0.479 s, leaving 0.264 s on the critical path. Hashing took another 0.368 s after the computation timer stopped.

## Kernel choices

* Short schoolbook products accumulate even and odd diagonals in separate AVX-512 lanes. At 64 balanced decimal limbs the measured kernel fell from 1.14 µs to 0.47 µs. The factored series shifts each term's numerator into the preceding matrix product, removing a multiplication from the leaf sum.
* The measured 32-worker memory-copy ceilings were 39.6 GB/s with cached stores and 61.6 GB/s with non-temporal stores. An unpacked nine-field transform of two 16-million-limb operands moves about 2.42 GB in the inter-pass traffic model and took 66.7 ms. That localized the large-product work to residue movement. These are modeled bytes and measured copy rates. The guest did not expose memory-controller counters.
* For high products, the decimal CRT schedule skips inverse-output stores below the carry guard and repeats the full product if the guard is ambiguous. This reduced a 12-million-limb high product from 45.68 to 42.66 ms. Packing eligible full-product output tiles reduced the same-size full product from 50.20 to 47.35 ms. Neither technique gave a consistent win at four million limbs.
* For two-field products with radix D = 10¹², choosing the prime p = 1536D + 1 splits the CRT coefficient into two radix-D terms with shifts and additions. For eight or nine fields, collecting twelve decimal digits per cofactor weight reduced single-worker reconstruction time at 32,768 points by 5.5% and 6.7%. Six and seven fields retained eight-digit weights.
* A 1B counter run used the new two-field prime 17,116 times and skipped stores for 506 million field residues. The compact full-product rule did not fire at this precision. A faster isolated kernel can have little effect when the series uses a different operand shape.

## Choosing the release schedules

The standard schedule includes the short multiplication, series and parallel-transform improvements. The decimal schedule adds the specialized prime, wider CRT weights and inverse-output changes. The comparison below uses two independent native PGO builds, default memory settings, fresh processes and 32 workers. Paired runs reverse build order where there is more than one pair. Times are medians within each build at that precision. The main README reports separate measurements of the integrated executable.

| Digits | Pairs | Standard (s) | Decimal (s) | Release choice |
|---|---:|---:|---:|---|
| 1B | 3 | 14.676 | 14.427 | decimal |
| 2B | 2 | 31.675 | 32.763 | standard |
| 3B | 2 | 52.686 | 51.343 | decimal |
| 4B | 2 | 72.112 | 71.423 | decimal |
| 5B | 1 | 91.404 | 93.994 | standard |

For intermediate precisions the release uses the nearest measured size, with the endpoint schedules extending below 1B and above 5B. These crossover points interpolate five measurements on one machine. [Per-run comparison](schedule-comparison.csv) · [Integrated release measurements](../../benchmarks/ryzen9950x3d.csv)
