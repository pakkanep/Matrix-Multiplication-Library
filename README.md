#  matmul
## Performance Benchmarks
I ran these benchmarks under WSL2, so these are more indicative than really accurate.

Performed using `double` precision (FP64).

The theoretical peak performance of the Ryzen 5 7500F at its
3.7 GHz base clock is:

$$
\text{Peak GFLOP/s}
= 6 \times 3.7 \times (2 \times 4 \times 2)
= 355.2\ \text{GFLOP/s}
$$

where:

- `6` = physical CPU cores
- `3.7` = base clock frequency in GHz
- `2` = vector FMAs per cycle
- `4` = doubles per vector
- `2` = FLOPs per double (multiply + add)

| Implementation | Description | Block Size | Median Time (s) | Performance (GFLOP/s) | Peak Performance |
|:---------------|:------------|:----------:|----------------:|----------------------:|-----------------:|
| `matMul` | Naive implementation | — | 7.891 | 2.03 | 0.57% |
| `matMulV1` | Linear memory access for A and B | — | 1.895 | 8.44 | 2.38% |
| `matMulV2` | Cache blocking with linear memory access | 64 | 1.161 | 13.78 | 3.88% |
| `matMulV3` | Cache blocking with reduced memory traffic | 64 | 0.763 | 20.96 | 5.90% |
| `matMulV4` | Baseline OpenMP implementation | — | 3.908 | 4.09 | 1.15% |
| `matMulV5` | Cache blocking + OpenMP + linear memory access | 64 | **0.188** | **85.26** | **24.00%** |

**Theoretical FP64 peak:** 355.2 GFLOP/s

