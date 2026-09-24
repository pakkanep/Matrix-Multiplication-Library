#  matmul
## Performance Benchmarks

Benchmarks were performed using `double` precision (FP64). The theoretical peak
performance of the Ryzen 5 7500F at its 3.7 GHz base clock is:

\[
6 \times 3.7 \times (2 \times 4 \times 2)
= 355.2 \text{ GFLOP/s}
\]

| Implementation | Description | Block Size | Median Time (s) | Performance (GFLOP/s) | Peak Performance |
|:---------------|:------------|:----------:|----------------:|----------------------:|-----------------:|
| `matMul` | Naive implementation | — | 7.891 | 2.03 | 0.57% |
| `matMulV1` | Linear memory access for A and B | — | 1.895 | 8.44 | 2.38% |
| `matMulV2` | Cache blocking with linear memory access | 64 | 1.161 | 13.78 | 3.88% |
| `matMulV3` | Cache blocking with reduced memory traffic | 64 | 0.763 | 20.96 | 5.90% |
| `matMulV4` | Baseline OpenMP implementation | — | 3.908 | 4.09 | 1.15% |
| `matMulV5` | Cache blocking + OpenMP + linear memory access | 64 | **0.188** | **85.26** | **24.00%** |

**Theoretical FP64 peak:** 355.2 GFLOP/s


matMul (Naive)
Median time:         7.89061 s
Performance:         2.02773 GFLOP/s
Theoretical peak:    355.2 GFLOP/s
Percent of peak:     0.570869 %

matMulV1 (linear access for both A and B)
Median time:         1.89546 s
Performance:         8.4412 GFLOP/s
Theoretical peak:    355.2 GFLOP/s
Percent of peak:     2.37646 %

matMulV2 (cache blocked version with linear access) with block size 64
Median time:         1.16142 s
Performance:         13.7762 GFLOP/s
Theoretical peak:    355.2 GFLOP/s
Percent of peak:     3.87844 %

matMulV3 (blocked version with a kernel which should have less memory traffic) with block size 64
Median time:         0.763481 s
Performance:         20.9567 GFLOP/s
Theoretical peak:    355.2 GFLOP/s
Percent of peak:     5.89996 %

matMulV4 (baseline OpenMP implementation, perfomance suffers from non linear memory access)
Median time:         3.90808 s
Performance:         4.09408 GFLOP/s
Theoretical peak:    355.2 GFLOP/s
Percent of peak:     1.15261 %

matMulV5 (cache blocked, OpenMP and linear access) with block size 64
Median time:         0.187663 s
Performance:         85.2592 GFLOP/s
Theoretical peak:    355.2 GFLOP/s
Percent of peak:     24.0031 %
