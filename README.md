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

| Implementation | Description | Block Size | Median Time (s) | Performance (GFLOP/s) | % of Theoretical Peak | Speedup vs. Naive |
|:---------------|:------------|:----------:|----------------:|----------------------:|----------------------:|------------------:|
| `matMul`   | Naive implementation | — | 7.891 | 2.03 | 0.57% | 1.00× |
| `matMulV1` | Linear memory access for A and B | — | 1.895 | 8.44 | 2.38% | 4.16× |
| `matMulV2` | Cache blocking with linear memory access | 64 | 1.161 | 13.78 | 3.88% | 6.80× |
| `matMulV3` | Cache blocking with reduced memory traffic | 64 | 0.763 | 20.96 | 5.90% | 10.34× |
| `matMulV4` | Baseline OpenMP implementation | — | 3.908 | 4.09 | 1.15% | 2.02× |
| `matMulV5` | Cache blocking + OpenMP + linear memory access | 64 | 0.139 | 115.23 | 32.44% | 56.77× |
| `matMulV6` | V5 + 256-bit SIMD | 64 | 0.176 | 89.26 | 25.13% | 44.84× |
| `matMulV7` | V5 + 512-bit SIMD | 64 | 0.204 | 78.74 | 22.17% | 38.68× |
| `matMulV8` | V5 + 512-bit SIMD + register blocking | 64 | **0.110** | **145.39** | **40.93%** | **71.70×** |

### SIMD Performance

One might expect that if we process 4 or 8 elements simultaneously instead of 1, we would also gain similar 4x or 8x performance boost.
But, as we can see that V6 or V7 actually perform similarly or worse than the scalar V5.


Possible reasons for this:
- Compiler auto-vectorization: V5 is compiled with -O3 -march=native, so the compiler might already auto vectorize some parts for us
- Register and memory traffic: We can compute faster than move data

next kernel improvement will be here (Matrix.cpp, line 206): (implemented in V8)

    int j = 0;
    for ( ; (j+width) < jBlock; j += width) // run as long as (j+width) < jBlock
    {
        const __m512d bReg = _mm512_loadu_pd( &b[ j ] );
        __m512d cReg = _mm512_loadu_pd( &dest[ j ] );
        cReg = _mm512_fmadd_pd( aReg, bReg, cReg );
        _mm512_storeu_pd( &dest[j], cReg );
    }

this is a bottleneck as it repeatedly loads from and stores to dest jblock times.

We got pretty impressive results in V8 by fixing this problem in the SIMD kernel by using register blocking



**Theoretical FP64 peak:** 355.2 GFLOP/s

