#  matmul
## Performance Benchmarks

## TODO:
- add padding. Some design qs:
    - entirely hidden from the user?
    - should the padded dims only be used internally and the user only sees the dims that was initialized
    - below

- bite the bullet and implement packing. Organize the data in the same order it's accessed during computation. pack the data every time before matmul?

- certain dimensions such as 1024, 2048 or 4096 performs extremely bad due to something called cache associativity conflicts. Find some solution to this, maybe add padding or maybe the packing solves this.

- add counters for atleast cycles and l1d access and a easier way to profile certain sections.

- 

Performed using `double` precision (FP64) and matrix dims: 2000×2000

Compiled with following flags: -O3 -march=native -fopenmp

The theoretical peak performance of my Intel i5-1135G7
4 cores
2.4 GHz
2 * 4 * 2 = 16 FLOPs/cycle/core
peak FP64 = 4 (cores) * 2.4 (GHz) * (2 * 4 * 2 ) = 153.6 GFLOP/s


| Implementation | Description | Block Size | Median Time (s) | Performance (GFLOP/s) | % of Theoretical Peak | Speedup vs. Naive |
|:---------------|:------------|:----------:|----------------:|----------------------:|----------------------:|------------------:|
| `matMul`    | Naive implementation | — | 7.891 | 2.03 | 0.57% | 1.00× |
| `matMulV1`  | Linear memory access for A and B | — | 3.960 | 4.04 | 2.63% | 1.99× |
| `matMulV2`  | Cache blocking with linear memory access | 64 | 1.716 | 9.32 | 6.07% | 4.60× |
| `matMulV3`  | Cache blocking with reduced memory traffic | 64 | 1.218 | 13.14 | 8.55% | 6.48× |
| `matMulV4`  | Baseline OpenMP implementation | — | 3.908 | 4.09 | 1.15% | 2.02× |
| `matMulV5`  | Cache blocking + OpenMP + linear memory access | 64 | 0.351 | 45.60 | 29.69% | 22.49× |
| `matMulV6`  | V5 + 256-bit SIMD | 64 | 0.534 | 29.99 | 19.53% | 14.79× |
| `matMulV7`  | V5 + 512-bit SIMD | 64 | 0.480 | 33.31 | 21.69% | 16.43× |
| `matMulV8`  | V7 + separate block sizes per dimension | TODO | 0.332 | 48.12 | 31.33% | 23.73× |
| `matMulV9`  | V5 + 512-bit SIMD + register blocking | 64 | 0.303 | 52.78 | 34.36% | 26.03× |
| `matMulV10` | V9 + separate block sizes per dimension | TODO | **0.175** | **91.58** | **59.62%** | **45.17×** |

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
and even more impressiv results when  

