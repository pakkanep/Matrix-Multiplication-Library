#include <iostream>
#include <cstdint>
#include <memory>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

#include "matmul_CPU/matrix.h"
#include "perf_counting.hpp"
#include "utils.cpp"


template <typename T, typename Func>
double benchmark(
    Func testFunc,
    Matrix<T>& dest,
    const Matrix<T>& A,
    const Matrix<T>& B,
    int reps = 10)
{
    std::vector<double> times;
    times.reserve(reps);

    for (int i = 0; i < reps; ++i)
    {
        dest.initToZero();
        auto start = std::chrono::steady_clock::now();
        testFunc(dest, A, B);
        auto end = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(end - start).count();
        times.push_back(time);
    }

    std::sort(times.begin(), times.end());
    double median;

    if (reps % 2 == 0)
    {
        median = (times[reps / 2 - 1] + times[reps / 2]) / 2.0;
    }
    else
    {
        median = times[reps / 2];
    }

    // matmul performs ~ 2*M*K*N FLOPs
    double flops =
        2.0 *
        static_cast<double>(A.rows) *
        static_cast<double>(A.cols) *
        static_cast<double>(B.cols);

    // FLOP/s to GFLOP/s
    double gflops = flops / median / 1e9;
    // calculations shown at the end of this file
    // constexpr double theoreticalPeak = 355.2;
    constexpr double theoreticalPeak = 153.6;

    double percentOfPeak = (gflops / theoreticalPeak) * 100.0;

    std::cout << "Median time:         "<< median << " s\n";
    std::cout << "Performance:         "<< gflops << " GFLOP/s\n";
    std::cout << "Theoretical peak:    "<< theoreticalPeak << " GFLOP/s\n";
    std::cout << "Percent of peak:     "<< percentOfPeak << " %\n";

    return median;
}


template<typename T>
void matMulTestSuite()
{
    int rows = 2000;
    int cols = 2000;

    Matrix m1   = Matrix<T>(rows, cols);
    Matrix m2   = Matrix<T>(rows, cols);
    Matrix dest = Matrix<T>(rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            m1.Set(i, j, cols * i + j);
            m2.Set(i, j, cols * i + j);
        }
    }
    
    // input 2000x2000 for all
    // benchmark<T>(Matrix<double>::matMul, dest, m1, m2);
    // benchmark<T>(Matrix<double>::matMulV1, dest, m1, m2);
    // benchmark<T>(Matrix<double>::matMulV2, dest, m1, m2);
    // benchmark<T>(Matrix<double>::matMulV3, dest, m1, m2);
    // benchmark<T>(Matrix<double>::matMulV4, dest, m1, m2);
    // benchmark<T>(Matrix<double>::matMulV5, dest, m1, m2);
    // benchmark<T>(Matrix<double>::matMulV6, dest, m1, m2);
    // benchmark<T>(Matrix<double>::matMulV7, dest, m1, m2);
    // benchmark<T>(Matrix<double>::matMulV8, dest, m1, m2);
    PerfCounter pc;
    
    auto instructions = pc.measure( [&] {
        Matrix<double>::matMul(dest, m1, m2);
    });

    std::cout << "used instructions: " << instructions << "\n";
    // Matrix<double>::matMulV8(dest, m1, m2);
    // Matrix<double>::matMul(dest, m1, m2);
    //dest.printScalar();
}

int main()
{
    detectSimd();
    matMulTestSuite<double>();
}


/*
Notes:

general Peak GFLOP/s = cores x GHz x FLOPs/cycle/core

for my CPU (Ryzen 5 7500f) using double (FP64)

Peak GFLOP/s = 6 x 3.7 Ghz x (2 x 4 x 2) = 355.2
where: 
6   = physical CPU cores
3.7 = GHz base clock
2   = 256-bit vector FMAs per cycle per core
4   = doubles per 256-bit vector
2   = FLOPs per double in an FMA, as it includes mul&add

FMA      Fused Multiply-Add
FLOP     Floating-Point Operation
GFLOP/s  10^9 floating operations per second

i5
4 cores
2.4 GHz
2 * 4 * 2 = 16 FLOPs/cycle/core
peak FP64 = 4 (cores) * 2.4 (GHz) * (2 * 4 * 2 ) = 153.6 GFLOP/s
*/