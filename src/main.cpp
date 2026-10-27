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
    

    benchmark<T>(Matrix<double>::matMulV10, dest, m1, m2);
    

    dest.initToZero();
    PerfCounter pc;
    
    auto instructions = pc.measure( [&] {
        Matrix<double>::matMulV10(dest, m1, m2);
    });

    std::cout << "used instructions: " << instructions << "\n";

}

int main()
{
    matMulTestSuite<double>();
}


/*
Notes:

general Peak GFLOP/s = cores x GHz x FLOPs/cycle/core

naive:
keep the same

V1:
Median time:         3.95974 s
Performance:         4.04067 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     2.63065 %

V2:
Median time:         1.71642 s
Performance:         9.32175 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     6.06885 %

V3:
Median time:         1.2179 s
Performance:         13.1374 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     8.553 %

V4:
keep the same

V5:
Median time:         0.350853 s
Performance:         45.6032 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     29.6896 %

V6:
Median time:         0.533503 s
Performance:         29.9905 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     19.5251 %

V7:
Median time:         0.48033 s
Performance:         33.3104 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     21.6865 %

V8:
Median time:         0.332481 s
Performance:         48.123 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     31.3301 %

V9:
Median time:         0.303146 s
Performance:         52.7799 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     34.3619 %

V10:
Median time:         0.174707 s
Performance:         91.5819 GFLOP/s
Theoretical peak:    153.6 GFLOP/s
Percent of peak:     59.6236 %
*/