#include <iostream>
#include <cstdint>
#include "Matrix.h"
#include <memory>
#include <chrono>
#include <cmath>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

double random_double(uint32_t seed)
{
    return static_cast<double>(pcg_hash(seed)) / static_cast<double>(UINT32_MAX);
}

void detectSimd()
{
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();

        if (__builtin_cpu_supports("avx512f")) {
            std::cout << "AVX-512: 8 doubles per vector\n";
        } else if (__builtin_cpu_supports("avx2")) {
            std::cout << "AVX2: 4 doubles per vector\n";
        } else if (__builtin_cpu_supports("sse2")) {
            std::cout << "SSE2: 2 doubles per vector\n";
        } else {
            std::cout << "Scalar implementation\n";
        }
    #endif
}

void createVecMat(int rows, int cols, const double *data, double4_t *vecMat)
{
    // elements per vector
    constexpr int nb = 4;
    // vectors per input row
    int na = (cols + nb - 1) / nb; // Each row has cols elements
    

    for (int row = 0; row < rows; ++row)
    {
        for (int vec = 0; vec < na; ++vec)
        {
            for (int i = 0; i < nb; ++i)
            {
                int j = vec * nb + i;
                vecMat[na * row + vec][i] = j < cols ? static_cast<double>(data[cols * row + j]) : 0.0;
            }
            
        }
    }
}

void normalizeRows(int ny, int nx, const double *data, double *result)
{
    double coeff = 1.0 / (double)nx;
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; ++i)
    {
        double sum = 0;
        for (int j = 0; j < nx; ++j)
        {
            sum += (double)data[j + (i * nx)];
        }
        
        double mean = coeff * sum;
        
        for (int j = 0; j < nx; ++j)
        {
            double x = data[j + (i * nx)];
            result[j + (i * nx)] = (x - mean);
        }
    }
}

void euclideanNorm(int ny, int nx, double *X)
{
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < ny; ++i)
    {
        double sumOfSquares = 0.0;

        // _____sum of squares of each row_____
       
        for (int j = 0; j < nx; ++j) // items
        {
            double x = X[j + (i * nx)]; 
            sumOfSquares += x * x;
        }

        double norm = std::sqrt(sumOfSquares);
        
        for (int j = 0; j < nx; ++j)
        {
            double x = X[j + (i * nx)];
            X[j + (i * nx)] = x / norm;
        }
    }
}

int main()
{
    int rows = 4;
    int cols = 4;

    // elements per vector
    constexpr int nb = 4;
    // vectors per input row
    int na = (cols + nb - 1) / nb; // Each row has cols elements

    // Matrix data = Matrix(rows, cols);
    Matrix m1 =   Matrix(rows, cols);
    Matrix m1_T = Matrix(cols, rows);
    Matrix dest = Matrix(rows, rows);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            m1.Set  (i, j, cols * i + j);
            m1_T.Set(j, i, cols * i + j);

        }
    }
    // m1.printScalar();
    // m1_T.printScalar();
    // normalizeRows(rows, cols, data.scalarData(0), m1.scalarData(0));
    // euclideanNorm(rows, cols, m1.scalarData(0));


    // #pragma omp parallel for schedule(dynamic)
    // for (int row = 0; row < rows; ++row)
    // {
    //     for (int vec = 0; vec < na; ++vec)
    //     {
    //         for (int i = 0; i < nb; ++i)
    //         {
    //             int j = vec * nb + i;
    //             if (j < cols) { m1.SetVec(row, vec, i, m1(row, j)); }
    //             else { m1.SetVec(row, vec, i, 0.0); }
    //         }
    //     }
    // }


    auto start = std::chrono::high_resolution_clock::now();

    Matrix::matMulV5(dest, m1, m1_T); // 2000*2000, Time: 136 ms, parallel
    // Matrix::matMulV8(dest, m1);       // 2000*2000, Time: 541 ms, parallel
    // Matrix::matMulV9(dest, m1);       // 2000*2000, Time: 459 ms, parallel
    // Matrix::matMulV10(dest, m1);      // 2000*2000, Time:  ms, parallel

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time: " << duration.count() << " ms\n";
    dest.printScalar();
}
/*
Peak performance:
FLOPS = cores * (cycles / seconds) * (FLOPs / cycles)

// sizeof(double) = 8 bytes
// SSE:     128 bits 16 bytes 
// AVX:     256 bits 32 bytes
// AVX-512: 512 bits 64 bytes
// cache line 64 bytes
// register size < 1KB
// L1 32  KB
// L2 256 KB
// L3 6   MB  


g++ -std=c++20 -march=native -g -O0 -fno-omit-frame-pointer -Wall -Wextra main.cpp Matrix.cpp -o program
*/