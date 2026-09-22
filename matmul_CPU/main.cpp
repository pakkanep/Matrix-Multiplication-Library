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


int main()
{
    int rows = 1000;
    int cols = 1000;


    constexpr int nb = 4;
    int na = (cols + nb - 1) / nb;

    Matrix m1   = Matrix(rows, cols);
    Matrix m2   = Matrix(rows, cols);

    Matrix dest = Matrix(rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            m1.Set(i, j, cols * i + j);
            m2.Set(i, j, 1 + cols * j + i);
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Matrix::matMulV2(dest, m1, m2);
    Matrix::matMulV6(dest, m1, m2);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time: " << duration.count() << " ms\n";
    // dest.printScalar();
}


/*
g++ -std=c++20 -O2 -march=native main.cpp Matrix.cpp -o program
*/