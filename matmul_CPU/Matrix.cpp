#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include <immintrin.h>

template <typename T>
void Matrix<T>::printScalar() const
{
    constexpr int width = 6;

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            std::cout << std::setw(width) << getItem(row, col);
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

template <typename T>
void Matrix<T>::Set(int row, int col, T v)
{
    data[this->cols * row + col] = v;
}

template <typename T>
T Matrix<T>::getItem(int row, int col) const
{
    return data[this->cols * row + col];
}

template <typename T>
void Matrix<T>::initToZero()
{
    std::fill(data.begin(), data.end(), T{});
}


// baseline kernel
template <typename T>
void Matrix<T>::baseLineKernelV1(
    const T* a,
    const T* mb,
    T* c,
    int Acols,
    int Bcols,
    int blockRows,
    int blockDepth,
    int blockCols
)
{
    for (int i = 0; i < blockRows; ++i, c += Bcols, a += Acols)
    {
        const T* b = mb;

        for (int k = 0; k < blockDepth; ++k, b += Bcols)
        {
            const T a_ik = a[k];

            for (int j = 0; j < blockCols; ++j)
            {
                c[j] += a_ik * b[j];
            }
        }
    }
}


template <typename T>
void Matrix<T>::baseLineKernelV2(
    const T* a,
    const T* mb,
    T* c,
    int Acols,
    int Bcols,
    int blockRows,
    int blockDepth,
    int blockCols
)
{
    constexpr int CREGS = 64;
    int barrier = std::min(CREGS, blockCols);

    for (int i = 0; i < blockRows; ++i, c += Bcols, a += Acols)
    {
        const T* b = mb;

        T c_temp[CREGS] = {};

        for (int k = 0; k < blockDepth; ++k, b += Bcols)
        {
            const T a_ik = a[k];

            for (int j = 0; j < barrier; ++j)
            {
                c_temp[j] += a_ik * b[j];
            }
        }

        for (int j = 0; j < barrier; ++j)
        {
            c[j] += c_temp[j];
        }
    }
}


template <>
void Matrix<double>::kernelSIMD256(
    const double* srcA,
    const double* srcB,
    double* dest,
    int Acols,
    int Bcols,
    int iBlock,
    int jBlock,
    int kBlock
)
{
    constexpr int width = 256 / (sizeof(double) * 8); // so we process 4 doubles/per computation
    for (int i = 0; i < iBlock; ++i, dest += Bcols, srcA += Acols)
    {
        const double* b = srcB;

        for (int k = 0; k < kBlock; ++k, b += Bcols)
        {
            __m256d aReg = _mm256_set1_pd( srcA[ k ] ); // aReg = [a[k], a[k], a[k], a[k]]
            int j = 0;
            for ( ; (j+width) <= jBlock; j += width) // run as long as (j+width) <= jBlock
            {
                __m256d bReg = _mm256_loadu_pd( &b[ j ] ); // bReg = [b[j], ... , b[j+n]]
                __m256d cReg = _mm256_loadu_pd( &dest[ j ] );
                cReg = _mm256_fmadd_pd( aReg, bReg, cReg );
                _mm256_storeu_pd( &dest[j], cReg );
            }

            for ( ; j < jBlock; ++j)
            {
                dest[j] += srcA[k] * b[j];
            }
        }
    }
}

template <>
void Matrix<double>::kernelSIMD512(
    const double* srcA,
    const double* srcB,
    double* dest,
    int Acols,
    int Bcols,
    int iBlock,
    int jBlock,
    int kBlock
)
{
    constexpr int width = 512 / (sizeof(double) * 8); // so we process 8 doubles/per computation
    for (int i = 0; i < iBlock; ++i, dest += Bcols, srcA += Acols)
    {
        const double* b = srcB;

        for (int k = 0; k < kBlock; ++k, b += Bcols)
        {
            const __m512d aReg = _mm512_set1_pd( srcA[ k ] ); // aReg = [a[k], a[k], a[k], a[k]]
            int j = 0;

            for ( ; (j+width) <= jBlock; j += width) // run as long as (j+width) <= jBlock
            {
                const __m512d bReg = _mm512_loadu_pd( &b[ j ] ); // bReg = [b[j], ... , b[j+8]]
                __m512d cReg       = _mm512_loadu_pd( &dest[ j ] );
                cReg               = _mm512_fmadd_pd( aReg, bReg, cReg );
                _mm512_storeu_pd( &dest[j], cReg );
            }

            for ( ; j < jBlock; ++j)
            {
                dest[j] += srcA[k] * b[j];
            }
        }
    }
}

template <>
void Matrix<double>::kernelSIMD512V2(
    const double* a,
    const double* mb,
    double* c,
    int Acols,
    int Bcols,
    int blockRows,
    int blockDepth,
    int blockCols
)
{
    constexpr int WIDTH = 8;
    constexpr int CREGS = 8;
    constexpr int COLS  = WIDTH * CREGS;

    const int barrier = std::min(COLS, blockCols);

    for (int i = 0; i < blockRows; ++i, c += Bcols, a += Acols)
    {
        __m512d c_temp[CREGS];

        for (int r = 0; r < CREGS; ++r)
        {
            c_temp[r] = _mm512_setzero_pd();
        }

        const double* b = mb;

        int vectorCols = (barrier / WIDTH) * WIDTH;
        int numVecs = vectorCols / WIDTH;

        for (int k = 0; k < blockDepth; ++k, b += Bcols)
        {
            const __m512d aReg = _mm512_set1_pd(a[k]);
            for (int r = 0; r < numVecs; ++r)
            {
                const __m512d bReg = _mm512_loadu_pd(&b[r * WIDTH]);
                c_temp[r] =_mm512_fmadd_pd(aReg, bReg, c_temp[r]);
            }
        }

        for (int r = 0; r < numVecs; ++r)
        {
            __m512d cReg = _mm512_loadu_pd(&c[r * WIDTH]);
            cReg = _mm512_add_pd(cReg, c_temp[r]);
            _mm512_storeu_pd(&c[r * WIDTH], cReg);
        }

        for (int j = vectorCols; j < barrier; ++j)
        {
            double sum = 0.0;
            const double* bScalar = mb;

            for (int k = 0; k < blockDepth; ++k, bScalar += Bcols)
            {
                sum += a[k] * bScalar[j];
            }

            c[j] += sum;
        }
    }
}

// true baseLine 
template <typename T>
void Matrix<T>::matMul(Matrix<T>& dest, const Matrix<T>& srcA, const Matrix<T>& srcB)
{
    int Arows = srcA.rows;
    int Acols = srcA.cols;
    int Bcols = srcB.cols;
    
    for (int i = 0; i < Arows; ++i)
    {
        for (int j = 0; j < Bcols; ++j)
        {
            T sum = {};
            for (int k = 0; k < Acols; ++k)
            {
                sum += srcA(i, k) * srcB(k, j);
            }
            dest(i, j) = sum;
        }
    }
}

// linear access for both A and B
template <typename T>
void Matrix<T>::matMulV1(Matrix<T>& dest, const Matrix<T>& srcA, const Matrix<T>& srcB)
{
    int Arows = srcA.rows;
    int Acols = srcA.cols;
    int Bcols = srcB.cols;
    
    for (int i = 0; i < Arows; ++i)
    {
        for (int k = 0; k < Acols; ++k)
        {
            auto A_ik = srcA(i, k);
            for (int j = 0; j < Bcols; ++j)
            {
                dest(i, j) += A_ik * srcB(k, j);
            }
        }
    }
}

// cache blocked version with linear access
template <typename T>
void Matrix<T>::matMulV2(Matrix<T>& dest, const Matrix<T>& srcA, const Matrix<T>& srcB)
{
    const int Arows = srcA.rows;
    const int Acols = srcA.cols;
    const int Bcols = srcB.cols;

    constexpr int BLOCK = 64;

    for (int ib = 0; ib < Arows; ib += BLOCK)
    {
        const int blockRows = std::min(BLOCK, Arows - ib);

        for (int kb = 0; kb < Acols; kb += BLOCK)
        {
            const int blockDepth = std::min(BLOCK, Acols - kb);

            for (int jb = 0; jb < Bcols; jb += BLOCK)
            {
                const int blockCols = std::min(BLOCK, Bcols - jb);

                const T* a = &srcA(ib, kb);
                const T* b = &srcB(kb, jb);
                T* c       = &dest(ib, jb);

                baseLineKernelV1(
                    a,
                    b,
                    c,
                    Bcols,
                    Acols,
                    blockRows,
                    blockDepth,
                    blockCols
                );
            }
        }
    }
}

// blocked version with a kernel that has less memory traffic
template <typename T>
void Matrix<T>::matMulV3(Matrix<T>& dest, const Matrix<T>& srcA, const Matrix<T>& srcB)
{
    const int Arows = srcA.rows;
    const int Acols = srcA.cols;
    const int Bcols = srcB.cols;

    constexpr int BLOCK = 64;

    for (int ib = 0; ib < Arows; ib += BLOCK)
    {
        const int blockRows = std::min(BLOCK, Arows - ib);

        for (int kb = 0; kb < Acols; kb += BLOCK)
        {
            const int blockDepth = std::min(BLOCK, Acols - kb);

            for (int jb = 0; jb < Bcols; jb += BLOCK)
            {
                const int blockCols = std::min(BLOCK, Bcols - jb);

                const T* a = &srcA(ib, kb);
                const T* b = &srcB(kb, jb);
                T* c       = &dest(ib, jb);

                baseLineKernelV2(
                    a,
                    b,
                    c,
                    Bcols,
                    Acols,
                    blockRows,
                    blockDepth,
                    blockCols
                );
            }
        }
    }
}

// baseline OpenMP implementation, perfomance suffers from non linear memory access 
template <typename T>
void Matrix<T>::matMulV4(Matrix<T>& dest, const Matrix<T>& srcA, const Matrix<T>& srcB)
{
    int rowsA = srcA.rows;
    int colsA = srcA.cols;
    int colsB = srcB.cols;
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rowsA; ++i)
    {
        for (int j = 0; j < colsB; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k)
            {
                sum += srcA(i, k) * srcB(k, j);
            }
            dest.Set(i, j, sum);
        }
    }
}

// cache blocked, OpenMP and linear access
template <typename T>
void Matrix<T>::matMulV5(Matrix<T>& dest, const Matrix<T>& srcA, const Matrix<T>& srcB)
{
    int rowsA = srcA.rows;
    int colsA = srcA.cols;
    int colsB = srcB.cols;

    constexpr int BLOCK = 64;

    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < rowsA; ib += BLOCK)
    {
        const int iBlock = std::min(BLOCK, rowsA - ib);

        for (int jb = 0; jb < colsB; jb += BLOCK)
        {
            const int jBlock = std::min(BLOCK, colsB - jb);

            for (int kb = 0; kb < colsA; kb += BLOCK)
            {
                const int kBlock = std::min(BLOCK, colsA - kb);

                const T* a = &srcA(ib, kb);
                const T* b = &srcB(kb, jb);
                T* c       = &dest(ib, jb);

                baseLineKernelV2(
                    a,
                    b,
                    c,
                    colsA,
                    colsB,
                    iBlock,
                    jBlock,
                    kBlock
                );
            }
        }
    }
}


template <>
void Matrix<double>::matMulV6(Matrix<double>& dest, const Matrix<double>& srcA, const Matrix<double>& srcB)
{
    int rowsA = srcA.rows;
    int colsA = srcA.cols;
    int colsB = srcB.cols;

    constexpr int BLOCK = 64;

    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < rowsA; ib += BLOCK)
    {
        const int iBlock = std::min(BLOCK, rowsA - ib);

        for (int jb = 0; jb < colsB; jb += BLOCK)
        {
            const int jBlock = std::min(BLOCK, colsB - jb);

            for (int kb = 0; kb < colsA; kb += BLOCK)
            {
                const int kBlock = std::min(BLOCK, colsA - kb);

                const double* a = &srcA(ib, kb);
                const double* b = &srcB(kb, jb);
                double* c       = &dest(ib, jb);

                Matrix<double>::kernelSIMD256(
                    a,
                    b,
                    c,
                    colsA,
                    colsB,
                    iBlock,
                    jBlock,
                    kBlock
                );
            }
        }
    }
}

template <>
void Matrix<double>::matMulV7(Matrix<double>& dest, const Matrix<double>& srcA, const Matrix<double>& srcB)
{
    int rowsA = srcA.rows;
    int colsA = srcA.cols;
    int colsB = srcB.cols;

    constexpr int BLOCK = 64;
    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < rowsA; ib += BLOCK)
    {
        const int iBlock = std::min(BLOCK, rowsA - ib);

        for (int jb = 0; jb < colsB; jb += BLOCK)
        {
            const int jBlock = std::min(BLOCK, colsB - jb);

            for (int kb = 0; kb < colsA; kb += BLOCK)
            {
                const int kBlock = std::min(BLOCK, colsA - kb);

                const double* a = &srcA(ib, kb);
                const double* b = &srcB(kb, jb);
                double* c       = &dest(ib, jb);

                Matrix<double>::kernelSIMD512(
                    a,
                    b,
                    c,
                    colsA,
                    colsB,
                    iBlock,
                    jBlock,
                    kBlock
                );
            }
        }
    }
}

template <>
void Matrix<double>::matMulV8(Matrix<double>& dest, const Matrix<double>& srcA, const Matrix<double>& srcB)
{
    int rowsA = srcA.rows;
    int colsA = srcA.cols;
    int colsB = srcB.cols;

    constexpr int BLOCK = 64;
    #pragma omp parallel for schedule(static)
    for (int ib = 0; ib < rowsA; ib += BLOCK)
    {
        const int iBlock = std::min(BLOCK, rowsA - ib);

        for (int jb = 0; jb < colsB; jb += BLOCK)
        {
            const int jBlock = std::min(BLOCK, colsB - jb);

            for (int kb = 0; kb < colsA; kb += BLOCK)
            {
                const int kBlock = std::min(BLOCK, colsA - kb);

                const double* a = &srcA(ib, kb);
                const double* b = &srcB(kb, jb);
                double* c       = &dest(ib, jb);

                Matrix<double>::kernelSIMD512V2(
                    a,
                    b,
                    c,
                    colsA,
                    colsB,
                    iBlock,
                    jBlock,
                    kBlock
                );
            }
        }
    }
}


// template <>
// void Matrix<float>::matMulSIMD256(Matrix<float>& dest, const Matrix<float>& A, const Matrix<float>& B)
// {
//     __m512 vec;

//     // ...
// }

// Explicitly instantiate Matrix for the supported floating-point types
// to allow the template implementations to remain in Matrix.cpp
template class Matrix<float>;
template class Matrix<double>;