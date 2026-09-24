#include "Matrix.h"
#include <iostream>
#include <iomanip>

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

// void Matrix::printVec() const
// {
//     constexpr int width = 8;
//     std::cout << std::fixed
//               << std::showpos
//               << std::setprecision(8);

//     for (int row = 0; row < rows; ++row)
//     {
//         for (int vec = 0; vec < na; ++vec)
//         {
//             std::cout << "[";
//             for (int i = 0; i < nb; ++i)
//             {
//                 std::cout << ' ' << getVecItem(row, vec, i);
//             }
//             std::cout << "]";
//         }
//         std::cout << '\n';
//     }
//     std::cout << '\n';
//     std::cout << std::noshowpos << std::defaultfloat;
// }

// template <typename T>
// void Matrix<T>::SetVec(int row, int col, int i, T val)
// {
//     vecData[this->na * row + col][i] = val;
// }


// template <typename T>
// T Matrix<T>::getVecItem(int row, int col, int i) const
// {
//     return vecData[this->na * row + col][i];
// }


// baseline kernel
template <typename T>
void Matrix<T>::baseLineKernelV1(
    const T* a,
    const T* mb,
    T* c,
    int Bcols,
    int Acols,
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
    int Bcols,
    int Acols,
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


// baseline kernel with a different access pattern
template <typename T>
void Matrix<T>::kernel(
    const T* srcA,
    const T* srcB,
    T* dest,
    int colsA,
    int colsB,
    int iBlock,
    int jBlock,
    int kBlock
)
{
    for (int i = 0; i < iBlock; ++i)
    {
        const T* A_ = srcA + i * colsA;
        T* C_       = dest + i * colsB;

        for (int k = 0; k < kBlock; ++k)
        {
            const T a_ik = A_[k];
            const T* B_ = srcB + k * colsB;

            for (int j = 0; j < jBlock; ++j)
            {
                C_[j] += a_ik * B_[j];
            }
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

                Matrix<T>::kernel(
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


// Explicitly instantiate Matrix for the supported floating-point types
// to allow the template implementations to remain in Matrix.cpp
template class Matrix<float>;
template class Matrix<double>;