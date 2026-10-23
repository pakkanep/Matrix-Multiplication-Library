#pragma once
#include <vector>
#include <iostream>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
typedef double double8_t __attribute__ ((vector_size (8 * sizeof(double))));

template <typename T>
class Matrix
{
    public:
    int rows;
    int cols;
    std::vector<T> data;

    Matrix(int rows, int cols)
        : rows(rows),
          cols(cols),
          data(rows * cols)
    {
    }

    T& operator()(int row, int col)
    {
        return data[this->cols * row + col];
    }

    const T& operator()(int row, int col) const
    {
        return data[this->cols * row + col];
    }

    T* scalarData(int row)
    {
        return data.data() + row * this->cols;
    }

    const T* scalarData(int row) const
    {
        return data.data() + row * this->cols;
    }

    void printScalar() const;
    void Set(int col, int row, T v);
    T getItem(int col, int row) const;
    void initToZero();


    static void baseLineKernelV1(
    const T* a,
    const T* mb,
    T* c,
    int Acols,
    int Bcols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    static void baseLineKernelV2(
    const T* a,
    const T* mb,
    T* c,
    int Acols,
    int Bcols,
    int blockRows,
    int blockDepth,
    int blockCols
    );


    static void kernelSIMD256(
        const T* srcA,
        const T* srcB,
        T* dest,
        int Arows,
        int Bcols,
        int iBlock,
        int jBlock,
        int kBlock
    );

    static void kernelSIMD512(
        const T* srcA,
        const T* srcB,
        T* dest,
        int Arows,
        int Bcols,
        int iBlock,
        int jBlock,
        int kBlock
    );
    static void kernelSIMD512V2(
        const T* srcA,
        const T* srcB,
        T* dest,
        int Arows,
        int Bcols,
        int iBlock,
        int jBlock,
        int kBlock
    );


    static void        matMul(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV1(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV2(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV3(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV4(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV5(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV6(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV7(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV8(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    
};
