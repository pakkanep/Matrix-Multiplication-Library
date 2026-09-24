#pragma once
#include <vector>
#include <iostream>
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

template <typename T>
class Matrix
{
    public:
    int rows;
    int cols;
    std::vector<T> data;

    //___vectorized___
    // int na;
    // static constexpr int nb = 4;
    // std::vector<double4_t> vecData; // broken beacuse template

    Matrix(int rows, int cols)
        : rows(rows),
          cols(cols),
          data(rows * cols)
          // na( (cols + nb - 1 ) / nb ),
          // vecData( (rows * na ), double4_t{0.0, 0.0, 0.0, 0.0})
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

    
    // ____vectorized____
    // double4_t* vectorData(int row)
    // {
    //     return vecData.data() + row * this->na;
    // }

    // void SetVec(int col, int row, int i, double val);
    // double getVecItem(int col, int row, int i) const;
    // void printVec()    const;
        
    
    static void baseLineKernelV1(
    const T* a,
    const T* mb,
    T* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    static void baseLineKernelV2(
    const T* a,
    const T* mb,
    T* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    static void kernel(
        const T* srcA,
        const T* srcB,
        T* dest,
        int rows,
        int cols,
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
    
};
