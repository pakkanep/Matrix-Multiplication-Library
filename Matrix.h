#pragma once
#include <vector>
#include <iostream>
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

class Matrix
{
    static constexpr int nb = 4;

    int rows;
    int cols;
    int na;

    std::vector<double> data;
    std::vector<double4_t> vecData;

    public:
    Matrix(int rows, int cols)
        : rows(rows),
          cols(cols),
          na( (cols + nb - 1 ) / nb ),
          data(rows * cols, 0.0),
          vecData( (rows * na ), double4_t{0.0, 0.0, 0.0, 0.0})
    {
    }


        // Non-const version
    double& operator()(int row, int col)
    {
        return data[this->cols * row + col];
    }

    // Const version
    const double& operator()(int row, int col) const
    {
        return data[this->cols * row + col];
    }

    double* scalarData(int row)
    {
        return data.data() + row * this->cols;
    }

    const double* scalarData(int row) const
    {
        return data.data() + row * this->cols;
    }

    double4_t* vectorData(int row)
    {
        return vecData.data() + row * this->na;
    }

    void Set(int col, int row, double v);
    void SetVec(int col, int row, int i, double val);

    double getItem(int col, int row) const;

    double getVecItem(int col, int row, int i) const;


    static void MOV(Matrix& dest, Matrix& src);

    static void scalarBlock(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols,
    const Matrix &dest
    );

    static void scalarBlockV2(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    static void scalarBlockV3(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    
    static void scalarBlockV4(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    static void avxBlock(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    static void avxBlockV2(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    static void avx512Block(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    template<int BcolsReg, int ArowsReg, int BcolsCache, int AcolsCache>
    static void cacheAwareAvxBlock(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols
    );

    static void kernel1(
        int rows,
        int cols,
        int iBlock,
        int jBlock,
        int kBlock,
        const double* srcA,
        const double* srcB,
        double* dest,
        int an, int bn, int cn
    );
    static void kernel2(
        int rows,
        int cols,
        int iBlock,
        int jBlock,
        int kBlock,
        const double* srcA,
        const double* srcB,
        double* dest
    );


    static void        matMulV4(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV5(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV6(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV7(Matrix& dest, const Matrix& srcA, const Matrix& srcB);
    static void        matMulV8(Matrix& dest, const Matrix& srcA);
    static void        matMulV9(Matrix& dest, const Matrix& srcA);
    static void        matMulV10(Matrix& dest, const Matrix& srcA);

    void printScalar() const;
    void printVec()    const;
};
