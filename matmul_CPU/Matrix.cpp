#include "Matrix.h"
#include <iostream>
#include <iomanip>
#include <immintrin.h>


void Matrix::printScalar() const
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

void Matrix::printVec() const
{
    constexpr int width = 8;
    std::cout << std::fixed
              << std::showpos
              << std::setprecision(8);

    for (int row = 0; row < rows; ++row)
    {
        for (int vec = 0; vec < na; ++vec)
        {
            std::cout << "[";
            for (int i = 0; i < nb; ++i)
            {
                std::cout << ' ' << getVecItem(row, vec, i);
            }
            std::cout << "]";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
    std::cout << std::noshowpos << std::defaultfloat;
}

void Matrix::Set(int row, int col, double v)
{
    data[this->cols * row + col] = v;
}

double Matrix::getItem(int row, int col) const 
{
    return data[this->cols * row + col];
}

void Matrix::SetVec(int row, int col, int i, double val)
{
    vecData[this->na * row + col][i] = val;
}

double Matrix::getVecItem(int row, int col, int i) const 
{
    return vecData[this->na * row + col][i];
}

void Matrix::scalarBlock(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols,
    const Matrix &dest
)
{
    for (int i = 0; i < blockRows; ++i, c += Bcols, a += Acols)
    {
        const double* b = mb;
        for (int k = 0; k < blockDepth; ++k, b += Bcols)
        {
            const double a_ik = a[k];
            for (int j = 0; j < blockCols; ++j)
            {
                c[j] += a_ik * b[j];
            }
        }
    }
}

void Matrix::scalarBlockV2(
    const double* a,    // pointer to a data row in A
    const double* mb,   // pointer to a data row in B
    double* c,          // pointer to the result row of C 
    int Bcols,          // length of row or number of cols in B 
    int Acols,          // ----|||-----
    int blockRows,      // std::min(BLOCK, Arows - ib)
    int blockDepth,     // std::min(BLOCK, Acols - kb)
    int blockCols       // std::min(BLOCK, Bcols - jb)
)
{
    const int CREGS = 64;
    int barrier = std::min(CREGS, blockCols);    

    for (int i = 0; i < blockRows; ++i, c += Bcols, a += Acols)
    {
        const double* b = mb;
        double c_temp[CREGS] = {0.0};

        for (int k = 0; k < blockDepth; ++k, b += Bcols)
        {
            const double a_ik = a[k];

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


void Matrix::avxBlock(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols)
{
    constexpr int regWidth = 4;
    for (int i = 0; i < blockRows; ++i, c += Bcols, a += Acols)
    {
        const double* b = mb;

        for (int k = 0; k < blockDepth; ++k, b += Bcols)
        {
            __m256d aReg = _mm256_broadcast_sd( &a[ k ] );

            int j = 0;
            for ( ; j + regWidth <= blockCols; j += regWidth)
            {
                __m256d bReg = _mm256_loadu_pd( &b[ j ] );
                __m256d cReg = _mm256_loadu_pd( &c[ j ] );
                cReg = _mm256_fmadd_pd( aReg, bReg, cReg );
                _mm256_storeu_pd( &c[j], cReg );
            }

            for ( ; j < blockCols; ++j)
            {
                c[j] += a[k] * b[j];
            }
        }
    }
}

void Matrix::avxBlockV2(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols)
{
    constexpr int regWidth = 4;

    const int CREGS = 32;
    int barrier = std::min(CREGS, blockCols);    

    for (int i = 0; i < blockRows; ++i, c += Bcols, a += Acols)
    {
        const double* b = mb;
        double c_temp[CREGS] = {0.0};

        for (int k = 0; k < blockDepth; ++k, b += Bcols)
        {
            __m256d aReg = _mm256_broadcast_sd( &a[ k ] );

            int j = 0;
            for ( ; j + regWidth <= barrier; j += regWidth)
            {
                __m256d bReg = _mm256_loadu_pd( &b[ j ] );
                __m256d cReg = _mm256_loadu_pd( &c[ j ] );
                cReg = _mm256_fmadd_pd( aReg, bReg, cReg );
                _mm256_storeu_pd( &c_temp[j], cReg );
            }

            for ( ; j < barrier; ++j)
            {
                c_temp[j] += a[k] * b[j];
            }

        }
        for (int j = 0 ; j < barrier; ++j)
        {
            c[j] += c_temp[j];
        }
    }
}


void Matrix::avx512Block(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols)
{
    constexpr int regWidth = 8;
    for (int i = 0; i < blockRows; ++i, c += Bcols, a += Acols)
    {
        const double* b = mb;

        for (int k = 0; k < blockDepth; ++k, b += Bcols)
        {
            const __m512d aReg = _mm512_set1_pd(a[k]);
            

            int j = 0;
            for ( ; j + regWidth <= blockCols; j += regWidth)
            {
                const __m512d bReg = _mm512_loadu_pd(&b[j]);
                __m512d cReg = _mm512_loadu_pd(&c[j]);

                cReg = _mm512_fmadd_pd(aReg, bReg, cReg);

                _mm512_storeu_pd(&c[j], cReg);
            }

            for ( ; j < blockCols; ++j)
            {
                c[j] += a[k] * b[j];
            }
        }
    }
}

template<int BcolsReg, int ArowsReg, int AcolsCache, int BcolsCache> 

void Matrix::cacheAwareAvxBlock(
    const double* a,
    const double* mb,
    double* c,
    int Bcols,
    int Acols,
    int blockRows,
    int blockDepth,
    int blockCols)
{
    constexpr int regWidth = 4;

    for (int i = 0; i < ArowsReg; ++i, c += Bcols, a += Acols)
    {
        const double* b = mb;

        for (int k = 0; k < AcolsCache; ++k, b += Bcols)
        {
            __m256d aReg = _mm256_broadcast_sd( &a[ k ] );

            int j = 0;
            for ( ; j + regWidth <= BcolsReg; j += regWidth)
            {
                __m256d bReg = _mm256_loadu_pd( &b[ j ] );
                __m256d cReg = _mm256_loadu_pd( &c[ j ] );
                cReg = _mm256_fmadd_pd( aReg, bReg, cReg );
                _mm256_storeu_pd( &c[j], cReg );
            }

            for ( ; j < blockCols; ++j)
            {
                c[j] += a[k] * b[j];
            }
        }
    }
}


void Matrix::kernel(
    int colsA,
    int colsB,
    int iBlock,
    int jBlock,
    int kBlock,
    const double* srcA,
    const double* srcB,
    double* dest
)
{
    for (int i = 0; i < iBlock; ++i)
    {
        const double* A_ = srcA + i * colsA;
        double* C_ = dest + i * colsB;

        for (int k = 0; k < kBlock; ++k)
        {
            const double a = A_[k];
            const double* B_ = srcB + k * colsB;

            for (int j = 0; j < jBlock; ++j)
            {
                C_[j] += a * B_[j];
            }
        }
    }
}



void Matrix::matMulV1(Matrix& dest, const Matrix& srcA, const Matrix& srcB)
{
    int Arows = srcA.rows;
    int Acols = srcA.cols;
    int Bcols = srcB.cols;
    
    for (int row = 0; row < Arows; ++row)
    {
        for (int k = 0; k < Acols; ++k)
        {
            auto A_ik = srcA(row, k);
            for (int col = 0; col < Bcols; ++col)
            {
                dest(row, col) += A_ik * srcB(k, col);
            }
        }
    }
}

void Matrix::matMulV2(Matrix& dest, const Matrix& srcA, const Matrix& srcB)
{
    const int Arows = srcA.rows;
    const int Acols = srcA.cols;
    const int Bcols = srcB.cols;
    
    constexpr int BLOCK = 32;
    
    for (int ib = 0; ib < Arows; ib += BLOCK)
    {
        
        const int blockRows = std::min(BLOCK, Arows - ib);
        for (int kb = 0; kb < Acols; kb += BLOCK)
        {
            
            const int blockDepth = std::min(BLOCK, Acols - kb);
            for (int jb = 0; jb < Bcols; jb += BLOCK)
            {
                
                const int blockCols = std::min(BLOCK, Bcols - jb);
                const double* a = &srcA(ib, kb);
                const double* b = &srcB(kb, jb);
                double*       c = &dest(ib, jb);
                
                scalarBlock(
                    a,
                    b,
                    c,
                    Bcols,
                    Acols,
                    blockRows,
                    blockDepth,
                    blockCols,
                    dest
                );
                //dest.printScalar();
            }
        }
    }
}

void Matrix::matMulV3(Matrix& dest, const Matrix& srcA, const Matrix& srcB)
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
                
                const double* a = &srcA(ib, kb);
                const double* b = &srcB(kb, jb);
                double* c =       &dest(ib, jb);
                
                scalarBlockV2(
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

void Matrix::matMulV4(Matrix& dest, const Matrix& srcA, const Matrix& srcB)
{
    const int Arows = srcA.rows;
    const int Acols = srcA.cols;
    const int Bcols = srcB.cols;
    
    constexpr int ArowsCache = 180;
    constexpr int AcolsCache = 240;
    constexpr int BcolsCache = 96;
    constexpr int BcolsReg = 12;
    constexpr int ArowsReg = 4;
    
    for (int ib = 0; ib < Arows; ib += ArowsCache)
    {
        const int blockRows = std::min(ArowsCache, Arows - ib);
        
        for (int kb = 0; kb < Acols; kb += AcolsCache)
        {
            const int blockDepth = std::min(AcolsCache, Acols - kb);
            
            for (int jb = 0; jb < Bcols; jb += BcolsCache)
            {
                const int blockCols = std::min(BcolsCache, Bcols - jb);
                
                const double* ma = &srcA(ib, kb);
                const double* mb = &srcB(kb, jb);
                double*       mc = &dest(ib, jb);
                
                for (int i2 = 0; i2 < ArowsCache; i2 += ArowsReg)
                {
                    for (int j2 = 0; j2 < BcolsCache; j2 += BcolsReg)
                    {
                        const double* a = &ma[i2 * Acols];
                        const double* b = &mb[j2];
                        double*       c = &mc[i2 * Bcols + j2];
                        cacheAwareAvxBlock<BcolsReg, ArowsReg, AcolsCache, BcolsCache>(
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
    }
}

void Matrix::matMulV5(Matrix& dest, const Matrix& srcA)
{
    int rows = srcA.rows;
    int cols = srcA.cols;
    
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < rows; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < cols; ++k)
            {
                sum += srcA(i, k) * srcA(j, k);
            }
            dest.Set(i, j, sum);
        }
    }
}

void Matrix::matMulV6(Matrix& dest, const Matrix& srcA, const Matrix& srcB)
{
    int rowsA = srcA.rows;
    int colsA = srcA.cols;
    int colsB = srcB.cols;
    constexpr int BLOCK = 64;
    
    #pragma omp parallel for schedule(dynamic)
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
                double*       c = &dest(ib, jb); 

                Matrix::kernel
                (
                    colsA,
                    colsB,
                    iBlock, 
                    jBlock, 
                    kBlock, 
                    a, 
                    b, 
                    c
                );
            }
        }
    }
}
