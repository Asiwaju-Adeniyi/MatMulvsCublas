#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>


class MatSPre {
public:
    const int n_cols;
    const int n_rows;
    float* ptr;

    MatSPre(int n_rows, int n_cols);
    ~MatSPre();  
};

MatSPre::MatSPre(int n_rows_, int n_cols_)
    : n_rows(n_rows_), n_cols(n_cols_)
{
    ptr = new float[n_rows * n_cols];
}

MatSPre::~MatSPre() {
    delete[] ptr;
}

void cpu_matmul(MatSPre matA, MatSPre matB, MatSPre matC) {
    int n_rowsA = matA.n_rows;
    int n_colsA = matA.n_cols;

    int n_rowsB = matB.n_rows;
    int n_colsB = matB.n_cols;

    int n_rowsC = matC.n_rows;
    int n_colsC = matC.n_cols;

    for (int row = 0; row < n_rowsA; row++) {
        for (int col = 0; col < n_rowsB; col++) {
            float accum = 0.0f;
            for (int k = 0; k < n_colsB; k++) {
                accum += matA.ptr[row*n_colsA+k] * matB.ptr[k * n_colsB + col];
            }

            matC.ptr[row * n_colsC + col] = accum;
        }
    }
}




