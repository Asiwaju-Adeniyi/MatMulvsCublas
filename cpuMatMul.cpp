#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>


class MatrixFP32{
    public: 
    const int n_rows;
    const int n_cols;

    float *ptr;

    MatrixFP32(int n_rows, int n_cols);

    void Free_Mat();
};

MatrixFP32::MatrixFP32(int n_rows_, int n_cols) : n_rows(n_rows_), n_cols(n_cols){
    ptr = new float[n_rows * n_cols];
}

void MatrixFP32::Free_Mat(){
     delete[] ptr;
}


void cpuMatMul(MatrixFP32 matA, MatrixFP32 matB, MatrixFP32 matC) {
    int rowA = matA.n_rows;
    int colA = matA.n_cols;

    int rowB = matB.n_rows;
    int colB = matB.n_cols;

    int rowC = matC.n_rows;
    int colC = matC.n_cols;

    assert (colA == rowB && "cols in A should equal rows in B.");
    assert (rowA == rowC && "rows in A should equal rows in C.");
    assert (colB == colC && "cols in B should equal cols in C.");


    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
            float accum = 0.0f;

            for (int k = 0; k < colA; k++) {
                accum += matA.ptr[i * colA + k] * matB.ptr[j * colB + k];
            }

            matC.ptr[i * colC + j] = accum;
        }
    }
}





