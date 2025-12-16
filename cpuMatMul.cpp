#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>


class MatrixFP32{
    public: 
    const int n_rows;
    const int n_cols;

    float *ptr;
    const bool on_device;

    MatrixFP32(int n_rows, int n_cols);

    void Free_Mat();
};

MatrixFP32::MatrixFP32(int n_rows_, int n_cols) : n_rows(n_rows_), n_cols(n_cols){
    if (on_device == false ) {

    ptr = new float[n_rows * n_cols];
    } else {
        cudaError_t err = cudaMalloc((void**) &ptr, n_rows * n_cols * sizeof(float));
    }
}

void MatrixFP32::Free_Mat(){

    if (on_device == false)
     delete[] ptr;
} else {
    cudaFree(ptr);
}

void MatrixFP32::DeviceCopy(MatrixFP32 d_Mat){
assert(on_device == false && "Matrix must be in host memory");
assert(d_mat.on_device == true && "Input Matrix to this function must be in device memory");

cudaError_t err = cudaMemcpy(d_mat.ptr, ptr, n_rows*n_cols*sizeof(float), cudaMemcpyHostToDevice);
cuda_check(err);
}

void MatrixFP32::copy_to_host(MatrixFP32 h_mat)
{

    assert(on_device == true && "Matrix must be in device memory");
    assert(h_mat.on_device == false && "Input Matrix to this function must be in host memory");

    
    cudaError_t err = cudaMemcpy(h_mat.ptr, ptr, n_rows*n_cols*sizeof(float), cudaMemcpyDeviceToHost);
    cuda_check(err);
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





