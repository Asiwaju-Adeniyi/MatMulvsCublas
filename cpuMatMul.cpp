#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

class MatrixFP32{
    public: 
    const int n_rows;
    const int n_cols;
    bool to_dev;

    MatrixFP32(const int n_rows, const int n_cols, bool to_dev);
    ~MatrixFP32();

    float *ptr;

    void dev2copy(MatrixFP32 d_mat);
    void host2copy(MatrixFP32 h_mat);

    
    void free_Mat();
};

MatrixFP32::~MatrixFP32() {
    if (to_dev) cudaFree(ptr);
    else delete[] ptr;
};

MatrixFP32::MatrixFP32(const int n_rows_, const int n_cols, bool to_dev_) : n_rows(n_rows_), n_cols(n_cols), to_dev(to_dev_){
    if (to_dev == false) {
   ptr = new float[n_rows* n_cols];
    } else {
        cudaMalloc((void**)&ptr, n_rows * n_cols*sizeof(float));
    }
};

void MatrixFP32::free_Mat() {
    if (to_dev == false) {
        delete[] ptr;
    } else {
        cudaFree(ptr);
    }

}

void MatrixFP32::dev2copy(MatrixFP32 d_mat) {
    cudaMemcpy(d_mat.ptr, ptr, n_rows * n_cols * sizeof(float), cudaMemcpyHostToDevice);
    
}

void MatrixFP32::host2copy(MatrixFP32 h_mat) {
    cudaMemcpy(h_mat.ptr, ptr, n_rows * n_cols * sizeof(float), cudaMemcpyDeviceToHost);
}






