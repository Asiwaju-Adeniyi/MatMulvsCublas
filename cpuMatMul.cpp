#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#define cuda_check(call)                                   \
do {                                                       \
    cudaError_t err = call;                                \
    if (err != cudaSuccess) {                              \
        fprintf(stderr,                                   \
                "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__,                        \
                cudaGetErrorString(err));                  \
        exit(EXIT_FAILURE);                                \
    }                                                      \
} while (0)



class MatrixFP32 {
public:
    const int n_rows;
    const int n_cols;
    const bool on_device;
    float* ptr;

    MatrixFP32(int rows, int cols, bool device);
    ~MatrixFP32();

    void copy_to_device(MatrixFP32& d_mat);
    void copy_to_host(MatrixFP32& h_mat);
};

MatrixFP32::MatrixFP32(int r, int c, bool device)
    : n_rows(r), n_cols(c), on_device(device) {
    if (on_device)
        cuda_check(cudaMalloc(&ptr, r * c * sizeof(float)));
    else
        ptr = new float[r * c];
}

MatrixFP32::~MatrixFP32() {
    if (on_device) cudaFree(ptr);
    else delete[] ptr;
};


void cpuMatMul(const MatrixFP32& matA,
               const MatrixFP32& matB,
               MatrixFP32& matC)
{
    assert(!matA.on_device && !matB.on_device && !matC.on_device);

    for (int i = 0; i < matA.n_rows; i++) {
        for (int j = 0; j < matB.n_cols; j++) {
            float accum = 0.0f;
            for (int k = 0; k < matA.n_cols; k++) {
                accum += matA.ptr[i * matA.n_cols + k] *
                         matB.ptr[k * matB.n_cols + j];
            }
            matC.ptr[i * matC.n_cols + j] = accum;
        }
    }
}






