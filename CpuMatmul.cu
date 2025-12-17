#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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
