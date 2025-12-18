#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void coalescedkernel(float *d_A, float *d_B, float *d_C, int CnCol, int CnRow, int N) {
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < CnRow && col < CnCol){
        float value = 0.0f;
        for (int k = 0; k < N; k++) {
            value += d_A[row * N + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = 1*value + (0 * d_C[row * N + col]);
    }

}
