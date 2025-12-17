#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


__global__ void naivematmul(float *d_A, float *d_B, float *d_C, int CnCol, int CnRow, int N) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    const int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < CnRow && col < CnCol) {
        float accum = 0.0f;
        for (int k = 0; k < N; k++) {
            accum += d_A[row * N + k] * d_B[k * N + col];
        }

        d_C[row * N + col] = 1 * accum + 0 * d_C[row * N + col];
    }
}
