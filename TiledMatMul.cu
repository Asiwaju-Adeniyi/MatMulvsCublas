#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

void __global__ matmul_tiled(float *d_A, float *d_B, float *d_C, int CnCol, int CnRow, int N) {
    assert(TILE_WIDTH == blockDim.y);
    assert(TILE_WIDTH == blockDim.x);

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = bx * TILE_WIDTH + ty;
    int col = by * TILE_WIDTH + tx;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    const int phases = ceil((float)N/TILE_WIDTH);

    float value = 0.0f;

    for (int phase = 0; phase < phases; phase++) {
        if ((row < CnRow) && ((phase * TILE_WIDTH + tx) < N)) {
            sh_A[ty][tx] = d_A[row * N + (phase * TILE_WIDTH + tx)]; 
        } else {
            sh_A[ty][tx] = 0.0f;
        }
        if ((col < CnCol) && ((phase * TILE_WIDTH + ty) < N)) {
            sh_B[ty][tx] = d_B[(phase * TILE_WIDTH + ty) * N + col];
        } else {
            sh_B[ty][tx] = 0.0f;

            __syncthreads();


            for (int k_phase = 0; k_phase < TILE_WIDTH; k_phase++) {
                value += sh_A[ty][k_phase] * sh_B[k_phase][tx];
            }
        }

        d_C[row * N + col] = 1 * value + (0 * d_C[row * N + col]);
    }

}
