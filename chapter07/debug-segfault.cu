#include "../common/common.h"
#include <stdio.h>
 
#define N   1025
#define M   12

__device__ int foo(int row, int col)
{
    return (2 * row);
}

__global__ void kernel(int **arr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
 
    for ( ; tid < N; tid++)
    {
        for (i = 0; i < M; i++)
        {
            arr[tid][i] = foo(tid, i);
        }
    }
}

int main(int argc, char **argv)
{
    int i; 
    int **h_matrix; 
    int **d_ptrs; 
    int **d_matrix;

    h_matrix = (int **)malloc(N * sizeof(int *));
    d_ptrs = (int **)malloc(N * sizeof(int *));
    CHECK(cudaMalloc((void **)&d_matrix, N * sizeof(int *)));
    CHECK(cudaMemset(d_matrix, 0x00, N * sizeof(int *)));
 
    for (i = 0; i < N; i++)
    {
        h_matrix[i] = (int *)malloc(M * sizeof(int));
        CHECK(cudaMalloc((void **)&d_ptrs[i], M * sizeof(int)));
        CHECK(cudaMemset(d_ptrs[i], 0x00, M * sizeof(int)));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_matrix);
 
    for (i = 0; i < N; i++)
    {
        CHECK(cudaMemcpy(h_matrix[i], d_ptrs[i], M * sizeof(int),
                        cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_ptrs[i]));
        free(h_matrix[i]);
    }

    CHECK(cudaFree(d_matrix));
    free(h_matrix);

    return 0;
}
