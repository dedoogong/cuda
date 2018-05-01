#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
 

__global__ void checkIndex(void)
{
    printf("threadIdx:(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockIdx:(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);

    printf("blockDim:(%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("gridDim:(%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);

}

int main(int argc, char **argv)
{ 
    int nElem = 6;
 
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
 
    printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
    printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
 
    checkIndex<<<grid, block>>>();
 
    CHECK(cudaDeviceReset());

    return(0);
}
