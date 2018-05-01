#include "../common/common.h"
#include <stdio.h>

__global__ void helloOnGPU()
{
    printf("Hello World on GPU!\n");
}

int main(int argc, char **argv)
{
    printf("Hello World on CPU!\n");

    helloOnGPU<<<1, 10>>>();
    CHECK(cudaDeviceReset());
    return 0;
}


