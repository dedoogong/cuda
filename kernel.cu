//build commnad : nvcc -std=c++11 -lcudnn -lcublas kernel.cu -o kernel

#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include "common/common.h"
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

///////////////////////////////////////////////////////////////////////

__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64];
  int t = threadIdx.x;
  int trail = n-t-1;
  s[t] = d[t];// write step : global to shared
  __syncthreads();
  d[t] = s[trail]; // read step : shared to global
}

__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];
  int t = threadIdx.x;
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int shared_memory_reverse(void)
{
  const int n = 64;
  int h_a[n], h_r[n], h_d[n];

  for (int i = 0; i < n; i++) {
    h_a[i] = i;
    h_r[i] = n-i-1;
    h_d[i] = 0;
  }
  printf("original array elemtns\n");

  for (int i = 0; i < n; i++) {
      printf("%d ",a[i]);
  }

  int *d_d;
  cudaMalloc(&d_d, n * sizeof(int));

  // 정적 공유 메모리 버전
  cudaMemcpy(d_d, h_a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n);
  cudaMemcpy(h_d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

  // 동적 공유 메모리 버전
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  printf("\nreverse results\n");
  int flag=1;
  for (int i = 0; i < n; i++)
    if (d[i] != r[i]){ flag=0; printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);}
    else printf("%d ",r[i]);
  if(flag)printf("\nall array elements are correctly reversed\n");

}


///////////////////////////////////////////////////////////////////////

__global__ void kernel(float *a, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  float x = (float)i;
  float s = sinf(x);
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s*s+c*c);
}

float maxError(float *a, int n)
{
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}


int overlap(int argc, char **argv)
{
  const int blockSize = 256, nStreams = 4;// blockSize=threadCount
  const int n = 4 * 1024 * blockSize * nStreams;
  const int streamSize = n / nStreams;// == one stream size == 4 * 1024 * blockSize
  const int streamBytes = streamSize * sizeof(float);
  const int total_bytes = n * sizeof(float);

  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );

  // 호스트  고정 메모리와 디바이스 메모리 할당
  float *a, *d_a;
  checkCuda( cudaMallocHost((void**)&a, total_bytes) );      // host pinned
  checkCuda( cudaMalloc((void**)&d_a, total_bytes) ); // device

  float ms; // milliseconds 타이머

  // 이벤트 및 스트림 생성
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );

  // 기본 케이스 - 순차적 메모리 전송과 커널 호출
  memset(a, 0, total_bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  checkCuda( cudaMemcpy(d_a, a, total_bytes, cudaMemcpyHostToDevice) );
  kernel<<<n/blockSize, blockSize>>>(d_a, 0);//gridSize=4*1024(blockCount)
  checkCuda( cudaMemcpy(a, d_a, total_bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // 비동기 버전 1: [복사-커널호출-복사]를 루프로 반복 수행
  memset(a, 0, total_bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i) {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset],
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // 비동기 버전 2:
  // 복사 루프, 커널 호출 루프, 복사 루프를 별개로 수행
  memset(a, 0, total_bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&d_a[offset], &a[offset],
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[i]) );
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
  }
  for (int i = 0; i < nStreams; ++i)
  {
    int offset = i * streamSize;
    checkCuda( cudaMemcpyAsync(&a[offset], &d_a[offset],
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[i]) );
  }
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // 메모리 해제
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < nStreams; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}

///////////////////////////////////////////////////////////////////////
void profileCopies(float        *h_a,
                   float        *h_b,
                   float        *d,
                   unsigned int  n,
                   char         *desc)
{
  printf("\n%s transfers\n", desc);

  unsigned int bytes = n * sizeof(float);

  cudaEvent_t startEvent, stopEvent;

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  float time;
  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Host to Device bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  checkCuda( cudaEventRecord(startEvent, 0) );
  checkCuda( cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );

  checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
  printf("  Device to Host bandwidth (GB/s): %f\n", bytes * 1e-6 / time);

  for (int i = 0; i < n; ++i) {
    if (h_a[i] != h_b[i]) {
      printf("*** %s transfers failed ***", desc);
      break;
    }
  }

  // 이벤트 해제
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}

int data_transfer_pageable_vs_pinned()
{
  unsigned int nElements = 4*1024*1024;
  const unsigned int bytes = nElements * sizeof(float);

  //호스트 배열
  float *h_aPageable, *h_bPageable;
  float *h_aPinned, *h_bPinned;

  //디바이 스  배열
  float *d_a;

  //할당 및 초기화
  h_aPageable = (float*)malloc(bytes);                    // 호스트 pageable 메모리 할당
  h_bPageable = (float*)malloc(bytes);                    // 호스트 pageable 메모리 할당
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes) ); // 호스트 pinned 메모리 할당
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes) ); // 호스트 pinned 메모리 할당
  checkCuda( cudaMalloc((void**)&d_a, bytes) );           // 디바이스 메모리 할당

  for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;
  memcpy(h_aPinned, h_aPageable, bytes);
  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, 0) );

  printf("\nDevice: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", bytes / (1024 * 1024));

  // 고정 메모리 전송 성능 비교
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
  profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
  profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

  printf("\n");

  // 메모리 해제
  cudaFree(d_a);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable);
  free(h_bPageable);

  return 0;
}


///////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void offset(T* a, int s)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x + s;
  a[i] = a[i] + 1;
}

template <typename T>
__global__ void stride(T* a, int s)
{
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * s;
  a[i] = a[i] + 1;
}

template <typename T>
void runTest(int deviceId, int nMB)
{
  int blockSize = 256;
  float ms;

  T *d_a;
  cudaEvent_t startEvent, stopEvent;

  int n = nMB*1024*1024/sizeof(T);

  // NB:  d_a(33*nMB)
  checkCuda( cudaMalloc(&d_a, n * 33 * sizeof(T)) );

  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );

  printf("Offset, Bandwidth (GB/s):\n");

  offset<<<n/blockSize, blockSize>>>(d_a, 0); // warm up

  for (int i = 0; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    offset<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  printf("\n");
  printf("Stride, Bandwidth (GB/s):\n");

  stride<<<n/blockSize, blockSize>>>(d_a, 1); // warm up
  for (int i = 1; i <= 32; i++) {
    checkCuda( cudaMemset(d_a, 0, n * sizeof(T)) );

    checkCuda( cudaEventRecord(startEvent,0) );
    stride<<<n/blockSize, blockSize>>>(d_a, i);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );

    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    printf("%d, %f\n", i, 2*nMB/ms);
  }

  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  cudaFree(d_a);
}
//------------------------------------------------------------------
int coaleascing(int argc, char **argv)
{
  int nMB = 4;
  int deviceId = 0;
  bool bFp64 = false;

  for (int i = 1; i < argc; i++) {
    if (!strncmp(argv[i], "dev=", 4))
      deviceId = atoi((char*)(&argv[i][4]));
    else if (!strcmp(argv[i], "fp64"))
      bFp64 = true;
  }

  cudaDeviceProp prop;

  checkCuda( cudaSetDevice(deviceId) );
  checkCuda( cudaGetDeviceProperties(&prop, deviceId) );
  printf("Device: %s\n", prop.name);
  printf("Transfer size (MB): %d\n", nMB);

  printf("%s Precision\n", bFp64 ? "Double" : "Single");

  if (bFp64) runTest<double>(deviceId, nMB);
  else       runTest<float>(deviceId, nMB);
}

///////////////////////////////////////////////////////////////////////
const int N = 1 << 20;

__global__ void kernel_target(float *x, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}
void *launch_kernel(void *dummy){
    float *data;
    cudaMalloc(&data, N * sizeof(float));
    kernel_target<<<1, 64>>>(data, N);
    cudaStreamSynchronize(0);
    return NULL;
}
int multithread(){
    const int num_threads = 8;
    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, launch_kernel, 0)) {
            fprintf(stderr, "Error creating threadn");
     }
   }
    for (int i = 0; i < num_threads; i++) {
        if(pthread_join(threads[i], NULL)) {
            fprintf(stderr, "Error joining threadn");
            return 2;
        }
    }
    cudaDeviceReset();
    return 0;
}

///////////////////////////////////////////////////////////////////////

int ROWS = 1024;
int COLS = 1024;

void generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rand_max = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    for (j = 0; j < N; j++){//열
        for (i = 0; i < M; i++){//행
            double drand = (double)rand();
            A[j * M + i] = (drand / rand_max) * 100.0; //0-100 사이 값
        }
    }
    *outA = A;
}

int cublasMM(int argc, char **argv)
{
    int i, j;
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    float beta;
    float alpha;
    cublasHandle_t handle = 0;

    alpha = 3.0f;
    beta = 4.0f;
    int N = ROWS;
    int M = COLS;
    // 입력 데이터 초기화
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);
    memset(C, 0x00, sizeof(float) * M * M);

    // cuBLAS 핸들러 생성
    CHECK_CUBLAS(cublasCreate(&handle));

    // 디바이스 메모리 할당
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * M));

    // 디바이스로 데이터 전송
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dA, M));
    CHECK_CUBLAS(cublasSetMatrix(N, M, sizeof(float), B, N, dB, N));
    CHECK_CUBLAS(cublasSetMatrix(M, M, sizeof(float), C, M, dC, M));

    // 행렬-벡터 곱 수행
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha,
                dA, M, dB, N, &beta, dC, M));

    // 결과 값 반환 및 확인
    CHECK_CUBLAS(cublasGetMatrix(M, M, sizeof(float), dC, M, C, M));

    for (j = 0; j < 10; j++)
    {
        for (i = 0; i < 10; i++)
        {
            printf("%2.2f ", C[j * M + i]);
        }
        printf("...\n");
    }

    printf("...\n");

    free(A);
    free(B);
    free(C);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}

///////////////////////////////////////////////////////////////////////

int cublasMMAsync(int argc, char **argv)
{
    int i, j;
    float *A, *dA;
    float *B, *dB;
    float *C, *dC;
    float beta;
    float alpha;
    cublasHandle_t handle = 0;
    cudaStream_t stream = 0;

    alpha = 3.0f;
    beta = 4.0f;
    int N = ROWS;
    int M = COLS;
    // 입력 데이터 초기화
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);
    memset(C, 0x00, sizeof(float) * M * M);

    // cuBLAS 핸들러 생성
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // 디바이스 메모리 할당
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * M));

    // 디바이스로 데이터 비동기 전송
    CHECK_CUBLAS(cublasSetMatrixAsync(M, N, sizeof(float), A, M, dA, M, stream));
    CHECK_CUBLAS(cublasSetMatrixAsync(N, M, sizeof(float), B, N, dB, N, stream));
    CHECK_CUBLAS(cublasSetMatrixAsync(M, M, sizeof(float), C, M, dC, M, stream));

    // 행렬-벡터 곱 수행
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha,
                dA, M, dB, N, &beta, dC, M));

    // 결과 값 반환 및 확인
    CHECK_CUBLAS(cublasGetMatrixAsync(M, M, sizeof(float), dC, M, C, M,
                stream));
    CHECK(cudaStreamSynchronize(stream));

    for (j = 0; j < 10; j++)
    {
        for (i = 0; i < 10; i++)
        {
            printf("%2.2f ", C[j * M + i]);
        }
        printf("...\n");
    }

    printf("...\n");

    free(A);
    free(B);
    free(C);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
    CHECK(cudaStreamDestroy(stream));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}

///////////////////////////////////////////////////////////////////////

__global__ void kernel(float *g_data, float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + value;
}

int checkResult(float *data, const int n, const float x)
{
    for (int i = 0; i < n; i++)
    {
        if (data[i] != x)
        {
            printf("Error! data[%d] = %f, ref = %f\n", i, data[i], x);
            return 0;
        }
    }

    return 1;
}

int async(int argc, char *argv[])
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    printf("> %s running on", argv[0]);
    printf(" CUDA device [%s]\n", deviceProps.name);

    int num = 1 << 24;
    int nbytes = num * sizeof(int);
    float value = 10.0f;

    // 호스트 메모리 할당
    float *h_a = 0;
    CHECK(cudaMallocHost((void **)&h_a, nbytes));
    memset(h_a, 0, nbytes);

    // 디바이스 메모리 할당
    float *d_a = 0;
    CHECK(cudaMalloc((void **)&d_a, nbytes));
    CHECK(cudaMemset(d_a, 255, nbytes));

    // 스레드 레이아웃 설정
    dim3 block = dim3(512);
    dim3 grid  = dim3((num + block.x - 1) / block.x);

    // 이벤트 핸들러 생성
    cudaEvent_t stop;
    CHECK(cudaEventCreate(&stop));

    // 비동기 메모리 복사 및 커널 호출(모두 스트림 0으로)
    CHECK(cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice));
    kernel<<<grid, block>>>(d_a, value);
    CHECK(cudaMemcpyAsync(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));

    // GPU 작업이 진행되는 동안 CPU도 작업 수행
    unsigned long int counter = 0;

    while (cudaEventQuery(stop) == cudaErrorNotReady) {
        counter++;
    }

    printf("CPU executed %lu iterations while waiting for GPU to finish\n",
           counter);

    bool bFinalResults = (bool) checkResult(h_a, num, value);

    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFree(d_a));

    CHECK(cudaDeviceReset());

    exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}


///////////////////////////////////////////////////////////////////////

#define BDIMX 32
#define BDIMY 32
#define IPAD  1

void printData(char *msg, int *in,  const int size)
{
    printf("%s: ", msg);

    for (int i = 0; i < size; i++)
    {
        printf("%5d", in[i]);
        fflush(stdout);
    }

    printf("\n");
    return;
}

__global__ void setRowReadRow (int *out)
{
    // 정적 공유 메모리
    __shared__ int tile[BDIMY][BDIMX]; // x, y

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx; // x, y

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x] ;// x, y
}

__global__ void setColReadCol (int *out)
{
    // 정적 공유 메모리
    __shared__ int tile[BDIMX][BDIMY]; // y, x

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;// y, x

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];// y, x
}

__global__ void setRowReadCol(int *out)
{
    // 정적 공유 메모리
    __shared__ int tile[BDIMY][BDIMX];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDyn(int *out)
{
    // 동적 공유 메모리
    extern  __shared__ int tile[];

    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    tile[row_idx] = row_idx;

    __syncthreads();

    out[row_idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out)
{
    // 정적 공유 메모리 패딩
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadColDynPad(int *out)
{
    // 동적 공유 메모리 패딩
    extern  __shared__ int tile[];

    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.x + IPAD) + threadIdx.y;

    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[row_idx] = g_idx;
    __syncthreads();
    out[g_idx] = tile[col_idx];
}


int smemSquare(int argc, char **argv)
{
    // 디바이스 설정
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig ( &pConfig ));
    printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

    // 배열 크기 설정(2048)
    int nx = BDIMX;
    int ny = BDIMY;

    bool iprintf = 0;

    if (argc > 1) iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    // 실행 구성 설정
    dim3 block (BDIMX, BDIMY);
    dim3 grid  (1, 1);
    printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x,
           block.y);

    // 디바이스 메모리 할당
    int *d_C;
    CHECK(cudaMalloc((int**)&d_C, nBytes));
    int *gpuRef  = (int *)malloc(nBytes);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set col read col   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read row   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col   ", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDyn<<<grid, block, BDIMX*BDIMY*sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col dyn", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColPad<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col pad", gpuRef, nx * ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadColDynPad<<<grid, block, (BDIMX + IPAD)*BDIMY*sizeof(int)>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    if(iprintf)  printData("set row read col DP ", gpuRef, nx * ny);

    CHECK(cudaFree(d_C));
    free(gpuRef);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}


///////////////////////////////////////////////////////////////////////
#define DIM 128



extern __shared__ int dsmem[];

int recursiveReduce(int *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

// unroll4 + complete unroll for loop + gmem
__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // boundary check
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // set to smem by each threads
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemDyn(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int smem[];

    // set thread ID
    unsigned int tid = threadIdx.x;
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // set to smem by each threads
    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)  smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)  smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

// unroll4 + complete unroll for loop + gmem
__global__ void reduceGmemUnroll(int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 4;

    // unrolling 4
    if (idx < n)
    {
        int a1, a2, a3, a4;
        a1 = a2 = a3 = a4 = 0;
        a1 = g_idata[idx];
        if (idx + blockDim.x < n) a2 = g_idata[idx + blockDim.x];
        if (idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
        if (idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }

    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmemUnroll(int *g_idata, int *g_odata, unsigned int n)
{
    // static shared memory
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;

    // global index, 4 blocks of input data processed at a time
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4 blocks
    int tmpSum = 0;

    // boundary check
    if (idx < n)
    {
        int a1, a2, a3, a4;
        a1 = a2 = a3 = a4 = 0;
        a1 = g_idata[idx];
        if (idx + blockDim.x < n) a2 = g_idata[idx + blockDim.x];
        if (idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
        if (idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in shared memory
    if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)  smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)  smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)   smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceSmemUnrollDyn(int *g_idata, int *g_odata, unsigned int n)
{
    extern __shared__ int smem[];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    // unrolling 4
    int tmpSum = 0;

    if (idx < n)
    {
        int a1, a2, a3, a4;
        a1 = a2 = a3 = a4 = 0;
        a1 = g_idata[idx];
        if (idx + blockDim.x < n) a2 = g_idata[idx + blockDim.x];
        if (idx + 2 * blockDim.x < n) a3 = g_idata[idx + 2 * blockDim.x];
        if (idx + 3 * blockDim.x < n) a4 = g_idata[idx + 3 * blockDim.x];
        tmpSum = a1 + a2 + a3 + a4;
    }

    smem[tid] = tmpSum;
    __syncthreads();

    // in-place reduction in global memory
    if (blockDim.x >= 1024 && tid < 512)  smem[tid] += smem[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)  smem[tid] += smem[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) smem[tid] += smem[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

__global__ void reduceNeighboredGmem(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceNeighboredSmem(int *g_idata, int *g_odata,
                                     unsigned int  n)
{
    __shared__ int smem[DIM];

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    smem[tid] = idata[tid];
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            smem[tid] += smem[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = smem[0];
}

int reduceSum(int argc, char **argv)
{
    // 디바이스 설정
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // 초기화
    int size = 1 << 22;
    printf("    with array size %d  ", size);

    // 실행 구성 설정
    int blocksize = DIM;

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // 호스트 메모리 할당
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    int gpu_sum = 0;

    // 디바이스 메모리 할당
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    int cpu_sum = recursiveReduce (tmp, size);
    printf("cpu reduce          : %d\n", cpu_sum);


    CHECK(cudaMemcpy(d_idata, h_idata, bytes,                cudaMemcpyHostToDevice));
    reduceNeighboredGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceNeighboredGmem: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceNeighboredSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceNeighboredSmem: %d <<<grid %d block %d>>>\n", gpu_sum, grid.x, block.x);

    // reduce gmem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceGmem<<<grid.x, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceGmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
           block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceSmem<<<grid.x, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceSmem          : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
           block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceSmemDyn<<<grid.x, block, blocksize*sizeof(int)>>>(d_idata, d_odata,
            size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("reduceSmemDyn       : %d <<<grid %d block %d>>>\n", gpu_sum, grid.x,
           block.x);

    // reduce gmem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceGmemUnroll<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("reduceGmemUnroll4   : %d <<<grid %d block %d>>>\n", gpu_sum,
            grid.x / 4, block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceSmemUnroll<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("reduceSmemUnroll4   : %d <<<grid %d block %d>>>\n", gpu_sum,
            grid.x / 4, block.x);

    // reduce smem
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    reduceSmemUnrollDyn<<<grid.x / 4, block, DIM*sizeof(int)>>>(d_idata,
            d_odata, size);
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("reduceSmemDynUnroll4: %d <<<grid %d block %d>>>\n", gpu_sum,
            grid.x / 4, block.x);

    // 메모리 해제
    free(h_idata);
    free(h_odata);
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    CHECK(cudaDeviceReset());

    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}

__device__ void clock_block(clock_t *d_o, clock_t clock_count)
{
	unsigned int start_clock = (unsigned int)clock();

	clock_t clock_offset = 0;

	while (clock_offset < clock_count)
	{
		unsigned int end_clock = (unsigned int)clock();

		// The code below should work like
		// this (thanks to modular arithmetics):
		//
		// clock_offset = (clock_t) (end_clock > start_clock ?
		//                           end_clock - start_clock :
		//                           end_clock + (0xffffffffu - start_clock));
		//
		// Indeed, let m = 2^32 then
		// end - start = end + m - start (mod m).

		clock_offset = (clock_t)(end_clock - start_clock);
	}

	d_o[0] = clock_offset;
}


////////////////////////////////////////////////////////////////////////////////
// clock_block()을 호출하는 커널.
//두 커널이 동일한 스트림 상에서 의존하도록 한다.

__global__ void kernel_A(clock_t *d_o, clock_t clock_count)
{
	clock_block(d_o, clock_count);
}
__global__ void kernel_B(clock_t *d_o, clock_t clock_count)
{
	clock_block(d_o, clock_count);
}
__global__ void kernel_C(clock_t *d_o, clock_t clock_count)
{
	clock_block(d_o, clock_count);
}
__global__ void kernel_D(clock_t *d_o, clock_t clock_count)
{
	clock_block(d_o, clock_count);
}
int simpleHyperQ(int argc, char **argv)
{
	int nstreams = 8;       // 스트림 개수
	float kernel_time = 10; // 커널이 실행될 ms 단위 시간
	float elapsed_time;
	int cuda_device = 0;

	char * iname = "CUDA_DEVICE_MAX_CONNECTIONS";
	setenv(iname, "4", 1); // 4 or 32
	char *ivalue = getenv(iname);
	printf("%s = %s\n", iname, ivalue);

	cudaDeviceProp deviceProp;
	CHECK(cudaGetDevice(&cuda_device));
	CHECK(cudaGetDeviceProperties(&deviceProp, cuda_device));

	printf("> Detected Compute SM %d.%d hardware with %d multi-processors\n",
		deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

	// 호스트 메모리 할당
	clock_t *a = 0;
	CHECK(cudaMallocHost((void **)&a, sizeof(clock_t)));

	// 디바이스 메모리 할당
	clock_t *d_a = 0;
	CHECK(cudaMalloc((void **)&d_a, 2 * nstreams * sizeof(clock_t)));

	// 스트림 객체에 대한 메모리 할당 및 생성
	cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));

	for (int i = 0; i < nstreams; i++)
	{
		CHECK(cudaStreamCreate(&(streams[i])));
	}

	// 이벤트 핸들러 생성
	cudaEvent_t start_event, stop_event;
	CHECK(cudaEventCreate(&start_event));
	CHECK(cudaEventCreate(&stop_event));

	// Target time per kernel = kernel_time ms, clockRate = in KHz
	// Target number of clocks = target time * clock frequency
#if defined(__arm__) || defined(__aarch64__)
	clock_t time_clocks = (clock_t)(kernel_time * (deviceProp.clockRate / 1000));
#else
	clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);
#endif
	clock_t total_clocks = 0;

	CHECK(cudaEventRecord(start_event, 0));

	for (int i = 0; i < nstreams; ++i)
	{
		kernel_A <<<1, 1, 0, streams[i] >>>(&d_a[2 * i], time_clocks);
		total_clocks += time_clocks;
		kernel_B <<<1, 1, 0, streams[i] >>>(&d_a[2 * i + 1], time_clocks);
		total_clocks += time_clocks;
		kernel_C <<<1, 1, 0, streams[i] >>>(&d_a[2 * i], time_clocks);
		total_clocks += time_clocks;
		kernel_D <<<1, 1, 0, streams[i] >>>(&d_a[2 * i + 1], time_clocks);
		total_clocks += time_clocks;
	}

	// 스트림 0 상의 중단 이벤트
	CHECK(cudaEventRecord(stop_event, 0));

	// 여기서 CPU는 GPU와 독립적으로 병렬 작업 수행 진행.
	// 여기서는 모든 작업이 완료될 때까지 대기한다.

	CHECK(cudaEventSynchronize(stop_event));
	CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));

	printf("Expected time for serial execution of %d sets of kernels is between approx. %.3fs and %.3fs\n", nstreams, (nstreams + 1) * kernel_time / 1000.0f, 2 * nstreams *kernel_time / 1000.0f);
	printf("Expected time for fully concurrent execution of %d sets of kernels is approx. %.3fs\n", nstreams, 2 * kernel_time / 1000.0f);
	printf("Measured time for sample = %.3fs\n", elapsed_time / 1000.0f);

	bool bTestResult = (a[0] >= total_clocks);

	for (int i = 0; i < nstreams; i++)
	{
		cudaStreamDestroy(streams[i]);
	}

	free(streams);
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	cudaFreeHost(a);
	cudaFree(d_a);

	return (bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

/////////////////////////////////////////////////////////////////////////////////
#define LOOP_COUNT 3000000

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data)
{
	printf("callback from stream %d\n", *((int *)data));
}

__global__ void kernel_1()
{
	double sum = 0.0;

	for (int i = 0; i < LOOP_COUNT; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_2()
{
	double sum = 0.0;

	for (int i = 0; i < LOOP_COUNT; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_3()
{
	double sum = 0.0;

	for (int i = 0; i < LOOP_COUNT; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}

__global__ void kernel_4()
{
	double sum = 0.0;

	for (int i = 0; i < LOOP_COUNT; i++)
	{
		sum = sum + tan(0.1) * tan(0.1);
	}
}



int simpleCallback(int argc, char **argv)
{
	int n_streams = 8;

	if (argc > 2) n_streams = atoi(argv[2]);

	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("> %s Starting...\n", argv[0]);
	printf("> Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
		deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

	// 최대 연결 수 설정
	char * iname = "CUDA_DEVICE_MAX_CONNECTIONS";
	setenv(iname, "8", 1);
	char *ivalue = getenv(iname);
	printf("> %s = %s\n", iname, ivalue);
	printf("> with streams = %d\n", n_streams);

	// 스트림 할당 및 초기화
	cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(
		cudaStream_t));

	for (int i = 0; i < n_streams; i++)
	{
		CHECK(cudaStreamCreate(&(streams[i])));
	}

	dim3 block(1);
	dim3 grid(1);
	cudaEvent_t start_event, stop_event;
	CHECK(cudaEventCreate(&start_event));
	CHECK(cudaEventCreate(&stop_event));

	int stream_ids[4];

	CHECK(cudaEventRecord(start_event, 0));

	for (int i = 0; i < n_streams; i++)
	{
		stream_ids[i] = i;
		kernel_1 <<<grid, block, 0, streams[i] >>>();
		kernel_2 <<<grid, block, 0, streams[i] >>>();
		kernel_3 <<<grid, block, 0, streams[i] >>>();
		kernel_4 <<<grid, block, 0, streams[i] >>>();
		CHECK(cudaStreamAddCallback(streams[i], my_callback,
			(void *)(stream_ids + i), 0));
	}

	CHECK(cudaEventRecord(stop_event, 0));
	CHECK(cudaEventSynchronize(stop_event));

	float elapsed_time;
	CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
	printf("Measured time for parallel execution = %.3fs\n", elapsed_time);

	// 스트림 해제
	for (int i = 0; i < n_streams; i++)
	{
		CHECK(cudaStreamDestroy(streams[i]));
	}

	free(streams);

	CHECK(cudaDeviceReset());

	return 0;
}
int main(int argc, char* argv[]){

   int ex=0;
   ex=atoi(argv[1]);
   printf("run ex : %d\n",ex);
   switch(ex){
    case 1:{
     printf("multithread\n");//stream
     multithread();
     break;
    }
    case 2:{
     printf("coaleascing\n");
     coaleascing(argc, argv);
     break;
    }
    case 3:{
     printf("shared_memory_reverse\n");//simple smem + sync
     shared_memory_reverse();
     break;
    }
    case 4:{
     printf("reduceSum\n");
     reduceSum(argc,argv);
     break;
    }
    case 5:{
     printf("smemSquare\n");//smem + sync
     smemSquare(argc,argv);
     break;
    }
    case 6:{
     printf("simpleHyperQ\n");//hyper q
     simpleHyperQ(argc,argv);
     break;
    }
    case 7:{
     printf("simpleCallback\n");//stream
     simpleCallback(argc,argv);
     break;
    }
    case 8:{
     printf("async\n");//simple async memcpy
     async(argc,argv);
     break;
    }
    case 9:{
     printf("data_transfer_pageable_vs_pinned\n");
     data_transfer_pageable_vs_pinned();
     break;
    }
    case 10:{
     printf("overlap\n");//stream
     overlap(argc,argv);
     break;
    }
  }
  return 0;
}