#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <conio.h>
#include <assert.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/random.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>


#define N   1025
#define M   12

//#include "Utilities.cuh"
//#include "TimingGPU.cuh"


/*simpleBLAS Matrix size */
#define N2 (275) 

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

/**********************/
/* cuBLAS ERROR CHECK */
/**********************/
#ifndef cublasSafeCall
#define cublasSafeCall(err)     __cublasSafeCall(err, __FILE__, __LINE__)
#endif


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

int debug_segfault(int argc, char **argv)
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


int debug_segfault_fixed(int argc, char **argv)
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

    CHECK(cudaMemcpy(d_matrix, d_ptrs, N * sizeof(int *),
                    cudaMemcpyHostToDevice));

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


__global__ void simple_reduction(int *shared_var, int *input_values, int N,
                                 int iters)
{
    __shared__ int local_mem[256];
    int iter, i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    int local_dim = blockDim.x;
    int minThreadInThisBlock = blockIdx.x * blockDim.x;
    int maxThreadInThisBlock = minThreadInThisBlock + (blockDim.x - 1);

    if (maxThreadInThisBlock >= N)
    {
        local_dim = N - minThreadInThisBlock;
    }

    for (iter = 0; iter < iters; iter++)
    {
        if (tid < N)
        {
            local_mem[local_tid] = input_values[tid];
        }

        // Required for correctness
        // __syncthreads();
 
        if (local_tid == 0)
        {
            int sum = 0;

            for (i = 0; i < local_dim; i++)
            {
                sum = sum + local_mem[i];
            }

            atomicAdd(shared_var, sum);
        }

        // Required for correctness
        // __syncthreads();
    }
}

int debug_hazard(int argc, char **argv)
{
    int N = 20480;
    int block = 256;
    int device_iters = 3;
    int runs = 1;
    int i, true_value;
    int *d_shared_var, *d_input_values, *h_input_values;
    int h_sum;
    double mean_time = 0.0;

    CHECK(cudaMalloc((void **)&d_shared_var, sizeof(int)));
    CHECK(cudaMalloc((void **)&d_input_values, N * sizeof(int)));
    h_input_values = (int *)malloc(N * sizeof(int));

    for (i = 0; i < N; i++)
    {
        h_input_values[i] = i;
        true_value += i;
    }

    true_value *= device_iters;

    for (i = 0; i < runs; i++)
    {
        CHECK(cudaMemset(d_shared_var, 0x00, sizeof(int)));
        CHECK(cudaMemcpy(d_input_values, h_input_values, N * sizeof(int),
                         cudaMemcpyHostToDevice));
        double start = seconds();

        simple_reduction<<<N / block, block>>>(d_shared_var,
                d_input_values, N, device_iters);

        CHECK(cudaDeviceSynchronize());
        mean_time += seconds() - start;
        CHECK(cudaMemcpy(&h_sum, d_shared_var, sizeof(int),
                         cudaMemcpyDeviceToHost));

        if (h_sum != true_value)
        {
            fprintf(stderr, "Validation failure: expected %d, got %d\n",
                    true_value, h_sum);
            return 1;
        }
    }

    mean_time /= runs;

    printf("Mean execution time for reduction: %.4f ms\n",
           mean_time * 1000.0);

    return 0;
}



/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C) {
  int i;
  int j;
  int k;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float prod = 0;

      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}

int simpleCublas(int argc, char **argv) {
  cublasStatus_t status;
  float *h_A;
  float *h_B;
  float *h_C;
  float *h_C_ref;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  int n2 = N2 * N2;
  int i;
  float error_norm;
  float ref_norm;
  float diff;
  cublasHandle_t handle;

  int dev = findCudaDevice(argc, (const char **)argv);

  if (dev == -1) {
    return EXIT_FAILURE;
  }

  /* Initialize CUBLAS */
  printf("simpleCUBLAS test running..\n");

  status = cublasCreate(&handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  /* Allocate host memory for the matrices */
  h_A = reinterpret_cast<float *>(malloc(n2 * sizeof(h_A[0])));

  if (h_A == 0) {
    fprintf(stderr, "!!!! host memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  h_B = reinterpret_cast<float *>(malloc(n2 * sizeof(h_B[0])));

  if (h_B == 0) {
    fprintf(stderr, "!!!! host memory allocation error (B)\n");
    return EXIT_FAILURE;
  }

  h_C = reinterpret_cast<float *>(malloc(n2 * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Fill the matrices with test data */
  for (i = 0; i < n2; i++) {
    h_A[i] = rand() / static_cast<float>(RAND_MAX);
    h_B[i] = rand() / static_cast<float>(RAND_MAX);
    h_C[i] = rand() / static_cast<float>(RAND_MAX);
  }

  /* Allocate device memory for the matrices */
  if (cudaMalloc(reinterpret_cast<void **>(&d_A), n2 * sizeof(d_A[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B), n2 * sizeof(d_B[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C), n2 * sizeof(d_C[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  /* Initialize the device matrices with the host matrices */
  status = cublasSetVector(n2, sizeof(h_A[0]), h_A, 1, d_A, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write A)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(n2, sizeof(h_B[0]), h_B, 1, d_B, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write B)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(n2, sizeof(h_C[0]), h_C, 1, d_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write C)\n");
    return EXIT_FAILURE;
  }

  /* Performs operation using plain C code */
  simple_sgemm(N2, alpha, h_A, h_B, beta, h_C);
  h_C_ref = h_C;

  /* Performs operation using cublas */
  status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N2, N2, N2, &alpha, d_A,
                       N2, d_B, N2, &beta, d_C, N2);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }

  /* Allocate host memory for reading back the result from device memory */
  h_C = reinterpret_cast<float *>(malloc(n2 * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Read the result back */
  status = cublasGetVector(n2, sizeof(h_C[0]), d_C, 1, h_C, 1);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (read C)\n");
    return EXIT_FAILURE;
  }

  /* Check result against reference */
  error_norm = 0;
  ref_norm = 0;

  for (i = 0; i < n2; ++i) {
    diff = h_C_ref[i] - h_C[i];
    error_norm += diff * diff;
    ref_norm += h_C_ref[i] * h_C_ref[i];
  }

  error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
  ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));

  if (fabs(ref_norm) < 1e-7) {
    fprintf(stderr, "!!!! reference norm is 0\n");
    return EXIT_FAILURE;
  }

  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasDestroy(handle);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  if (error_norm / ref_norm < 1e-6f) {
    printf("simpleCUBLAS test passed.\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("simpleCUBLAS test failed.\n");
    exit(EXIT_FAILURE);
  }
}

/*
Calculating the all-pairs distances between points in two different sets in CUDA can be solved by observing that

||x-y||^2=||x||^2+||y||^2-2*<x,y>
where ||×|| is the l2 norm and <x,y> denotes the scalar product between x and y.

The norms ||x|| and ||y|| can be calculated by approaches inspired by Reduce matrix rows with CUDA, while the scalar products <x,y> can then be calculated as the matrix-matrix multiplication X*Y^T using cublas<t>gemm().

A fully worked out implementation is available on our GitHub page.

Please, note that for the calculation of the norms ||×|| two approaches are reported, one using cuBLAS cublas<t>gemv and one using Thurst’s transform.

For a problem size of typical interest (1000/2000 elements in the sets each with 128 dimensions), we have experienced the following timings on a GT540M card:


Approach nr. 1   0.12 ms
Approach nr. 2   0.59 ms
*/


/***********************************************************/
/* SQUARED ABSOLUTE VALUE FUNCTOR - NEEDED FOR APPROACH #1 */
/***********************************************************/
struct abs2 {
	__host__ __device__ double operator()(const float &x) const { return x * x; }
};

// --- Required for approach #2
__device__ float *vals;

/******************************************/
/* ROW_REDUCTION - NEEDED FOR APPROACH #2 */
/******************************************/
struct row_reduction {

    const int Ncols;    // --- Number of columns

    row_reduction(int _Ncols) : Ncols(_Ncols) {}

    __device__ float operator()(float& x, int& y ) {
        float temp = 0.f;
        for (int i = 0; i<Ncols; i++)
            temp += vals[i + (y*Ncols)] * vals[i + (y*Ncols)];
        return temp;
    }
};

/************************************************/
/* KERNEL FUNCTION TO ASSEMBLE THE FINAL RESULT */
/************************************************/
__global__ void assemble_final_result(const float * __restrict__ d_norms_x_2, const float * __restrict__ d_norms_y_2, float * __restrict__ d_dots,
									  const int NX, const int NY) {

	const int i = threadIdx.x + blockIdx.x * gridDim.x;
	const int j = threadIdx.y + blockIdx.y * gridDim.y;

	if ((i < NY) && (j < NX)) d_dots[i * NX+ j] = d_norms_x_2[j] + d_norms_y_2[i] - 2 * d_dots[i * NX+ j];

}

/********/
/* MAIN */
/********/
int two_point_pair_distance(){
    //const int Ndims = 128;		// --- Number of rows
    //const int NX	= 1000;		// --- Number of columns
    //const int NY	= 2000;		// --- Number of columns

    const int Ndims = 3;		// --- Number of rows
    const int NX	= 4;		// --- Number of columns
    const int NY	= 5;		// --- Number of columns

	// --- Random uniform integer distribution between 10 and 99
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(10, 99);

    // --- Matrices allocation and initialization
    thrust::device_vector<float> d_X(Ndims * NX);
    thrust::device_vector<float> d_Y(Ndims * NY);
    for (size_t i = 0; i < d_X.size(); i++) d_X[i] = (float)dist(rng);
    for (size_t i = 0; i < d_Y.size(); i++) d_Y[i] = (float)dist(rng);

    TimingGPU timerGPU;

	// --- cuBLAS handle creation
	cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));

	/**********************************************/
    /* CALCULATING THE NORMS OF THE ELEMENTS OF X */
    /**********************************************/
    thrust::device_vector<float> d_norms_x_2(NX);

	// --- Approach nr. 1
	//timerGPU.StartCounter();
	thrust::device_vector<float> d_X_2(Ndims * NX);
	thrust::transform(d_X.begin(), d_X.end(), d_X_2.begin(), abs2());

	thrust::device_vector<float> d_ones(Ndims, 1.f);

    float alpha = 1.f;
    float beta  = 0.f;
    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, Ndims, NX, &alpha, thrust::raw_pointer_cast(d_X_2.data()), Ndims, 
                               thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_norms_x_2.data()), 1));
	
	//printf("Timing for approach #1 = %f\n", timerGPU.GetCounter());

    // --- Approach nr. 2
	//timerGPU.StartCounter();
 //   float *s_vals = thrust::raw_pointer_cast(&d_X[0]);
 //   gpuErrchk(cudaMemcpyToSymbol(vals, &s_vals, sizeof(float *)));
 //   thrust::transform(d_norms_x_2.begin(), d_norms_x_2.end(), thrust::counting_iterator<int>(0),  d_norms_x_2.begin(), row_reduction(Ndims));

	//printf("Timing for approach #2 = %f\n", timerGPU.GetCounter());

	/**********************************************/
    /* CALCULATING THE NORMS OF THE ELEMENTS OF Y */
    /**********************************************/
    thrust::device_vector<float> d_norms_y_2(NX);

	thrust::device_vector<float> d_Y_2(Ndims * NX);
	thrust::transform(d_Y.begin(), d_Y.end(), d_Y_2.begin(), abs2());

    cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_T, Ndims, NY, &alpha, thrust::raw_pointer_cast(d_Y_2.data()), Ndims, 
                               thrust::raw_pointer_cast(d_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_norms_y_2.data()), 1));


	/***********************************/
    /* CALCULATING THE SCALAR PRODUCTS */
    /***********************************/
    thrust::device_vector<float> d_dots(NX * NY);

	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, NX, NY, Ndims, &alpha,
		                       thrust::raw_pointer_cast(d_X.data()), Ndims, thrust::raw_pointer_cast(d_Y.data()), Ndims, &beta,
							   thrust::raw_pointer_cast(d_dots.data()), NX));

	/*****************************/
	/* ASSEMBLE THE FINAL RESULT */
	/*****************************/
	
	dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	dim3 dimGrid(iDivUp(NX, BLOCK_SIZE_X), iDivUp(NY, BLOCK_SIZE_Y));
	assemble_final_result<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(d_norms_x_2.data()), thrust::raw_pointer_cast(d_norms_y_2.data()), 
		                                         thrust::raw_pointer_cast(d_dots.data()), NX, NY);
	
	for(int i = 0; i < NX * NY; i++) std::cout << d_dots[i] << "\n";

	return 0;
}

/*
We are here providing a full example on how using cublas <t>gemm to perform multiplications between submatrices of full matrices A and B and how assigning the result to a submatrix of a full matrix C.

The code makes use of

pointer arithmetics to access submatrices;
the concept of the leading dimension and of submatrix dimensions.
 

The code available on our GitHub page considers three matrices:

A – 10 x 9;
B – 15 x 13;
C – 10 x 12.
 
Matrix C is initialized to all 10s.
The code performs the following submatrix multiplication in Matlab language:

C(1+x3:5+x3,1+y3:3+y3) = A(1+x1:5+x1,1+y1:4+y1) * B(1+x2:4+x2,1+y2:3+x2);
*/


/********/
/* MAIN */
/********/
int submatrix_multiplication()
{
	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
  
	//const int Nrows1 = 10;			// --- Number of rows of matrix 1
	//const int Ncols1 = 10;			// --- Number of columns of matrix 1

	//const int Nrows2 = 15;			// --- Number of rows of matrix 2
	//const int Ncols2 = 15;			// --- Number of columns of matrix 2

	//const int Nrows3 = 12;			// --- Number of rows of matrix 3
	//const int Ncols3 = 12;			// --- Number of columns of matrix 3

	const int Nrows1 = 10;			// --- Number of rows of matrix 1
	const int Ncols1 = 9;			// --- Number of columns of matrix 1

	const int Nrows2 = 15;			// --- Number of rows of matrix 2
	const int Ncols2 = 13;			// --- Number of columns of matrix 2

	const int Nrows3 = 10;			// --- Number of rows of matrix 3
	const int Ncols3 = 12;			// --- Number of columns of matrix 3

	const int Nrows = 5;			// --- Number of rows of submatrix matrix 3 = Number of rows of submatrix 1
	const int Ncols = 3;			// --- Number of columns of submatrix matrix 3 = Number of columns of submatrix 2

	const int Nrowscols = 4;		// --- Number of columns of submatrix 1 and of rows of submatrix 2

	const int x1 = 3;				// --- Offset for submatrix multiplication along the rows
	const int y1 = 2;				// --- Offset for submatrix multiplication along the columns
	
	const int x2 = 6;				// --- Offset for submatrix multiplication along the rows
	const int y2 = 4;				// --- Offset for submatrix multiplication along the columns

	const int x3 = 3;				// --- Offset for submatrix multiplication along the rows
	const int y3 = 5;				// --- Offset for submatrix multiplication along the columns

	// --- Random uniform integer distribution between 0 and 100
	thrust::default_random_engine rng;
	thrust::uniform_int_distribution<int> dist(0, 20);

	// --- Matrix allocation and initialization
	thrust::device_vector<float> d_matrix1(Nrows1 * Ncols1);
	thrust::device_vector<float> d_matrix2(Nrows2 * Ncols2);
	for (size_t i = 0; i < d_matrix1.size(); i++) d_matrix1[i] = (float)dist(rng);
	for (size_t i = 0; i < d_matrix2.size(); i++) d_matrix2[i] = (float)dist(rng);

	printf("\n\nOriginal full size matrix A\n");
	for(int i = 0; i < Nrows1; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols1; j++) 
			std::cout << d_matrix1[j * Nrows1 + i] << " ";
		std::cout << "]\n";
	}

	printf("\n\nOriginal full size matrix B\n");
	for(int i = 0; i < Nrows2; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols2; j++) 
			std::cout << d_matrix2[j * Nrows2 + i] << " ";
		std::cout << "]\n";
	}

	/*************************/
	/* MATRIX MULTIPLICATION */
	/*************************/
	cublasHandle_t handle;

	cublasSafeCall(cublasCreate(&handle));

	thrust::device_vector<float> d_matrix3(Nrows3 * Ncols3, 10.f);

	float alpha = 1.f;
	float beta  = 0.f;
	cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nrows, Ncols, Nrowscols, &alpha,
				   thrust::raw_pointer_cast(d_matrix1.data())+x1+Nrows1*y1, Nrows1, thrust::raw_pointer_cast(d_matrix2.data())+x2+Nrows2*y2, Nrows2,
				   &beta, thrust::raw_pointer_cast(d_matrix3.data())+x3+Nrows3*y3, Nrows3));

	printf("\n\nResult full size matrix C\n");
	for(int i = 0; i < Nrows3; i++) {
		std::cout << "[ ";
		for(int j = 0; j < Ncols3; j++) 
			std::cout << d_matrix3[j * Nrows3 + i] << " ";
		std::cout << "]\n";
	}

	return 0; 
}
/*
Matrix transposition is a very common operation in linear algebra.
From a numerical point of view, it is a memory bound problem since there is practically no arithmetics in it and the operation essentially consists of rearranging the layout of the matrix in memory.

Due to the particular architecture of a GPU and to the cost of performing global memory operations, matrix transposition admits no naive implementation if performance is of interest.

We here compare two different possibilities of performing matrix transposition in CUDA, one using the Thrust library and one using cuBLAS cublas<t>geam.
The full code we have set up to perform the comparison is downloadable from a Visual Studio 2010 project.
Here are the results of the tests performed on a Kepler K20c card:

Matrix Transposition

As you can see, the cuBLAS cublas<t>geam definitely outperforms the solution using Thrust and proves to be a very efficient way to perform matrix transposition in CUDA.

 
*/

inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if( CUBLAS_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUBLAS error in file '%s', line %d\n \nerror %d \nterminating!\n",__FILE__, __LINE__,err); 
        getch(); cudaDeviceReset(); assert(0); 
    }
}

// convert a linear index to a linear index in the transpose 
struct transpose_index : public thrust::unary_function<size_t,size_t>
{
    size_t m, n;

    __host__ __device__
    transpose_index(size_t _m, size_t _n) : m(_m), n(_n) {}

    __host__ __device__
    size_t operator()(size_t linear_index)
    {
        size_t i = linear_index / n;
        size_t j = linear_index % n;

        return m * j + i;
    }
};

// convert a linear index to a row index
struct row_index : public thrust::unary_function<size_t,size_t>
{
    size_t n;

    __host__ __device__
    row_index(size_t _n) : n(_n) {}

    __host__ __device__

    size_t operator()(size_t i)
    {
        return i / n;
    }
};

// transpose an M-by-N array
template <typename T>
void transpose(size_t m, size_t n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
    thrust::counting_iterator<size_t> indices(0);

    thrust::gather
    (thrust::make_transform_iterator(indices, transpose_index(n, m)),
    thrust::make_transform_iterator(indices, transpose_index(n, m)) + dst.size(),
    src.begin(),dst.begin());
}

// print an M-by-N array
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
    thrust::host_vector<T> h_data = d_data;

    for(size_t i = 0; i < m; i++)
    {
        for(size_t j = 0; j < n; j++)
            std::cout << std::setw(8) << h_data[i * n + j] << " ";
            std::cout << "\n";
    }
}

int matrix_transpose(void)
{
    size_t m = 5; // number of rows
    size_t n = 4; // number of columns

    // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
    thrust::device_vector<double> data(m * n, 1.);
    data[1] = 2.;
    data[3] = 3.;

    std::cout << "Initial array" << std::endl;
    print(m, n, data);

    std::cout << "Transpose array - Thrust" << std::endl;
    thrust::device_vector<double> transposed_thrust(m * n);
    transpose(m, n, data, transposed_thrust);
    print(n, m, transposed_thrust);

    std::cout << "Transpose array - cuBLAS" << std::endl;
    thrust::device_vector<double> transposed_cuBLAS(m * n);
    double* dv_ptr_in  = thrust::raw_pointer_cast(data.data());
    double* dv_ptr_out = thrust::raw_pointer_cast(transposed_cuBLAS.data());
    double alpha = 1.;
    double beta  = 0.;
    cublasHandle_t handle;
    cublasSafeCall(cublasCreate(&handle));
    cublasSafeCall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &alpha, dv_ptr_in, n, &beta, dv_ptr_in, n, dv_ptr_out, m)); 
    print(n, m, transposed_cuBLAS);

    getch();

    return 0;
}


#define m 6 // a - mxk matrix
#define n 4 // b - kxn matrix
#define k 5 // c - mxn matrix

#define SWAP(a,b,tmp) { (tmp)=(a); (a)=(b); (b)=(tmp); }

// https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
void cublasRowMajorSgemm(float *a, float *b, float *c) {
    int i,j; // i-row valex, j-column valex
    cublasHandle_t handle; // CUBLAS context
    cublasCreate(&handle); // initialize CUBLAS context
    float al=1.0f; // al =1
    float bet=1.0f; // bet =1

    // b^T = nxk matrix
    // a^T = kxm matrix
    // c^T = nxm matrix
    // c^T = b^T * a^T

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,n,m,k,&al,b,n,a,k,&bet,c,n);
    cudaDeviceSynchronize();
    printf ("c after Sgemm :\n");
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            printf("%7.0f",c[i*n+j]);
        }
        printf("\n");
    }
    cublasDestroy(handle); // destroy CUBLAS context
}
int cublas_gemm_c(void) {
    int i,j, ind; // i-row valex, j-column valex
    float *a; // mxk matrix
    float *b; // kxn matrix
    float *c; // mxn matrix
    // unified memory for a,b,c
    cudaMallocManaged(&a, m*k*sizeof(cuComplex));
    cudaMallocManaged(&b, k*n*sizeof(cuComplex));
    cudaMallocManaged(&c, m*n*sizeof(cuComplex));
    // define an mxk matrix a column by column
    int val=0; // a:
    for(i=0;i<m*k;i++){ a[i] = (float)val++; }
    printf ("a:\n");
    ind=0;
    for (i=0;i<m;i++){
        for (j=0;j<k;j++){
            printf("%5.0f",a[ind++]);
        }
        printf ("\n");
    }
    // define a kxn matrix b column by column
    val=0; // b:
    for(i=0;i<k*n;i++){ b[i] = (float)val++; }
    printf ("b:\n");
    ind=0;
    for (i=0;i<k;i++){
        for (j=0;j<n;j++){
            printf("%5.0f",b[ind++]);
        }
        printf ("\n");
    }
    // define an mxn matrix c column by column
    val=0; // c:
    for(i=0;i<m*n;i++){ c[i] = (float)0; }
    printf ("c:\n");
    ind=0;
    for (i=0;i<m;i++){
        for (j=0;j<n;j++){
            printf("%5.0f",c[ind++]);
        }
        printf ("\n");
    }
    cublasRowMajorSgemm(a, b, c);
    cudaFree(a); // free memory
    cudaFree(b); // free memory
    cudaFree(c); // free memory
    return EXIT_SUCCESS ;
}


#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define m 6 // a - mxk matrix
#define n 4 // b - kxn matrix
#define k 5 // c - mxn matrix

int cublas_gemm_f(void) {
    cublasHandle_t handle; // CUBLAS context
    int i,j; // i-row valex, j-column valex
    float *a; // mxk matrix
    float *b; // kxn matrix
    float *c; // mxn matrix
    // unified memory for a,b,c
    cudaMallocManaged(&a, m*k*sizeof(cuComplex));
    cudaMallocManaged(&b, k*n*sizeof(cuComplex));
    cudaMallocManaged(&c, m*n*sizeof(cuComplex));
    // define an mxk matrix a column by column
    int val=0; // a:
    for (i=0;i<m;i++){
        for (j=0;j<k;j++){
            a[IDX2C(i,j,m)] = (float)val++;
        }
    }
    printf ("a:\n");
    for (i=0;i<m;i++){
        for (j=0;j<k;j++){
            printf("%5.0f",a[IDX2C(i,j,m)]);
        }
        printf ("\n");
    }
    // define a kxn matrix b column by column
    val=0; // b:
    for (i=0;i<k;i++){
        for (j=0;j<n;j++){
            b[IDX2C(i,j,k)] = (float)val++;
        }
    }
    printf ("b:\n");
    for (i=0;i<k;i++){
        for (j=0;j<n;j++){
            printf("%5.0f",b[IDX2C(i,j,k)]);
        }
        printf ("\n");
    }
    // define an mxn matrix c column by column
    val=0; // c:
    for (i=0;i<m;i++){
        for (j=0;j<n;j++){
            c[IDX2C(i,j,m)] = (float)0;
        }
    }
    printf ("c:\n");
    for (i=0;i<m;i++){
        for (j=0;j<n;j++){
            printf("%5.0f",c[IDX2C(i,j,m)]);
        }
        printf ("\n");
    }
    cublasCreate(&handle); // initialize CUBLAS context
    float al=1.0f; // al =1
    float bet=1.0f; // bet =1
    // matrix - matrix multiplication : c = al*a*b + bet *c
    // a -mxk matrix , b -kxn matrix , c -mxn matrix ;
    // al, bet - scalars
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
    cudaDeviceSynchronize();
    printf ("c after Sgemm :\n");
    for(i=0;i<m;i++){
        for(j=0;j<n;j++){
            printf("%7.0f",c[IDX2C(i,j,m)]); // print c after Sgemm
        }
        printf("\n");
    }
    cudaFree(a); // free memory
    cudaFree(b); // free memory
    cudaFree(c); // free memory
    cublasDestroy(handle); // destroy CUBLAS context
    return EXIT_SUCCESS ;
}


///////////////////////////////////////////////////////////////////////

/*
 * A simple example of performing matrix-vector multiplication using the cuBLAS
 * library and some randomly generated inputs.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int ROWS = 1024;
int COLS = 1024;

/*
 * Generate a matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0
 * and 100.
 */
void generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rand_max = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    for (j = 0; j < N; j++){// For each column
        for (i = 0; i < M; i++){// For each row
            double drand = (double)rand();
            A[j * M + i] = (drand / rand_max) * 100.0;
        }
    }
    *outA = A;
}

///////////////////////////////////////////////////////////////////////

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
    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);
    memset(C, 0x00, sizeof(float) * M * M);

    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device memory
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * M));

    // Transfer inputs to the device
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dA, M));
    CHECK_CUBLAS(cublasSetMatrix(N, M, sizeof(float), B, N, dB, N));
    CHECK_CUBLAS(cublasSetMatrix(M, M, sizeof(float), C, M, dC, M));

    // Execute the matrix-vector multiplication
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha,
                dA, M, dB, N, &beta, dC, M));

    // Retrieve the output vector from the device
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
    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_dense_matrix(N, M, &B);
    C = (float *)malloc(sizeof(float) * M * M);
    memset(C, 0x00, sizeof(float) * M * M);

    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // Allocate device memory
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dB, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dC, sizeof(float) * M * M));

    // Transfer inputs to the device
    CHECK_CUBLAS(cublasSetMatrixAsync(M, N, sizeof(float), A, M, dA, M,
                stream));
    CHECK_CUBLAS(cublasSetMatrixAsync(N, M, sizeof(float), B, N, dB, N,
                stream));
    CHECK_CUBLAS(cublasSetMatrixAsync(M, M, sizeof(float), C, M, dC, M,
                stream));

    // Execute the matrix-vector multiplication
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha,
                dA, M, dB, N, &beta, dC, M));

    // Retrieve the output vector from the device
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


int main(int argc, char* argv[]){

   int ex=0;
   ex=atoi(argv[1]);
   printf("run ex : %d\n",ex);
   switch(ex){
    case 1:{
     printf("debug_segfault\n");
     debug_segfault(argc, argv);
     break;
    }
    case 2:{
     printf("debug_segfault_fixed\n");
     debug_segfault_fixed(argc, argv);
     break;
    }
    case 3:{
     printf("debug_hazard\n");
     debug_hazard(argc, argv);
     break;
    }
    case 4:{
     printf("simpleCublas\n");//from nvidia
     simpleCublas(argc, argv);
     break;
    }
    case 5:{
     printf("two_point_pair_distance\n");//can be skipped
     two_point_pair_distance();
     break;
    }
    case 6:{
     printf("submatrix_multiplication\n");//can be skipped
     submatrix_multiplication();
     break;
    }
    case 7:{
     printf("matrix_transpose\n");//can be skipped
     matrix_transpose();
     break;
    }
    case 8:{
     printf("cublas_gemm_c\n");//can be skipped
     cublas_gemm_c();
     break;
    }
    case 9:{
     printf("cublas_gemm_f\n");//can be skipped
     cublas_gemm_f();
     break;
    } 
    case 10:{
     printf("cublasMM\n");
     cublasMM(argc,argv);
     break;
    } 
    case 11:{
     printf("cublasMMAsync\n");
     cublasMMAsync(argc,argv);
     break;
    } 

  }
  return 0;
}

