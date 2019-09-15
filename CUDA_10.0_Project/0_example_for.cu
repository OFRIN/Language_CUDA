#include <iostream>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define N (9 * 9)
#define THREADS_PER_BLOCK 9
#define BLOCK (N / THREADS_PER_BLOCK)

__global__ void loop(int *c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = (threadIdx.x + 1) * (blockIdx.x + 1);

	printf("%d * %d = %d\n", threadIdx.x + 1, blockIdx.x + 1, c[index]);
}

int main()
{
	int *result;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);

	// Allocate space for device copies of a, b, c
	cudaMalloc((void**)&d_c, size);

	// Setup input values
	result = (int*)malloc(size);

	// Launch loop() kernel on GPU
	loop <<<BLOCK, THREADS_PER_BLOCK >>>(d_c);

	// Copy result back to host
	cudaMemcpy(result, d_c, size, cudaMemcpyDeviceToHost);

	// Cleanup
	free(result);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
