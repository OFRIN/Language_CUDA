#include <iostream>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define BLOCK_N 512

__global__ void add(int *a, int *b, int *c) {
	// printf("# blockIdx : %d\n", blockIdx.x);
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int* a, int N)
{
	for (int i = 0; i < N; ++i)
		a[i] = rand();
}

int main()
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = BLOCK_N * sizeof(int);

	// Allocate space for device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	// Setup input values
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	/*for (int i = 0; i < BLOCK_N; i++)
	{
	a[i] = i;
	b[i] = i;
	c[i] = 0;
	}*/

	random_ints(a, BLOCK_N);
	random_ints(b, BLOCK_N);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with blocks
	add << <BLOCK_N, 1 >> >(d_a, d_b, d_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	cout << "# a" << endl;
	for (int i = 0; i<10; i++)
		cout << a[i] << " ";
	cout << endl;

	cout << "# b" << endl;
	for (int i = 0; i<10; i++)
		cout << b[i] << " ";
	cout << endl;

	cout << "# c" << endl;
	for (int i = 0; i<10; i++)
		cout << c[i] << " ";
	cout << endl;

	// Cleanup
	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
