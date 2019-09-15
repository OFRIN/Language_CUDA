#include <iostream>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define THREAD_N 512

__global__ void add(int *a, int *b, int *c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
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
	int size = THREAD_N * sizeof(int);

	// Allocate space for device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	// Setup input values
	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	random_ints(a, THREAD_N);
	random_ints(b, THREAD_N);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU with N threads
	add << <1, THREAD_N >> >(d_a, d_b, d_c);

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
