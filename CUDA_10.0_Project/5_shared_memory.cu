#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void get_max(int* src, int* max_dst, int length) {
	extern __shared__ int sm[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + tid;

	if (i<length) {
		sm[tid] = src[i];
		__syncthreads();

		for (int s = 1; s<blockDim.x; s *= 2) {
			if (tid % (2 * s) == 0) {
				// max
				if (sm[tid] < sm[tid + s])  
					sm[tid] = sm[tid + s];

				// sum
				// sm[tid] += sm[tid + s];
			}
			__syncthreads();
		}

		if (tid == 0) 
			max_dst[0] = sm[0];
	}
}

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 5, 4, 3 };
	int max[1] = { 0 };

	int* src_d;
	int* max_d;

	cudaMalloc((void**)&src_d, sizeof(int) * arraySize);
	cudaMalloc((void**)&max_d, sizeof(int) * 1);

	cudaMemcpy(src_d, a, sizeof(int)*arraySize, cudaMemcpyHostToDevice);

	// 3번째 메모리 동적할당
	get_max << <1, arraySize, arraySize * sizeof(int) >> >(src_d, max_d, arraySize);

	cudaMemcpy(max, max_d, sizeof(int), cudaMemcpyDeviceToHost);

	printf("max = %d \n", max[0]);

	return 0;
}
