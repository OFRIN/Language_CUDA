#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

/*
--- General Information for device 0 ---
Name : GeForce GTX 1050
Compute capability : 6.1
clock rate : 1493000
device copy overlap : enabled
Kernel execition timeout : Enabled
--- Memory Information for device 0 ---
total global mem : 0
Total constant mem : 65536
Max mem pitch : 2147483647
Texture Alignment : 512
--- MP Information for device 0 ---
Multiprocessor count : 5
Shared mem per mp : 49152
Registers per mp : 65536
Threads in warp : 32
Max threads per block : 1024
Max thread dimensions : (1024, 1024, 64)
Max grid imensions : (2147483647, 65535, 65535)
*/

int main()
{
	cudaDeviceProp prop;

	int count = 0;
	cudaGetDeviceCount(&count);

	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);

		printf("--- General Information for device %d ---\n", i);
		printf("Name : %s\n", prop.name);
		printf("Compute capability : %d.%d\n", prop.major, prop.minor);
		printf("clock rate : %d\n", prop.clockRate);

		printf("device copy overlap : "); 
		if (prop.deviceOverlap) 
			printf("enabled\n"); 
		else 
			printf("Disabled\n"); 
		
		printf("Kernel execition timeout : "); 
		if (prop.kernelExecTimeoutEnabled) 
			printf("Enabled\n"); 
		else 
			printf("Disabled\n");

		printf("--- Memory Information for device %d ---\n", i); 
		printf("total global mem : %ld\n", prop.totalGlobalMem); 
		printf("Total constant mem : %ld\n", prop.totalConstMem); 
		printf("Max mem pitch : %ld\n", prop.memPitch); 
		printf("Texture Alignment : %ld\n", prop.textureAlignment); 
		printf("--- MP Information for device %d ---\n", i); 
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); 
		printf("Shared mem per mp : %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp : %d\n", prop.regsPerBlock); 
		printf("Threads in warp : %d\n", prop.warpSize); 
		printf("Max threads per block : %d\n", prop.maxThreadsPerBlock); 
		printf("Max thread dimensions : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); 
		printf("Max grid imensions : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); 
		printf("\n");

	}
}