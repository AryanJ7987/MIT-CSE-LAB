#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void addVectorsBlockSizeN(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void addVectorsNThreads(int *a, int *b, int *c, int N) {
    int idx = threadIdx.x;
    while (idx < N) {
        c[idx] = a[idx] + b[idx];
        idx += blockDim.x;
    }
}

int main() {
    int N = 1024;  // Length of vectors
    int blockSize = N; // Block size equal to N
    int numBlocks = 1; // We use only one block in this case

    int *h_a, *h_b, *h_c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Allocate memory on the host
    h_a = (int*)malloc(N * sizeof(int));
    h_b = (int*)malloc(N * sizeof(int));
    h_c = (int*)malloc(N * sizeof(int));

    // Initialize input vectors on the host
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Copy input vectors from host to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with block size equal to N
    addVectorsBlockSizeN<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Copy result vector from device to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the first few elements of the result vector
    for (int i = 0; i < 10; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free memory on the host
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
