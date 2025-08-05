#include <cstdint>
#include <cstddef>
#include <iostream>
#include <cuda_runtime.h>
#define RANK 4

struct shape_t {
    union {
        const uint32_t* address;
        uint32_t sizes[RANK];
    };
};

struct strides_t {
    union {
        const uint64_t* address;
        uint64_t strides[RANK];
    };
};

__global__ void my_kernel(void* data, uint8_t rank, shape_t shape, strides_t strides) {
    // For demonstration, let's print sizes and strides of the tensor
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) { // Only one thread to print
        printf("Tensor rank: %d\n", rank);
        printf("Shape sizes: ");
        for (int i = 0; i < rank; ++i) {
            printf("%u ", shape.sizes[i]);
        }
        printf("\nStrides: ");
        for (int i = 0; i < rank; ++i) {
            printf("%llu ", (unsigned long long)strides.strides[i]);
        }
        printf("\n");
    }
    // Example access to the tensor data based on shape and strides can be done here...
}

int main() { 
    uint32_t cpu_sizes[RANK] = {2, 3, 4, 5};
    uint64_t cpu_strides[RANK] = {60, 20, 5, 1};

    shape_t gpu_shape;
    strides_t gpu_strides;
    
    for (int i = 0; i < RANK; ++i) {
        gpu_shape.sizes[i] = cpu_sizes[i];
        gpu_strides.strides[i] = cpu_strides[i];
    }

    void* gpu_data = nullptr;  
    uint8_t rank = RANK;

    my_kernel<<<1, 32>>>(gpu_data, rank, gpu_shape, gpu_strides);
    cudaDeviceSynchronize(); 
}