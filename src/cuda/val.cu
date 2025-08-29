#include <stdexcept>
#include "cuda/val.cuh"

namespace {

template<typename T>
__global__ void allcloseKernel(
    const T* src0_ptr, const strides_t src0_strides,
    const T* src1_ptr, const strides_t src1_strides,
    const shape_t shape, uint8_t rank, size_t ne,
    double rtol, double atol,
    int* result  // changed from bool* to int*
) {
    size_t indices[8] = {0};

    for (size_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x; 
         linear_idx < ne; 
         linear_idx += blockDim.x * gridDim.x) {
        
        size_t temp_idx = linear_idx;
        for (int d = rank - 1; d >= 0; --d) {
            indices[d] = temp_idx % shape.sizes[d];
            temp_idx /= shape.sizes[d];
        }

        size_t off0 = 0, off1 = 0;
        for (size_t d = 0; d < rank; ++d) {
            off0 += indices[d] * src0_strides.sizes[d];
            off1 += indices[d] * src1_strides.sizes[d];
        }

        double a = static_cast<double>(src0_ptr[off0]);
        double b = static_cast<double>(src1_ptr[off1]);
        if (std::fabs(a - b) > (atol + rtol * std::fabs(b))) {
            atomicAnd(result, 0);  // Use 0 instead of false
            return;
        }
    }
}

template<typename S>
bool launchAllcloseKernel(const tensor_t* src0, const tensor_t* src1, stream_t stream, double rtol, double atol) {
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    size_t ne = 1;
    for (uint8_t dim = 0; dim < src0->rank; ++dim) {
        ne *= src0->shape.sizes[dim];
    }

    int h_result = 1;
    int* d_result;

    cudaMallocAsync(&d_result, sizeof(int), cudaStream);
    cudaMemcpyAsync(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice, cudaStream);

    if (src0->rank == 0) {
        // Handle scalar case
        double a = static_cast<double>(*(const S*)(src0->address));
        double b = static_cast<double>(*(const S*)(src1->address));
        h_result = (std::fabs(a - b) <= (atol + rtol * std::fabs(b))) ? 1 : 0;
        cudaMemcpyAsync(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice, cudaStream);
    } else {
        int blockSize = 256;
        int gridSize = (ne + blockSize - 1) / blockSize;

        allcloseKernel<S><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S*)(src0->address), src0->strides,
            (const S*)(src1->address), src1->strides,
            src0->shape, src0->rank, ne, rtol, atol,
            d_result
        );
    }

    cudaMemcpyAsync(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost, cudaStream);
    cudaStreamSynchronize(cudaStream);
    cudaFreeAsync(d_result, cudaStream);

    return h_result == 1;
}


} // namespace

namespace cuda {

bool allclose(const tensor_t* src0, const tensor_t* src1, stream_t stream, double rtol, double atol) {
    switch (src0->dtype) { 
        case float32: return launchAllcloseKernel<float>(src0, src1, stream, rtol, atol);
        case float64: return launchAllcloseKernel<double>(src0, src1, stream, rtol, atol);
        default: throw std::runtime_error("Unsupported dtype");
    }
}

} // namespace cuda
