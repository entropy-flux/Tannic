#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include "cuda/exc.cuh"
#include "cuda/fns.cuh"  

namespace {
    
template<typename S, typename D, class Fn>
__global__ void scalarFnKernel(const S* src, D* dst) {
    Fn fn;
    *dst = fn(*src);
}  

template<typename S, typename D, class Fn>
__global__ void batchedFnKernel(
    const S* src, shape_t src_shape, strides_t src_strides,
    D* dst, shape_t dst_shape, strides_t dst_strides,
    uint8_t rank, size_t ne
) {
    Fn fn{};

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < ne; idx += blockDim.x * gridDim.x) { 
        size_t offs = 0;
        size_t remaining = idx;

        for (int dim = rank - 1; dim >= 0; --dim) {
            size_t dim_idx = remaining % dst_shape.sizes[dim];
            remaining /= dst_shape.sizes[dim];
 
            size_t src_idx = (src_shape.sizes[dim] == 1) ? 0 : dim_idx;
            offs += src_idx * src_strides.sizes[dim];
        }

        dst[idx] = fn(src[offs]);
    }
}  

template<typename S, typename D, class Fn>
status launchFnKernel(const tensor_t* src, tensor_t* dst, stream_t stream) { 
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    if (src->rank == 0) {
        scalarFnKernel<S, D, Fn><<<1, 1, 0, cudaStream>>>(
            (const S*)(src->address),
            (D*)(dst->address)
        ); 
    } 
    
    else {
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape.sizes[dim];
        }

        size_t blockSize = 256;
        size_t gridSize = (ne + blockSize - 1) / blockSize;

        batchedFnKernel<S, D, Fn><<<gridSize, blockSize, 0, cudaStream>>>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne
        ); 
    } 

    return SUCCESS;
} 

constexpr static status launchDefaultKernel(const tensor_t* src, tensor_t* dst, stream_t) {
    return UNSUPPORTED_DTYPE;
};   

struct Log { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(log(a))) {
        return log(a);
    }
};
 
struct Exp { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(exp(a))) {
        return exp(a);
    }
};
  
struct Sqrt { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(sqrt(a))) {
        return sqrt(a);
    }
};
 
struct Abs { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(abs(a))) {
        return abs(a);
    }
};
 
struct Sin { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(sin(a))) {
        return sin(a);
    }
};
 
struct Cos { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(cos(a))) {
        return cos(a);
    }
};

struct Tan { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(tan(a))) {
        return tan(a);
    }
}; 

struct Sinh { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(sinh(a))) {
        return sinh(a);
    }
};
 
struct Cosh{ 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(cosh(a))) {
        return cosh(a);
    }
};
 
struct Tanh { 
    template<class A>
    __device__ __forceinline__ auto operator()(A&& a) const noexcept(noexcept(tanh(a))) {
        return tanh(a);
    }
};    

constexpr static inline int index(type type) {
    return static_cast<int>(type);
}  
 
using Kernel = status(*)(const tensor_t*, tensor_t*, stream_t);       

constexpr auto dispatchLog = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Log>;
    table[index(float64)] = launchFnKernel<double, double, Log>;
    return table;
}();  

constexpr auto dispatchExp = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Exp>;
    table[index(float64)] = launchFnKernel<double, double, Exp>;
    return table;
}();  
 
constexpr auto dispatchSqrt = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Sqrt>;
    table[index(float64)] = launchFnKernel<double, double, Sqrt>;
    return table;
}();

constexpr auto dispatchAbs = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Abs>;
    table[index(float64)] = launchFnKernel<double, double, Abs>;
    table[index(int32)] = launchFnKernel<int32_t, int32_t, Abs>;
    table[index(int64)] = launchFnKernel<int64_t, int64_t, Abs>;
    return table;
}();

constexpr auto dispatchSin = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Sin>;
    table[index(float64)] = launchFnKernel<double, double, Sin>;
    return table;
}();

constexpr auto dispatchCos = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Cos>;
    table[index(float64)] = launchFnKernel<double, double, Cos>;
    return table;
}();

constexpr auto dispatchTan = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Tan>;
    table[index(float64)] = launchFnKernel<double, double, Tan>;
    return table;
}();

constexpr auto dispatchSinh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Sinh>;
    table[index(float64)] = launchFnKernel<double, double, Sinh>;
    return table;
}();

constexpr auto dispatchCosh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Cosh>;
    table[index(float64)] = launchFnKernel<double, double, Cosh>;
    return table;
}();

constexpr auto dispatchTanh = []() {
    std::array<Kernel, index(TYPES)> table{}; table.fill(launchDefaultKernel);
    table[index(float32)] = launchFnKernel<float, float, Tanh>;
    table[index(float64)] = launchFnKernel<double, double, Tanh>;
    return table;
}(); 

} namespace cuda {
 
status log(tensor_t const* src, tensor_t* dst, stream_t stream) {   
    return dispatchLog[index(src->dtype)](src, dst, stream); 
}

status exp(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchExp[index(src->dtype)](src, dst, stream); 
}

status sqrt(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchSqrt[index(src->dtype)](src, dst, stream); 
}

status abs(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchAbs[index(src->dtype)](src, dst, stream); 
}

status sin(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchSin[index(src->dtype)](src, dst, stream); 
}

status cos(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchCos[index(src->dtype)](src, dst, stream); 
}

status tan(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchTan[index(src->dtype)](src, dst, stream); 
}

status sinh(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchSinh[index(src->dtype)](src, dst, stream); 
}

status cosh(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchCosh[index(src->dtype)](src, dst, stream); 
}

status tanh(tensor_t const* src, tensor_t* dst, stream_t stream) {  
    return dispatchTanh[index(src->dtype)](src, dst, stream); 
}

} // namespace cuda