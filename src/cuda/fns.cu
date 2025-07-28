#include <cstdint>   
#include <cstdint>
#include <array>
#include <stdexcept>
#include "cuda/exc.cuh"
#include "cuda/fns.cuh"
#include "cuda/streams.cuh"

template<typename S, typename D, class Fn>
__global__ void scalarFnKernel(const S* src, D* dst) {
    Fn fn;
    *dst = fn(*src);
}

template<typename S, typename D, class Fn>
__global__ void batchedFnKernel(
    const S* src, const size_t* src_sz, const size_t* src_ne,           
    D* dst, const size_t* dst_sz, const size_t* dst_ne, 
    uint8_t rank, size_t ne
) {
    Fn fn;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ne) return;

    size_t offs = 0;
    size_t rmn = idx;
    for (int dim = rank - 1; dim >= 0; --dim) {
        size_t dim_idx = rmn % dst_sz[dim];
        offs += dim_idx * src_ne[dim];
        rmn /= dst_sz[dim];
    }

    dst[idx] = fn(src[offs]);
}

template<typename S, typename D, class Fn>
void launchFnKernel(const tensor_t* src, tensor_t* dst, cudaStream_t stream = 0) { 
    if (src->rank == 0) {
        scalarFnKernel<S, D, Fn><<<1, 1, 0,stream>>>(
            (const S*)(src->address),
            (D*)(dst->address)
        ); 
    } 
    
    else {
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape[dim];
        }

        size_t blockSize = 256;
        size_t gridSize = (ne + blockSize - 1) / blockSize;

        batchedFnKernel<S, D, Fn><<<gridSize, blockSize, 0, stream>>>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne
        ); 
    } 
}

void launchDefaultKernel(const tensor_t* src, tensor_t* dst, cudaStream_t) {
    exit(EXIT_FAILURE);
};  

struct Log { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(log(a))) {
        return log(a);
    }
};
 
struct Exp { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(exp(a))) {
        return exp(a);
    }
};
  
struct Sqrt { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(sqrt(a))) {
        return sqrt(a);
    }
};
 
struct Abs { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(abs(a))) {
        return abs(a);
    }
};
 
struct Sin { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(sin(a))) {
        return sin(a);
    }
};
 
struct Cos { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(cos(a))) {
        return cos(a);
    }
};

struct Tan { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(tan(a))) {
        return tan(a);
    }
}; 

struct Sinh { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(sinh(a))) {
        return sinh(a);
    }
};
 
struct Cosh{ 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(cosh(a))) {
        return cosh(a);
    }
};
 
struct Tanh { 
    template<class A>
    __device__ auto operator()(A&& a) const noexcept(noexcept(tanh(a))) {
        return tanh(a);
    }
};    

constexpr static inline int index(type type) {
    return static_cast<int>(type);
}  
 
using Kernel = void(*)(const tensor_t*, tensor_t*, cudaStream_t);       

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

namespace cuda {
 
void log(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchLog[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void exp(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchExp[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void sqrt(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchSqrt[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void abs(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchAbs[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void sin(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchSin[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void cos(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchCos[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void tan(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchTan[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void sinh(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchSinh[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void cosh(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchCosh[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

void tanh(device_t const* dvc, tensor_t const* src, tensor_t* dst) { 
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(dvc->id);
    dispatchTanh[index(src->dtype)](src, dst, stream);
    streams.put(dvc->id, stream);
}

} // namespace cuda