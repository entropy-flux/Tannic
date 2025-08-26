#include "tensor.hpp" 
#include "convolutions.hpp"  
#include "callback.hpp" 
#include "cpu/conv.hpp"    
#ifdef CUDA
#include "cuda/conv.cuh"    
#else   
namespace cuda {  
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;

status conv1d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, stream_t stream, const size_t pad, const size_t stride) { throw std::runtime_error("CUDA not available"); }
status conv2d(const tensor_t*, const tensor_t*, tensor_t*, stream_t, const size_t pad[2], const size_t stride[2])  { throw std::runtime_error("CUDA not available"); }
} // namespace cuda
#endif

namespace tannic::transformation {

void Convolution1D::forward(Tensor const& signal, Tensor const& kernel, Tensor& output) const { 
    Callback callback(
        [&](const tensor_t* sgnal,const tensor_t* knel, tensor_t* dst) -> status { return cpu::conv1d(sgnal, knel, dst, padding[0], strides[0]); },
        [&](const tensor_t* sgnal,const tensor_t* knel, tensor_t* dst, stream_t stream) -> status { return cuda::conv1d(sgnal, knel, dst, stream, padding[0], strides[0]); }
    );
    callback(signal, kernel, output);
}

void Convolution2D::forward(Tensor const& signal, Tensor const& kernel, Tensor& output) const {  
    Callback callback(
        [&](const tensor_t* sgnal,const tensor_t* knel, tensor_t* dst) -> status { return cpu::conv2d(sgnal, knel, dst, padding.data(), strides.data()); },
        [&](const tensor_t* sgnal,const tensor_t* knel, tensor_t* dst, stream_t stream) -> status { return cuda::conv2d(sgnal, knel, dst, stream, padding.data(), strides.data()); }
    );
    callback(signal, kernel, output);
} 

}