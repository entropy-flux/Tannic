#include "bindings.hpp"
#include "reductions.hpp"
#include "indexing.hpp"
#include "callback.hpp"
#include "runtime/streams.h"
#include "cpu/argred.hpp" 
#ifdef CUDA
#include "cuda/argred.cuh" 
#else 
namespace cuda { 
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
inline status argmax(const tensor_t*, tensor_t*, stream_t, uint8_t)  { throw std::runtime_error("CUDA not available"); }
inline status argmin(const tensor_t*, tensor_t*, stream_t, uint8_t)  { throw std::runtime_error("CUDA not available"); } 
inline status argsum(const tensor_t*, tensor_t*, stream_t, uint8_t)  { throw std::runtime_error("CUDA not available"); } 
inline status argmean(const tensor_t*, tensor_t*, stream_t, uint8_t) { throw std::runtime_error("CUDA not available"); }
}
#endif  

namespace tannic {    

void expression::Argmax::forward(Tensor const& input, Tensor& output) const {   
    Callback callback(
        [&](const tensor_t* src, tensor_t* dst) -> status { return cpu::argmax(src, dst, axis); },
        [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::argmax(src, dst, stream, axis); }
    );
    callback(input, output);
}

void expression::Argmin::forward(Tensor const& input, Tensor& output) const {    
    Callback callback(
        [&](const tensor_t* src, tensor_t* dst) -> status {return cpu::argmin(src, dst, axis); },
        [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::argmin(src, dst, stream, axis); }
    );
    callback(input, output);
}

void expression::Argsum::forward(Tensor const& input, Tensor& output) const {   
    Callback callback(
        [&](const tensor_t* src, tensor_t* dst) -> status {return cpu::argsum(src, dst, axis); },
        [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::argsum(src, dst, stream, axis); }
    );
    callback(input, output);
}

void expression::Argmean::forward(Tensor const& input, Tensor& output) const {   
    Callback callback(
        [&](const tensor_t* src, tensor_t* dst) -> status {return cpu::argmean(src, dst, axis); },
        [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::argmean(src, dst, stream, axis); }
    );
    callback(input, output);
}


} // namespace tannic