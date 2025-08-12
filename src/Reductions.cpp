#include "Bindings.hpp"
#include "Reductions.hpp"
#include "Indexing.hpp"
#include "Callback.hpp"
#include "runtime/streams.h"
#include "cpu/argcmp.hpp" 
#ifdef CUDA
#include "cuda/argcmp.cuh"
#else 
namespace cuda { 
inline status argmax(const tensor_t*, tensor_t*, stream_t, uint8_t)  { throw std::runtime_error("CUDA not available"); }
inline status argmin(const tensor_t*, tensor_t*, stream_t, uint8_t)  { throw std::runtime_error("CUDA not available"); } 
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

} // namespace tannic