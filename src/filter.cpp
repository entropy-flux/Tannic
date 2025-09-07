#include "filter.hpp"
#include "callback.hpp"
#include "tensor.hpp"
#include "cpu/triang.hpp"

#ifdef CUDA
#include "cuda/triang.cuh"
#else
namespace cuda {
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
inline status triu(const tensor_t*, tensor_t*, stream_t, int k) { throw std::runtime_error("CUDA not available"); }
inline status tril(const tensor_t*, tensor_t*, stream_t, int k) { throw std::runtime_error("CUDA not available"); }
}
#endif

namespace tannic::expression {

void Triangular::forward(Tensor const& input, Tensor& output) const {
    switch (this->position) {
        case Position::Upper: {  
            Callback callback(
                [&](const tensor_t* src, tensor_t* dst) -> status { return cpu::triu(src, dst, offset); },
                [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::triu(src, dst, stream, offset); }
            );
            callback(input, output); 
            break;
        } 
        
        case Position::Lower: { 
            Callback callback(
                [&](const tensor_t* src, tensor_t* dst) -> status { return cpu::tril(src, dst, offset); },
                [&](const tensor_t* src, tensor_t* dst, stream_t stream) -> status { return cuda::tril(src, dst, stream, offset); }
            );
            callback(input, output); 
            break; 
        }

        default:
            throw Exception("Unsupported position");
            break;
        }
}

}