#include "comparisons.hpp"
#include "callback.hpp"
#include "runtime/tensor.h"
#include "runtime/status.h" 
#include "runtime/graph.h" 
#include "runtime/streams.h"
#include "cpu/cmps.hpp"
#include "cpu/val.hpp"

#ifdef CUDA 
#include "cuda/val.cuh"
#else 
namespace cuda {
bool allclose(const tensor_t*, const tensor_t*, stream_t, double rtol = 1e-5, double atol = 1e-8) {throw std::runtime_error("CUDA not available");}
} 
#endif

namespace cuda {
    
using tannic::status;
using tannic::tensor_t;
using tannic::stream_t;
using tannic::device_t;
status eq(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status ne(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status gt(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status ge(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status lt(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
status le(const tensor_t*, const tensor_t*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}

} namespace tannic::comparison { 

void EQ::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::eq, cuda::eq);
    callback(first, second, target);
}

void NE::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::ne, cuda::ne);
    callback(first, second, target);
}

void GT::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::gt, cuda::gt);
    callback(first, second, target);
}

void GE::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::ge, cuda::ge);
    callback(first, second, target);
}

void LT::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::lt, cuda::lt);
    callback(first, second, target);
}

void LE::forward(Tensor const& first, Tensor const& second, Tensor& target) const {
    Callback callback(cpu::le, cuda::le);
    callback(first, second, target);
}

bool allclose(Tensor const& first, Tensor const& second, double rtol, double atol) {
    if (first.shape() != second.shape())
        throw Exception("allclose: shape mismatch"); 

    if (std::holds_alternative<Host>(first.environment())) {
        if (!std::holds_alternative<Host>(second.environment()))
            throw Exception("Cannot compare tensors from different environmments");
        tensor_t* fst = get_tensor(first.id());
        tensor_t* sec = get_tensor(second.id());
        return cpu::allclose(fst, sec, rtol, atol);
    } else if (std::holds_alternative<Device>(first.environment())) {
        if (!std::holds_alternative<Device>(second.environment()))
            throw Exception("Cannot compare tensors from different environmments");
        
        tensor_t* fst = get_tensor(first.id());
        tensor_t* sec = get_tensor(second.id());

        device_t device = fst->environment.resource.device;
        stream_t stream = pop_stream(&device);
        
        bool result = cuda::allclose(fst, sec, stream, rtol, atol);
        
        put_stream(&device, stream);
        
        return result;
    } else {
        throw std::runtime_error("Unsupported environment");
    }
}

}