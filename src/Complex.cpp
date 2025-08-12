#include "Complex.hpp"
#include "Bindings.hpp"
#include "Callback.hpp"
#include "cpu/complex.hpp"
#include "runtime/streams.h"
#include "runtime/graph.h"
#ifdef CUDA
#include "cuda/complex.cuh"
#else
namespace cuda {
inline status view_as_cartesian(tensor_t const*, tensor_t const*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
inline status view_as_polar(tensor_t const*, tensor_t const*, tensor_t*, stream_t) {throw std::runtime_error("CUDA not available");}
}
#endif

namespace tannic::expression {   

void Cartesian::forward(Tensor const& real, Tensor const& imaginary, Tensor& complex) {
    Callback callback(cpu::view_as_cartesian, cuda::view_as_cartesian);
    callback(real, imaginary, complex);
}

void Polar::forward(Tensor const& rho, Tensor const& theta, Tensor& complex) { 
    Callback callback(cpu::view_as_polar, cuda::view_as_polar);
    callback(rho, theta, complex);
}

}