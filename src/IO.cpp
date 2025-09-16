#include <vector>
#include <ostream>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "bindings.hpp"
#include "runtime/streams.h"
#include "types.hpp"
#include "shape.hpp"
#include "strides.hpp"
#include "tensor.hpp"
#include "cpu/io.hpp"
#ifdef CUDA
#include "cuda/mem.cuh"
#endif

namespace tannic {  
 
std::ostream& operator<<(std::ostream& os, Shape const& shape) {
    shape_t printable{.address = shape.address()};
    io::print(os, printable, shape.rank());
    return os;
} 

std::ostream& operator<<(std::ostream& os, Strides const& strides) {
    strides_t printable{.address = strides.address()};
    io::print(os, printable, strides.rank());
    return os;
} 

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    const Environment& alloc = tensor.environment();
    shape_t shape{tensor.shape().address()};
    strides_t strides{tensor.strides().address()}; 

    if (std::holds_alternative<Host>(alloc)) {
        Host const& resource = std::get<Host>(alloc);
        tensor_t printable{
            .address = (void*)(tensor.bytes()),
            .rank = tensor.rank(),
            .shape = shape,
            .strides = strides,
            .dtype = tensor.dtype(),
            .environment = {
                .environment = HOST,
                .resource = {.host = structure(resource)},
            }
        };

        io::print(os, &printable); 
    }  
#ifdef CUDA
    else {
        Device const& resource = std::get<Device>(alloc);
        void* buffer = std::malloc(tensor.nbytes());
        device_t dvc = structure(resource);
        cuda::copyDeviceToHost(&dvc,(const void*)(tensor.bytes()), buffer, tensor.nbytes());        
        tensor_t printable{
            .address = buffer,
            .rank = tensor.rank(),
            .shape = shape,
            .strides = strides,
            .dtype = tensor.dtype(),
            .environment = {
                .environment = DEVICE,
                .resource = {.device = structure(resource)},
            }
        };
        io::print(os, &printable); 
        std::free(buffer);
    } 
#else
    else {
        throw std::runtime_error("CUDA not supported!");
    } 
#endif
    return os;
}

} // namespace tannic