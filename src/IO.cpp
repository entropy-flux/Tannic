#include <vector>
#include <ostream>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "bindings.hpp"
#include "runtime/streams.h"
#include "types.hpp"
#include "tensor.hpp"
#ifdef CUDA
#include "cuda/mem.cuh"
#endif

namespace tannic { 
  
struct element_ref {
    void*  address;    
    size_t bit_index;  
};

template <typename T>
static void print_floating(std::ostream& os, T v) {
    if (std::isinf(v)) {
        os << (v > 0 ? "inf" : "-inf");
    } else if (std::isnan(v)) {
        os << "nan";
    } else {
        os << v;
    }
}

static void print(std::ostream& os, element_ref elem, type dtype) {
    switch (dtype) {
        case boolean: {
            std::byte b = *static_cast<const std::byte*>(elem.address);
            bool v = ((b >> elem.bit_index) & std::byte(1)) != std::byte(0);
            os << (v ? "true" : "false");
            break;
        }
        case int8:    os << static_cast<int>(*static_cast<const int8_t*>(elem.address)); break;
        case int16:   os << *static_cast<const int16_t*>(elem.address); break;
        case int32:   os << *static_cast<const int32_t*>(elem.address); break;
        case int64:   os << *static_cast<const int64_t*>(elem.address); break;

        case float16:  print_floating(os, *static_cast<const float16_t*>(elem.address)); break;
        case bfloat16: print_floating(os, *static_cast<const bfloat16_t*>(elem.address)); break;
        case float32:  print_floating(os, *static_cast<const float*>(elem.address)); break;
        case float64:  print_floating(os, *static_cast<const double*>(elem.address)); break;

        case complex64: {
            const float* c = static_cast<const float*>(elem.address);
            os << "(";
            print_floating(os, c[0]);
            os << (c[1] >= 0 ? "+" : "");
            print_floating(os, c[1]);
            os << "j)";
            break;
        }
        case complex128: {
            const double* c = static_cast<const double*>(elem.address);
            os << "(";
            print_floating(os, c[0]);
            os << (c[1] >= 0 ? "+" : "");
            print_floating(os, c[1]);
            os << "j)";
            break;
        }

        default: os << "?";
    }
}

 
std::ostream& operator<<(std::ostream& os, const tensor_t* tensor) {
    os << "Tensor(";

    auto get_element = [&](const std::vector<size_t>& indices) -> element_ref {
        size_t elem_offset = 0;
        for (size_t dim = 0; dim < tensor->rank; ++dim) {
            elem_offset += indices[dim] * static_cast<size_t>(tensor->strides.sizes[dim]);
        }
        if (tensor->dtype == boolean) {
            size_t byte_index = elem_offset / 8;
            size_t bit_index  = elem_offset % 8;
            return { static_cast<void*>(static_cast<unsigned char*>(tensor->address) + byte_index),
                     bit_index };
        } else {
            size_t byte_off = elem_offset * dsizeof(tensor->dtype);
            return { static_cast<void*>(static_cast<unsigned char*>(tensor->address) + byte_off), 0 };
        }
    };

    const auto print_recursive = [&](size_t dim, std::vector<size_t>& idx, const auto& self) -> void {
        if (dim == tensor->rank) {
            print(os, get_element(idx), tensor->dtype);
            return;
        }

        char open_bracket = (iostyle == IOStyle::PyTorch) ? '[' : '{';
        char close_bracket = (iostyle == IOStyle::PyTorch) ? ']' : '}';

        os << open_bracket;
        for (size_t i = 0; i < tensor->shape.sizes[dim]; ++i) {
            idx[dim] = i;
            self(dim + 1, idx, self);
            if (i + 1 != tensor->shape.sizes[dim]) {
                os << ", ";
                if (dim == 0) os << "\n        ";
            }
        }
        os << close_bracket;
    };

    std::vector<size_t> indices(tensor->rank, 0);
    print_recursive(0, indices, print_recursive);

    os << ", dtype=" << tensor->dtype << ", shape=(";
    for (size_t dim = 0; dim < tensor->rank; ++dim) {
        os << tensor->shape.sizes[dim];
        if (dim + 1 != tensor->rank) os << ", ";
    }
    return os << "))";
}

// --- Tensor wrapper printing ---
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    const Environment& alloc = tensor.environment();
    shape_t shape{};
    strides_t strides{};

    for (int dimension = 0; dimension < tensor.rank(); ++dimension) {
        shape.sizes[dimension] = tensor.shape()[dimension];
        strides.sizes[dimension] = tensor.strides()[dimension];
    }

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
        os << &printable;
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
        os << &printable; 
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