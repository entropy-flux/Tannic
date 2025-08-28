#include <vector>
#include <ostream>
#include <cstddef>
#include <cstdint>
#include "bindings.hpp"
#include "runtime/streams.h"
#include "types.hpp"
#include "tensor.hpp"

namespace tannic {
 
struct element_ref {
    void*  address;    // Pointer to the underlying byte in storage
    size_t bit_index;  // Valid only if dtype == boolean (0..7); otherwise 0
};
 
static void print(std::ostream& os, element_ref elem, type dtype) {
    switch (dtype) {
        case boolean: {
            std::byte b = *static_cast<const std::byte*>(elem.address);
            bool v = ((b >> elem.bit_index) & std::byte(1)) != std::byte(0);
            os << (v ? "true" : "false");
            break;
        }
        case int8: {
            auto v = *static_cast<const int8_t*>(elem.address);
            os << static_cast<int>(v);
            break;
        }
        case int16: {
            auto v = *static_cast<const int16_t*>(elem.address);
            os << v;
            break;
        }
        case int32: {
            auto v = *static_cast<const int32_t*>(elem.address);
            os << v;
            break;
        }
        case int64: {
            auto v = *static_cast<const int64_t*>(elem.address);
            os << v;
            break;
        }
        case float32: {
            auto v = *static_cast<const float*>(elem.address);
            os << v;
            break;
        }
        case float64: {
            auto v = *static_cast<const double*>(elem.address);
            os << v;
            break;
        }
        case complex64: {
            const float* c = static_cast<const float*>(elem.address);
            os << "(" << c[0] << (c[1] >= 0 ? "+" : "") << c[1] << "j)";
            break;
        }
        case complex128: {
            const double* c = static_cast<const double*>(elem.address);
            os << "(" << c[0] << (c[1] >= 0 ? "+" : "") << c[1] << "j)";
            break;
        }
        default:
            os << "?";
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
            return { static_cast<void*>(
                         static_cast<unsigned char*>(tensor->address) + byte_index),
                     bit_index };
        } else {
            // dsizeof(dtype) must be > 0 for non-boolean types
            size_t byte_off = elem_offset * dsizeof(tensor->dtype);
            return { static_cast<void*>(
                         static_cast<unsigned char*>(tensor->address) + byte_off),
                     0 };
        }
    };
 
    const auto print_recursive = [&](size_t dim, std::vector<size_t>& idx, const auto& self) -> void {
        if (dim == tensor->rank) {
            print(os, get_element(idx), tensor->dtype);
            return;
        }

        os << "[";
        for (size_t i = 0; i < tensor->shape.sizes[dim]; ++i) {
            idx[dim] = i;
            self(dim + 1, idx, self);
            if (i + 1 != tensor->shape.sizes[dim]) {
                os << ", ";
                if (dim == 0) os << "\n        ";
            }
        }
        os << "]";
    };

    std::vector<size_t> indices(tensor->rank, 0);
    print_recursive(0, indices, print_recursive);

    os << " dtype=" << tensor->dtype << ", shape=(";
    for (size_t dim = 0; dim < tensor->rank; ++dim) {
        os << tensor->shape.sizes[dim];
        if (dim + 1 != tensor->rank) os << ", ";
    }
    return os << "))";
}
 
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    tensor_t printable = structure(tensor);
    environment_t env  = structure(tensor.environment());

    if (env.environment == HOST) {
        os << &printable;
    } else {
        // you can later implement a device-side dump (copy to host then print)
        throw std::runtime_error("IO not implemented for cuda tensor, but will be!");
    }
    return os;
}

} // namespace tannic