#include "io.hpp"
#include "types.hpp"
#include <vector>

namespace {
 
struct element_t {
    void*  address;    
    size_t bit_index;  
};

template <typename T>
void print(std::ostream& os, T value) {
    if (std::isinf(value)) {
        os << (value > 0 ? "inf" : "-inf");
    } else if (std::isnan(value)) {
        os << "nan";
    } else {
        os << value;
    }
}

void print(std::ostream& os, element_t element, type dtype) {
    switch (dtype) {
        case boolean: {
            std::byte b = *static_cast<const std::byte*>(element.address);
            bool value = ((b >> element.bit_index) & std::byte(1)) != std::byte(0);
            os << (value ? "true" : "false");
            break;
        }
        case int8:    os << static_cast<int>(*static_cast<const int8_t*>(element.address)); break;
        case int16:   os << *static_cast<const int16_t*>(element.address); break;
        case int32:   os << *static_cast<const int32_t*>(element.address); break;
        case int64:   os << *static_cast<const int64_t*>(element.address); break;

        case float16:  print(os, *static_cast<const float16_t*>(element.address)); break;
        case bfloat16: print(os, *static_cast<const bfloat16_t*>(element.address)); break;
        case float32:  print(os, *static_cast<const float*>(element.address)); break;
        case float64:  print(os, *static_cast<const double*>(element.address)); break;

        case complex64: {
            const float* c = static_cast<const float*>(element.address);
            os << "(";
            print(os, c[0]);
            os << (c[1] >= 0 ? "+" : "");
            print(os, c[1]);
            os << "j)";
            break;
        }
        case complex128: {
            const double* c = static_cast<const double*>(element.address);
            os << "(";
            print(os, c[0]);
            os << (c[1] >= 0 ? "+" : "");
            print(os, c[1]);
            os << "j)";
            break;
        }

        default: os << "?";
    }
}

} namespace io {

void print(std::ostream& os, const shape_t& shape, uint8_t rank) {
    os << "Shape(";
    for (uint8_t dimension = 0; dimension < rank; ++dimension) {
        os << shape.address[dimension];
        if (dimension + 1 < rank) {
            os << ", ";
        }
    }
    os << ")"; 
}

void print(std::ostream& os, const strides_t& strides, uint8_t rank) {
    os << "Strides(";
    for (uint8_t dimension = 0; dimension < rank; ++dimension) {
        os << strides.address[dimension];
        if (dimension + 1 < rank) {
            os << ", ";
        }
    }
    os << ")"; 
}  

void print(std::ostream& os, const tensor_t* tensor) { 
    os << "Tensor(";

    auto get_element = [&](const std::vector<size_t>& indices) -> element_t {
        size_t elem_offset = 0;
        for (size_t dim = 0; dim < tensor->rank; ++dim) {
            elem_offset += indices[dim] * static_cast<size_t>(tensor->strides.address[dim]);
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

        char open  =  '[';
        char close =  ']';

        os << open;
        for (size_t i = 0; i < tensor->shape.address[dim]; ++i) {
            idx[dim] = i;
            self(dim + 1, idx, self);
            if (i + 1 != tensor->shape.address[dim]) {
                os << ", ";
                if (dim == 0) os << "\n        ";
            }
        }
        os << close;
    };

    std::vector<size_t> indices(tensor->rank, 0);
    print_recursive(0, indices, print_recursive);

    os << ", dtype=" << tensor->dtype << ", shape=(";
    for (size_t dim = 0; dim < tensor->rank; ++dim) {
        os << tensor->shape.address[dim];
        if (dim + 1 != tensor->rank) os << ", ";
    }
    os << "))";
}

};