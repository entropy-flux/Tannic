#include <vector>
#include <ostream>
#include "Bindings.hpp"
#include "runtime/streams.h"
#include "Types.hpp"
#include "Tensor.hpp"   

namespace tannic {  

static void print(std::ostream& ostream, void* address, type type) {
    switch (type) {
        case int8: {
            int8_t value = *(int8_t*)(address);
            ostream << (int)(value);
            break;
        }
        case int16: {
            int16_t value = *(int16_t*)(address);
            ostream << value;
            break;
        }
        case int32: {
            int32_t value = *(int32_t*)(address);
            ostream << value;
            break;
        }
        case int64: {
            int64_t value = *(int64_t*)(address);
            ostream << value;
            break;
        }
        case float32: {
            float value = *(float*)(address);
            ostream << value;
            break;
        }
        case float64: {
            double value = *(double*)(address);
            ostream << value;
            break;
        }
        case complex64: {
            float* c = (float*)(address);
            ostream << "(" << c[0] << (c[1] >= 0 ? "+" : "") << c[1] << "j)";
            break;
        }
        case complex128: {
            double* c = (double*)(address);
            ostream << "(" << c[0] << (c[1] >= 0 ? "+" : "") << c[1] << "j)";
            break;
        }
        default: ostream << "?";
    }
}
 
std::ostream& operator<<(std::ostream& os, const tensor_t* tensor) {   
    os << "Tensor(";
    auto get_element = [&](const std::vector<size_t>& indices) -> void* {
        size_t offset = 0;
        for (size_t dim = 0; dim < tensor->rank; ++dim) {
            offset += indices[dim] * tensor->strides.sizes[dim];
        }
        return (char*)(tensor->address) + offset * dsizeof(tensor->dtype);
    };
    
    const auto print_recursive = [&](size_t dim, std::vector<size_t>& indices, const auto& self) -> void {
        if (dim == tensor->rank) {
            print(os, get_element(indices), tensor->dtype);
            return;
        }
        
        if (dim < tensor->rank) {
            os << "[";
        }
        
        for (size_t sz = 0; sz < tensor->shape.sizes[dim]; ++sz) {
            indices[dim] = sz;
            self(dim + 1, indices, self);
            
            if (sz != tensor->shape.sizes[dim] - 1) {
                os << ", ";
            } 
            if (dim == 0 && sz != tensor->shape.sizes[dim] - 1) {
                os << "\n        ";
            }
        }
        
        if (dim < tensor->rank) {
            os << "]";
        }
    };
    
    std::vector<size_t> indices(tensor->rank, 0);
    print_recursive(0, indices, print_recursive);
    
    os << " dtype=" << tensor->dtype << ", shape=(";
    
    for (size_t dim = 0; dim < tensor->rank; ++dim) {
        os << tensor->shape.sizes[dim];
        if (dim != tensor->rank - 1) {
            os << ", ";
        }
    }
    return os << "))";
} 

std::ostream& operator<<(std::ostream& ostream, Tensor const& tensor) {
    tensor_t printable = structure(tensor); 
    allocator_t allocator = structure(tensor.allocator());
    if(allocator.environment == HOST) {
        ostream << &printable; 
    } 

    else {
        throw std::runtime_error("IO not implemented for cuda tensor, but will be!");
    } 
    return ostream;
}

}