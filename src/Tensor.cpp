#include "runtime/tensor.h"
#include "Types.hpp"
#include "Tensor.hpp"   
#include <ostream>
#include <cstring>
#include <functional>
 
namespace tannic {

void tannic::Tensor::assign(std::byte const* value, std::ptrdiff_t offset) {      
    std::byte* target = static_cast<std::byte*>(buffer_->address()) + offset;   
    std::memcpy(target, value, dsizeof(dtype_)); 
} 

bool tannic::Tensor::compare(std::byte const* value, std::ptrdiff_t offset) const {  
    std::byte const* target = static_cast<std::byte const*>(buffer_->address()) + offset;  
    return std::memcmp(value, target, dsizeof(dtype_)) == 0;  
} 
 
static void print(std::ostream& ostream, void* address, type type) {
    switch (type) {
        case int8: {
            int8_t value = *static_cast<int8_t*>(address);
            ostream << static_cast<int>(value);   
            break;
        }
        case int16: {
            int16_t value = *static_cast<int16_t*>(address);
            ostream << value;
            break;
        }
        case int32: {
            int32_t value = *static_cast<int32_t*>(address);
            ostream << value;
            break;
        }
        case int64: {
            int64_t value = *static_cast<int64_t*>(address);
            ostream << value;
            break;
        }
        case float32: {
            float value = *static_cast<float*>(address);
            ostream << value;
            break;
        }
        case float64: {
            double value = *static_cast<double*>(address);
            ostream << value;
            break;
        }
        case complex64: {
            float* c = static_cast<float*>(address);
            ostream << "(" << c[0] << (c[1] >= 0 ? "+" : "") << c[1] << "j)";
            break;
        }
        case complex128: {
            double* c = static_cast<double*>(address);
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
        for (size_t i = 0; i < tensor->rank; ++i) {
            offset += indices[i] * tensor->strides[i];
        }
        return static_cast<char*>(tensor->address) + offset * dsizeof(tensor->dtype);
    };
    
    const auto print_recursive = [&](size_t dim, std::vector<size_t>& indices, const auto& self) -> void {
        if (dim == tensor->rank) {
            print(os, get_element(indices), tensor->dtype);
            return;
        }
        
        if (dim < tensor->rank) {
            os << "[";
        }
        
        for (size_t i = 0; i < tensor->shape[dim]; ++i) {
            indices[dim] = i;
            self(dim + 1, indices, self);
            
            if (i != tensor->shape[dim] - 1) {
                os << ", ";
            } 
            if (dim == 0 && i != tensor->shape[dim] - 1) {
                os << "\n       ";
            }
        }
        
        if (dim < tensor->rank) {
            os << "]";
        }
    };
    
    std::vector<size_t> indices(tensor->rank, 0);
    print_recursive(0, indices, print_recursive);
    
    os << " dtype=" << tensor->dtype << ", shape=(";
    
    for (size_t i = 0; i < tensor->rank; ++i) {
        os << tensor->shape[i];
        if (i != tensor->rank - 1) {
            os << ", ";
        }
    }
    return os << "))";
}

void print(const tensor_t* tensor) {
    std::cout << tensor;
} 

static inline tensor_t structure(Tensor const& tensor) {
    return tensor_t{
        .rank = tensor.rank(),
        .address = reinterpret_cast<void*>(tensor.bytes()),
        .shape = tensor.shape().address(),
        .strides = tensor.strides().address(), 
        .dtype = tensor.dtype()
    };
}

std::ostream& operator<<(std::ostream& ostream, Tensor tensor) {
    tensor_t ctensor = structure(tensor);
    ostream << &ctensor;
    return ostream;
}

} // namespace tannic