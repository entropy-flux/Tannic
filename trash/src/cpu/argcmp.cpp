#include "cpu/argcmp.h"

namespace cpu { 

template<typename T, class Argcmp>
void argcmpKernel(
    const T* src, 
    uint64_t* dst,
    const size_t* src_shape,
    const size_t* src_strides,
    const size_t* dst_shape,
    uint8_t rank,
    int64_t dim,
    bool keepdim,
    Argcmp argcmp
) { 
    if (dim == -1) {
        size_t num_elements = 1;
        for (uint8_t i = 0; i < rank; ++i) {
            num_elements *= src_shape[i];
        }
        
        T max_val = src[0];
        uint64_t max_idx = 0;
        for (size_t i = 1; i < num_elements; ++i) {
            if (argcmp(src[i], max_val)) {
                max_val = src[i];
                max_idx = i;
            }
        }
        dst[0] = max_idx;
        return;
    }
 
    size_t outer_size = 1;
    for (uint8_t i = 0; i < dim; ++i) {
        outer_size *= src_shape[i];
    }

    size_t inner_size = 1;
    for (uint8_t i = dim + 1; i < rank; ++i) {
        inner_size *= src_shape[i];
    }

    size_t dim_size = src_shape[dim];

    for (size_t i = 0; i < outer_size; ++i) {
        for (size_t k = 0; k < inner_size; ++k) {
            size_t src_offset = i * dim_size * inner_size + k;
            T current_val = src[src_offset];
            uint64_t current_idx = 0;

            for (size_t j = 1; j < dim_size; ++j) {
                size_t idx = src_offset + j * inner_size;
                if (argcmp(src[idx], current_val)) {
                    current_val = src[idx];
                    current_idx = j;
                }
            }

            size_t dst_offset = keepdim 
                ? i * inner_size + k  
                : (i * inner_size) + k;  
            dst[dst_offset] = current_idx;
        }
    }
} 

template<Argcmp A>
struct Cmp; 

template<typename S, Argcmp A>
bool argcmp(
    const tensor_t* src,
    tensor_t* dst,
    int64_t dim,
    bool keepdim 
) { 
    if (dim < -1 || dim >= src->rank) {
        return false;
    }
 
    if (dim == -1) { 
        if (keepdim && !(dst->rank == 1 && dst->shape[0] == 1)) {
            return false;
        }
        if (!keepdim && !(dst->rank == 0)) {
            return false;
        }
    } else { 
        for (uint8_t i = 0, j = 0; i < src->rank; ++i) {
            if (i == dim) {
                if (keepdim && dst->shape[j++] != 1) return false;
                continue;  
            }
            if (src->shape[i] != dst->shape[j++]) return false;
        }
    } 
    argcmpKernel<S>(
        static_cast<S*>(src->address),
        static_cast<uint64_t*>(dst->address),
        src->shape,
        src->strides,
        dst->shape,
        src->rank,
        dim,
        keepdim,
        Cmp<A>{}
    );

    return true;
}
  
struct GE {
    template<class A, class B>
    bool operator()(A&& a, B&& b) const noexcept {
        return a > b;
    }
};


struct LE {
    template<class A, class B>
    bool operator()(A&& a, B&& b) const noexcept {
        return a < b;
    }
};
  
template bool argcmp<int8_t, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<int16_t, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<int32_t, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<int64_t, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<float, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<double, Argcmp::ARGMAX>(const tensor_t*, tensor_t*, int64_t, bool);
  
template bool argcmp<int8_t, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<int16_t, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<int32_t, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<int64_t, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<float, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);
template bool argcmp<double, Argcmp::ARGMIN>(const tensor_t*, tensor_t*, int64_t, bool);

} // cpu