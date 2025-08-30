#include "cpu/cmps.hpp" 
#include <cstring>

namespace {

template<typename T, class Crt>
void cmpKernel(
    const T* src0_ptr, const strides_t& src0_strides,
    const T* src1_ptr, const strides_t& src1_strides,
    uint8_t* dst_bytes, const shape_t& dst_shape,
    uint8_t rank, size_t ne
) {
    Crt crt;
    size_t indices[8] = {0};  

    for (size_t linear_idx = 0; linear_idx < ne; ++linear_idx) {
        size_t off0 = 0, off1 = 0;
        for (size_t d = 0; d < rank; ++d) {
            off0 += indices[d] * src0_strides.sizes[d];
            off1 += indices[d] * src1_strides.sizes[d];
        }

        bool result = crt(src0_ptr[off0], src1_ptr[off1]); 
        dst_bytes[linear_idx / 8] |= (static_cast<uint8_t>(result) << (linear_idx % 8));
 
        for (int d = rank - 1; d >= 0; --d) {
            if (++indices[d] < dst_shape.sizes[d]) break;
            indices[d] = 0;
        }
    }
} 

struct EQ {
    template<class T>
    bool operator()(const T& a, const T& b) const noexcept {
        return a == b;
    }
};

struct NE {
    template<class T>
    bool operator()(const T& a, const T& b) const noexcept {
        return a != b;
    }
};

struct GT {
    template<class T>
    bool operator()(const T& a, const T& b) const noexcept {
        return a > b;
    }
};

struct GE {
    template<class T>
    bool operator()(const T& a, const T& b) const noexcept {
        return a >= b;
    }
};

struct LT {
    template<class T>
    bool operator()(const T& a, const T& b) const noexcept {
        return a < b;
    }
};

struct LE {
    template<class T>
    bool operator()(const T& a, const T& b) const noexcept {
        return a <= b;
    }
};

template<typename S, class Op>
status launchCmpKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    if (dst->rank == 0) {
        Op op{};
        bool result = op(*(const S*)(src0->address), *(const S*)(src1->address));
        if (result) {
            ((uint8_t*)(dst->address))[0] |= 1u;
        }
        return SUCCESS;
    }
 
    size_t ne = dst->size;
 
    std::memset(dst->address, 0, (ne + 7) / 8); 
    cmpKernel<S, Op>(
        (const S*)(src0->address), src0->strides,
        (const S*)(src1->address), src1->strides,
        (uint8_t*)(dst->address), dst->shape, dst->rank, ne
    );
    return SUCCESS;
}


} namespace cpu {

status eq(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    switch (src0->dtype) {
        case int8:    return launchCmpKernel<int8_t,   EQ>(src0, src1, dst);
        case int16:   return launchCmpKernel<int16_t,  EQ>(src0, src1, dst);
        case int32:   return launchCmpKernel<int32_t,  EQ>(src0, src1, dst);
        case int64:   return launchCmpKernel<int64_t,  EQ>(src0, src1, dst);
        case float32: return launchCmpKernel<float,    EQ>(src0, src1, dst);
        case float64: return launchCmpKernel<double,   EQ>(src0, src1, dst);
        default:      return UNSUPPORTED_DTYPE;
    }
}

status ne(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    switch (src0->dtype) {
        case int8:    return launchCmpKernel<int8_t,   NE>(src0, src1, dst);
        case int16:   return launchCmpKernel<int16_t,  NE>(src0, src1, dst);
        case int32:   return launchCmpKernel<int32_t,  NE>(src0, src1, dst);
        case int64:   return launchCmpKernel<int64_t,  NE>(src0, src1, dst);
        case float32: return launchCmpKernel<float,    NE>(src0, src1, dst);
        case float64: return launchCmpKernel<double,   NE>(src0, src1, dst);
        default:      return UNSUPPORTED_DTYPE;
    }
}

status gt(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    switch (src0->dtype) {
        case int8:    return launchCmpKernel<int8_t,   GT>(src0, src1, dst);
        case int16:   return launchCmpKernel<int16_t,  GT>(src0, src1, dst);
        case int32:   return launchCmpKernel<int32_t,  GT>(src0, src1, dst);
        case int64:   return launchCmpKernel<int64_t,  GT>(src0, src1, dst);
        case float32: return launchCmpKernel<float,    GT>(src0, src1, dst);
        case float64: return launchCmpKernel<double,   GT>(src0, src1, dst);
        default:      return UNSUPPORTED_DTYPE;
    }
}

status ge(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    switch (src0->dtype) {
        case int8:    return launchCmpKernel<int8_t,   GE>(src0, src1, dst);
        case int16:   return launchCmpKernel<int16_t,  GE>(src0, src1, dst);
        case int32:   return launchCmpKernel<int32_t,  GE>(src0, src1, dst);
        case int64:   return launchCmpKernel<int64_t,  GE>(src0, src1, dst);
        case float32: return launchCmpKernel<float,    GE>(src0, src1, dst);
        case float64: return launchCmpKernel<double,   GE>(src0, src1, dst);
        default:      return UNSUPPORTED_DTYPE;
    }
}

status lt(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    switch (src0->dtype) {
        case int8:    return launchCmpKernel<int8_t,   LT>(src0, src1, dst);
        case int16:   return launchCmpKernel<int16_t,  LT>(src0, src1, dst);
        case int32:   return launchCmpKernel<int32_t,  LT>(src0, src1, dst);
        case int64:   return launchCmpKernel<int64_t,  LT>(src0, src1, dst);
        case float32: return launchCmpKernel<float,    LT>(src0, src1, dst);
        case float64: return launchCmpKernel<double,   LT>(src0, src1, dst);
        default:      return UNSUPPORTED_DTYPE;
    }
}

status le(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    switch (src0->dtype) {
        case int8:    return launchCmpKernel<int8_t,   LE>(src0, src1, dst);
        case int16:   return launchCmpKernel<int16_t,  LE>(src0, src1, dst);
        case int32:   return launchCmpKernel<int32_t,  LE>(src0, src1, dst);
        case int64:   return launchCmpKernel<int64_t,  LE>(src0, src1, dst);
        case float32: return launchCmpKernel<float,    LE>(src0, src1, dst);
        case float64: return launchCmpKernel<double,   LE>(src0, src1, dst);
        default:      return UNSUPPORTED_DTYPE;
    }
}

} // namespace cpu
