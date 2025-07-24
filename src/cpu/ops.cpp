#include "cpu.hpp"  
#include <stdexcept>
#include <vector>
#include <array>

template<typename S, typename D, class Op>
void unaryOpKernel(
    const void* src_ptr, const size_t* src_sz, const size_t* src_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
) {
    Op op{};
    const S* src = static_cast<const S*>(src_ptr);
    D* dst = static_cast<D*>(dst_ptr);

    if (rank == 0) {
        *dst = op(*src);
        return;
    }

    size_t total = 1;
    for (uint8_t i = 0; i < rank; ++i)
        total *= dst_sz[i];

    for (size_t idx = 0; idx < total; ++idx) {
        size_t offs = 0;
        for (uint8_t dim = 0; dim < rank; ++dim) {
            offs += cnt[dim] * src_ne[dim];
        }

        dst[idx] = op(src[offs]);
 
        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < dst_sz[dim])
                break;
            cnt[dim] = 0;
        }
    }
} 

template<typename S1, typename S2, typename D, class Op>
void binaryOpKernel(
    const void* src1_ptr, const size_t* src1_sz, const size_t* src1_ne,
    const void* src2_ptr, const size_t* src2_sz, const size_t* src2_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
) { 
    Op op{};
    const S1* src1 = static_cast<const S1*>(src1_ptr);
    const S2* src2 = static_cast<const S2*>(src2_ptr);
    D* dst = static_cast<D*>(dst_ptr);

    for (size_t idx = 0;; ++idx) {
        size_t offs1 = 0, offs2 = 0;

        for (uint8_t i = 0; i < rank; ++i) { 
            size_t idx1 = (src1_sz[i] == 1) ? 0 : cnt[i];
            size_t idx2 = (src2_sz[i] == 1) ? 0 : cnt[i];
            
            offs1 += idx1 * src1_ne[i];
            offs2 += idx2 * src2_ne[i];
        }

        dst[idx] = op(src1[offs1], src2[offs2]);

        bool done = false;
        for (int i = rank - 1; i >= 0; --i) {
            if (++cnt[i] < dst_sz[i])
                break;
            if (i == 0)
                done = true;
            cnt[i] = 0;
        }

        if (done) break;
    }
}  

struct Neg { 
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(-a)) {
        return -a;
    }
};

struct Add { 
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a + b)) {
        return a + b;
    }
};

struct Sub { 
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a - b)) {
        return a - b;
    }
};

struct Mul { 
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a * b)) {
        return a * b;
    }
}; 


using UnaryKernel = void(*)( 
    const void* src_ptr, const size_t* src_sz, const size_t* src_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
);    


using BinaryKernel = void(*)(
    const void* src1_ptr, const size_t* src1_sz, const size_t* src1_ne,
    const void* src2_ptr, const size_t* src2_sz, const size_t* src2_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
);  

void defaultUnaryKernel(
    const void* src_ptr, const size_t* src_sz, const size_t* src_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
) {
    throw std::runtime_error("Not supported dtype");
};

void defaultBinaryKernel( 
    const void* src1_ptr, const size_t* src1_sz, const size_t* src1_ne,
    const void* src2_ptr, const size_t* src2_sz, const size_t* src2_ne,
    void* dst_ptr, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt
) {
    throw std::runtime_error("Not supported dtype");
}; 

static constexpr inline int index(type type) {
    return static_cast<int>(type);
}


static constexpr inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}  

constexpr auto neg = []() {  
    std::array<UnaryKernel, index(TYPES)> table; table.fill(defaultUnaryKernel);
    table[index(int8)] = unaryOpKernel<int8_t, int8_t, Neg>;
    table[index(int16)] = unaryOpKernel<int16_t, int16_t, Neg>;
    table[index(int32)] = unaryOpKernel<int32_t, int32_t, Neg>;
    table[index(int64)] = unaryOpKernel<int64_t, int64_t, Neg>;
    table[index(float32)] = unaryOpKernel<float, float, Neg>;
    table[index(float64)] = unaryOpKernel<double, double, Neg>;
    return table;
}();


constexpr auto add = []() {
    std::array<BinaryKernel, index(TYPES, TYPES)> table; table.fill(defaultBinaryKernel);
    table[index(int8, int8)]   = binaryOpKernel<int8_t, int8_t, int8_t, Add>;
    table[index(int8, int16)]  = binaryOpKernel<int8_t, int16_t, int16_t, Add>; 
    table[index(int8, int32)]  = binaryOpKernel<int8_t, int32_t, int32_t, Add>;
    table[index(int8, int64)]  = binaryOpKernel<int8_t, int64_t, int64_t, Add>;

    table[index(int16, int8)]  = binaryOpKernel<int16_t, int8_t, int16_t, Add>;
    table[index(int16, int16)] = binaryOpKernel<int16_t, int16_t, int16_t, Add>;
    table[index(int16, int32)] = binaryOpKernel<int16_t, int32_t, int32_t, Add>;
    table[index(int16, int64)] = binaryOpKernel<int16_t, int64_t, int64_t, Add>;

    table[index(int32, int8)]  = binaryOpKernel<int32_t, int8_t, int32_t, Add>;
    table[index(int32, int16)] = binaryOpKernel<int32_t, int16_t, int32_t, Add>;
    table[index(int32, int32)] = binaryOpKernel<int32_t, int32_t, int32_t, Add>;
    table[index(int32, int64)] = binaryOpKernel<int32_t, int64_t, int64_t, Add>;

    table[index(int64, int8)]  = binaryOpKernel<int64_t, int8_t, int64_t, Add>;
    table[index(int64, int16)] = binaryOpKernel<int64_t, int16_t, int64_t, Add>;
    table[index(int64, int32)] = binaryOpKernel<int64_t, int32_t, int64_t, Add>;
    table[index(int64, int64)] = binaryOpKernel<int64_t, int64_t, int64_t, Add>;

    table[index(int32, float32)] = binaryOpKernel<int32_t, float, float, Add>;
    table[index(float32, int32)] = binaryOpKernel<float, int32_t, float, Add>;
    table[index(int32, float64)] = binaryOpKernel<int32_t, double, double, Add>;
    table[index(float64, int32)] = binaryOpKernel<double, int32_t, double, Add>;

    table[index(float32, float32)] = binaryOpKernel<float, float, float, Add>;
    table[index(float32, float64)] = binaryOpKernel<float, double, double, Add>;
    table[index(float64, float32)] = binaryOpKernel<double, float, double, Add>;
    table[index(float64, float64)] = binaryOpKernel<double, double, double, Add>;
    return table;
}(); 


constexpr auto sub = []() {
    std::array<BinaryKernel, index(TYPES, TYPES)> table; table.fill(defaultBinaryKernel);
    table[index(int8, int8)]   = binaryOpKernel<int8_t, int8_t, int8_t, Sub>;
    table[index(int8, int16)]  = binaryOpKernel<int8_t, int16_t, int16_t, Sub>; 
    table[index(int8, int32)]  = binaryOpKernel<int8_t, int32_t, int32_t, Sub>;
    table[index(int8, int64)]  = binaryOpKernel<int8_t, int64_t, int64_t, Sub>;

    table[index(int16, int8)]  = binaryOpKernel<int16_t, int8_t, int16_t, Sub>;
    table[index(int16, int16)] = binaryOpKernel<int16_t, int16_t, int16_t, Sub>;
    table[index(int16, int32)] = binaryOpKernel<int16_t, int32_t, int32_t, Sub>;
    table[index(int16, int64)] = binaryOpKernel<int16_t, int64_t, int64_t, Sub>;

    table[index(int32, int8)]  = binaryOpKernel<int32_t, int8_t, int32_t, Sub>;
    table[index(int32, int16)] = binaryOpKernel<int32_t, int16_t, int32_t, Sub>;
    table[index(int32, int32)] = binaryOpKernel<int32_t, int32_t, int32_t, Sub>;
    table[index(int32, int64)] = binaryOpKernel<int32_t, int64_t, int64_t, Sub>;

    table[index(int64, int8)]  = binaryOpKernel<int64_t, int8_t, int64_t, Sub>;
    table[index(int64, int16)] = binaryOpKernel<int64_t, int16_t, int64_t, Sub>;
    table[index(int64, int32)] = binaryOpKernel<int64_t, int32_t, int64_t, Sub>;
    table[index(int64, int64)] = binaryOpKernel<int64_t, int64_t, int64_t, Sub>;

    table[index(int32, float32)] = binaryOpKernel<int32_t, float, float, Sub>;
    table[index(float32, int32)] = binaryOpKernel<float, int32_t, float, Sub>;
    table[index(int32, float64)] = binaryOpKernel<int32_t, double, double, Sub>;
    table[index(float64, int32)] = binaryOpKernel<double, int32_t, double, Sub>;

    table[index(float32, float32)] = binaryOpKernel<float, float, float, Sub>;
    table[index(float32, float64)] = binaryOpKernel<float, double, double, Sub>;
    table[index(float64, float32)] = binaryOpKernel<double, float, double, Sub>;
    table[index(float64, float64)] = binaryOpKernel<double, double, double, Sub>;
    return table;
}();


constexpr auto mul = []() {
    std::array<BinaryKernel, index(TYPES, TYPES)> table; table.fill(defaultBinaryKernel);
    table[index(int8, int8)]   = binaryOpKernel<int8_t, int8_t, int8_t, Mul>;
    table[index(int8, int16)]  = binaryOpKernel<int8_t, int16_t, int16_t, Mul>; 
    table[index(int8, int32)]  = binaryOpKernel<int8_t, int32_t, int32_t, Mul>;
    table[index(int8, int64)]  = binaryOpKernel<int8_t, int64_t, int64_t, Mul>;

    table[index(int16, int8)]  = binaryOpKernel<int16_t, int8_t, int16_t, Mul>;
    table[index(int16, int16)] = binaryOpKernel<int16_t, int16_t, int16_t, Mul>;
    table[index(int16, int32)] = binaryOpKernel<int16_t, int32_t, int32_t, Mul>;
    table[index(int16, int64)] = binaryOpKernel<int16_t, int64_t, int64_t, Mul>;

    table[index(int32, int8)]  = binaryOpKernel<int32_t, int8_t, int32_t, Mul>;
    table[index(int32, int16)] = binaryOpKernel<int32_t, int16_t, int32_t, Mul>;
    table[index(int32, int32)] = binaryOpKernel<int32_t, int32_t, int32_t, Mul>;
    table[index(int32, int64)] = binaryOpKernel<int32_t, int64_t, int64_t, Mul>;

    table[index(int64, int8)]  = binaryOpKernel<int64_t, int8_t, int64_t, Mul>;
    table[index(int64, int16)] = binaryOpKernel<int64_t, int16_t, int64_t, Mul>;
    table[index(int64, int32)] = binaryOpKernel<int64_t, int32_t, int64_t, Mul>;
    table[index(int64, int64)] = binaryOpKernel<int64_t, int64_t, int64_t, Mul>;

    table[index(int32, float32)] = binaryOpKernel<int32_t, float, float, Mul>;
    table[index(float32, int32)] = binaryOpKernel<float, int32_t, float, Mul>;
    table[index(int32, float64)] = binaryOpKernel<int32_t, double, double, Mul>;
    table[index(float64, int32)] = binaryOpKernel<double, int32_t, double, Mul>;

    table[index(float32, float32)] = binaryOpKernel<float, float, float, Mul>;
    table[index(float32, float64)] = binaryOpKernel<float, double, double, Mul>;
    table[index(float64, float32)] = binaryOpKernel<double, float, double, Mul>;
    table[index(float64, float64)] = binaryOpKernel<double, double, double, Mul>;
    return table;
}(); 

namespace cpu {

void neg(tensor_t const* src, tensor_t* dst) { 
    std::vector<size_t> cnt(src->rank, 0); 
    ::neg[index(src->dtype)](
        src->address, src->shape, src->strides,
        dst->address, dst->shape, dst->strides,
        src->rank, cnt.data()
    );
}

void add(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) {
    std::vector<size_t> cnt(src1->rank, 0); 
    ::add[index(src1->dtype, src2->dtype)](
        src1->address, src1->shape, src1->strides,
        src2->address, src2->shape, src2->strides,
        dst->address, dst->shape, dst->strides,
        src1->rank, cnt.data()
    );
}

void sub(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) {
    std::vector<size_t> cnt(src1->rank, 0); 
    ::sub[index(src1->dtype, src2->dtype)](
        src1->address, src1->shape, src1->strides,
        src2->address, src2->shape, src2->strides,
        dst->address, dst->shape, dst->strides,
        src1->rank, cnt.data()
    );
} 

void mul(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) {
    std::vector<size_t> cnt(src1->rank, 0); 
    ::mul[index(src1->dtype, src2->dtype)](
        src1->address, src1->shape, src1->strides,
        src2->address, src2->shape, src2->strides,
        dst->address, dst->shape, dst->strides,
        src1->rank, cnt.data()
    );
}  

} // namespace cpu