#include <stdexcept>
#include <vector>
#include <array>
#include "cpu/ops.hpp" 

template<typename S, typename D, class Op>
void scalarUnaryOpKernel(
    const S* src_ptr, D* dst_ptr
) {
    Op op;
    *dst_ptr = op(*src_ptr);
}    

template<typename S0, typename S1, typename D, class Op>
void scalarBinaryOpKernel(
    const S0* src0_ptr,  
    const S1* src1_ptr, 
    D* dst_ptr
) {
    Op op;
    *dst_ptr = op(*src0_ptr, *src1_ptr);
}  
  
template<typename S, typename D, class Op>
void batchedUnaryOpKernel( 
    const S* src_ptr, const uint32_t* src_sz, const int64_t* src_ne,           
    D* dst_ptr, const uint32_t* dst_sz, const int64_t* dst_ne, 
    uint8_t rank, size_t ne
) { 
    Op op;  
    size_t cnt[8] = {0};
    for (size_t idx = 0; idx < ne; ++idx) {
        size_t offs = 0;
        for (int dim = 0; dim < rank; ++dim) {
            offs += cnt[dim] * src_ne[dim];
        }

        dst_ptr[idx] = op(src_ptr[offs]);

        for (int dim = rank - 1; dim >= 0; --dim) {
            if (++cnt[dim] < dst_sz[dim])
                break;
            cnt[dim] = 0;
        }
    } 
}     

template<typename S0, typename S1, typename D, class Op>
void batchedBinaryOpKernel(
    const S0* src0_ptr, const uint32_t* src0_sz, const int64_t* src0_ne,
    const S1* src1_ptr, const uint32_t* src1_sz, const int64_t* src1_ne,
    D* dst_ptr, const uint32_t* dst_sz, const int64_t* dst_ne,
    uint8_t rank
) { 
    Op op{};
    size_t cnt[8] = {0}; 
    for (size_t idx = 0;; ++idx) {
        size_t offs0 = 0, offs1 = 0;

        for (uint8_t i = 0; i < rank; ++i) { 
            size_t idx0 = (src0_sz[i] == 1) ? 0 : cnt[i];
            size_t idx1 = (src1_sz[i] == 1) ? 0 : cnt[i];
            
            offs0 += idx0 * src0_ne[i];
            offs1 += idx1 * src1_ne[i];
        }

        dst_ptr[idx] = op(src0_ptr[offs0], src1_ptr[offs1]);

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

template<typename S, typename D, class Op>
void launchUnaryOpKernel(const tensor_t* src, tensor_t* dst) {
    if (src->rank == 0) {
        scalarUnaryOpKernel<S, D, Op>(
            (const S*)(src->address), 
            (D*)(dst->address)
        ); 
    } 
    
    else {    
        size_t ne = 1;
        for (uint8_t dim = 0; dim < src->rank; ++dim) {
            ne *= dst->shape[dim];
        }

        batchedUnaryOpKernel<S, D, Op>(
            (const S*)(src->address), src->shape, src->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            src->rank, ne
        ); 
    } 
    return;
}        
 
template<typename S0, typename S1, typename D, class Op>
void launchBinaryOpKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    if (dst->rank == 0) {
        scalarBinaryOpKernel<S0, S1, D, Op>(
            (const S0*)(src0->address), 
            (const S1*)(src1->address),  
            (D*)(dst->address)
        ); 
    } 
    
    else {     
        batchedBinaryOpKernel<S0, S1, D, Op>(
            (const S0*)(src0->address), src0->shape, src0->strides,
            (const S1*)(src1->address), src1->shape, src1->strides,
            (D*)(dst->address), dst->shape, dst->strides,
            dst->rank
        ); 
    } 
    return;
}       

void launchDefaultUnaryOpKernel(const tensor_t* src, tensor_t* dst) {
    throw std::runtime_error("Not supported dtype");
};  

void launchDefaultBinaryOpKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    throw std::runtime_error("Not supported dtype");
};    

 
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

constexpr static inline int index(type type) {
    return static_cast<int>(type);
}


constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}  


using UnaryOpKernel = void(*)( const tensor_t* src, tensor_t* dst);      
using BinaryOpKernel = void(*)( const tensor_t*, const tensor_t*, tensor_t*);      

constexpr auto dispatchNeg = []() {  
    std::array<UnaryOpKernel, index(TYPES)> table; table.fill(launchDefaultUnaryOpKernel);
    table[index(int8)] = launchUnaryOpKernel<int8_t, int8_t, Neg>;
    table[index(int16)] = launchUnaryOpKernel<int16_t, int16_t, Neg>;
    table[index(int32)] = launchUnaryOpKernel<int32_t, int32_t, Neg>;
    table[index(int64)] = launchUnaryOpKernel<int64_t, int64_t, Neg>;
    table[index(float32)] = launchUnaryOpKernel<float, float, Neg>;
    table[index(float64)] = launchUnaryOpKernel<double, double, Neg>;
    return table;
}();      

constexpr auto dispatchAdd = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; table.fill(launchDefaultBinaryOpKernel);
    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Add>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Add>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Add>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Add>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Add>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Add>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Add>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Add>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Add>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Add>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Add>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Add>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Add>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Add>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Add>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Add>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Add>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Add>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Add>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Add>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Add>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Add>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Add>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Add>;
    return table;
}();  

constexpr auto dispatchSub = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; table.fill(launchDefaultBinaryOpKernel);
    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Sub>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Sub>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Sub>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Sub>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Sub>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Sub>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Sub>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Sub>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Sub>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Sub>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Sub>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Sub>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Sub>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Sub>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Sub>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Sub>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Sub>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Sub>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Sub>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Sub>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Sub>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Sub>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Sub>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Sub>;
    return table;
}(); 

constexpr auto dispatchMul = []() {
    std::array<BinaryOpKernel, index(TYPES, TYPES)> table; table.fill(launchDefaultBinaryOpKernel);
    table[index(int8, int8)]   = launchBinaryOpKernel<int8_t, int8_t, int8_t, Mul>;
    table[index(int8, int16)]  = launchBinaryOpKernel<int8_t, int16_t, int16_t, Mul>; 
    table[index(int8, int32)]  = launchBinaryOpKernel<int8_t, int32_t, int32_t, Mul>;
    table[index(int8, int64)]  = launchBinaryOpKernel<int8_t, int64_t, int64_t, Mul>;

    table[index(int16, int8)]  = launchBinaryOpKernel<int16_t, int8_t, int16_t, Mul>;
    table[index(int16, int16)] = launchBinaryOpKernel<int16_t, int16_t, int16_t, Mul>;
    table[index(int16, int32)] = launchBinaryOpKernel<int16_t, int32_t, int32_t, Mul>;
    table[index(int16, int64)] = launchBinaryOpKernel<int16_t, int64_t, int64_t, Mul>;

    table[index(int32, int8)]  = launchBinaryOpKernel<int32_t, int8_t, int32_t, Mul>;
    table[index(int32, int16)] = launchBinaryOpKernel<int32_t, int16_t, int32_t, Mul>;
    table[index(int32, int32)] = launchBinaryOpKernel<int32_t, int32_t, int32_t, Mul>;
    table[index(int32, int64)] = launchBinaryOpKernel<int32_t, int64_t, int64_t, Mul>;

    table[index(int64, int8)]  = launchBinaryOpKernel<int64_t, int8_t, int64_t, Mul>;
    table[index(int64, int16)] = launchBinaryOpKernel<int64_t, int16_t, int64_t, Mul>;
    table[index(int64, int32)] = launchBinaryOpKernel<int64_t, int32_t, int64_t, Mul>;
    table[index(int64, int64)] = launchBinaryOpKernel<int64_t, int64_t, int64_t, Mul>;

    table[index(int32, float32)] = launchBinaryOpKernel<int32_t, float, float, Mul>;
    table[index(float32, int32)] = launchBinaryOpKernel<float, int32_t, float, Mul>;
    table[index(int32, float64)] = launchBinaryOpKernel<int32_t, double, double, Mul>;
    table[index(float64, int32)] = launchBinaryOpKernel<double, int32_t, double, Mul>;

    table[index(float32, float32)] = launchBinaryOpKernel<float, float, float, Mul>;
    table[index(float32, float64)] = launchBinaryOpKernel<float, double, double, Mul>;
    table[index(float64, float32)] = launchBinaryOpKernel<double, float, double, Mul>;
    table[index(float64, float64)] = launchBinaryOpKernel<double, double, double, Mul>;
    return table;
}(); 

namespace cpu {

void neg(tensor_t const* src, tensor_t* dst) {  
    dispatchNeg[index(src->dtype)](src, dst);
}

void add(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) { 
    dispatchAdd[index(src1->dtype, src2->dtype)](src1, src2, dst);
}

void sub(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) { 
    dispatchSub[index(src1->dtype, src2->dtype)](src1, src2, dst);
} 

void mul(tensor_t const* src1, tensor_t const* src2, tensor_t* dst) { 
    dispatchMul[index(src1->dtype, src2->dtype)](src1, src2, dst);
}  

} // namespace cpu