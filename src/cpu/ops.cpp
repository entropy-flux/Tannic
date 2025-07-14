#include "cpu/ops.hpp" 
#include <vector>
 
template<typename S, typename D, class Op>
void unaryOp(
    const S* src, const size_t* src_sz, const size_t* src_ne,
    D* dst, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt, Op op
) {
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
void binaryOp(
    const S1* src1, const size_t* src1_sz, const size_t* src1_ne,
    const S2* src2, const size_t* src2_sz, const size_t* src2_ne,
    D* dst, const size_t* dst_sz, const size_t* dst_ne,
    uint8_t rank, size_t* cnt, Op op
) {
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

namespace cpu { 

template<Operation operation>
struct Operator;

template<typename S, typename D, Operation OP>
bool unary::operation(const tensor_t* src, tensor_t* dst) {
    uint8_t rank = src->rank;
    std::vector<std::size_t> indexes(rank, 0); 
    unaryOp<S, D, Operator<OP>>( 
        static_cast<const S*>(src->address), src->shape, src->strides,
        static_cast<D*>(dst->address), dst->shape, dst->strides,
        rank, indexes.data(), {}
    ); 
    return true; 
}

template<typename S1, typename S2, typename D, Operation OP>
bool binary::operation(const tensor_t* src1, const tensor_t* src2, tensor_t* dst) {
    uint8_t rank = src1->rank;
    std::vector<std::size_t> indexes(rank, 0); 
    binaryOp<S1, S2, D, Operator<OP>>(
        static_cast<const S1*>(src1->address), src1->shape, src1->strides,
        static_cast<const S2*>(src2->address), src2->shape, src2->strides,
        static_cast<D*>(dst->address), dst->shape, dst->strides,
        rank, indexes.data(), {}
    ); 
    return true;
}
 
template<>
struct Operator<Operation::ADD> { 
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a + b)) {
        return a + b;
    }
};

template<>
struct Operator<Operation::SUB> { 
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a - b)) {
        return a - b;
    }
};

template<>
struct Operator<Operation::MUL> { 
    template<class A, class B>
    constexpr auto operator()(A&& a, B&& b) const noexcept(noexcept(a * b)) {
        return a * b;
    }
};

template<>
struct Operator<Operation::NEG> { 
    template<class A>
    constexpr auto operator()(A&& a) const noexcept(noexcept(-a)) {
        return -a;
    }
};
 
 
template bool unary::operation<int8_t, int8_t, Operation::NEG>(const tensor_t*, tensor_t*);
template bool unary::operation<int16_t, int16_t, Operation::NEG>(const tensor_t*, tensor_t*);
template bool unary::operation<int32_t, int32_t, Operation::NEG>(const tensor_t*, tensor_t*);
template bool unary::operation<int64_t, int64_t, Operation::NEG>(const tensor_t*, tensor_t*);
template bool unary::operation<float, float, Operation::NEG>(const tensor_t*, tensor_t*);
template bool unary::operation<double, double, Operation::NEG>(const tensor_t*, tensor_t*);

template bool binary::operation<int8_t, int8_t, int8_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int16_t, int16_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int32_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int64_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int16_t, int8_t, int16_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int16_t, int16_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int32_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int64_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int32_t, int8_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int16_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int32_t, int32_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int64_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int64_t, int8_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int16_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int32_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int64_t, int64_t, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int32_t, float, float, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<float, int32_t, float, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, double, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, int32_t, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<float, float, float, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<float, double, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, float, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, double, double, Operation::ADD>(const tensor_t*, const tensor_t*, tensor_t*);

// Binary operations (SUB)
template bool binary::operation<int8_t, int8_t, int8_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int16_t, int16_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int32_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int64_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int16_t, int8_t, int16_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int16_t, int16_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int32_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int64_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int32_t, int8_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int16_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int32_t, int32_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int64_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int64_t, int8_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int16_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int32_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int64_t, int64_t, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int32_t, float, float, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<float, int32_t, float, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, double, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, int32_t, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<float, float, float, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<float, double, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, float, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, double, double, Operation::SUB>(const tensor_t*, const tensor_t*, tensor_t*);

// Binary operations (MUL)
template bool binary::operation<int8_t, int8_t, int8_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int16_t, int16_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int32_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int8_t, int64_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int16_t, int8_t, int16_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int16_t, int16_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int32_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int16_t, int64_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int32_t, int8_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int16_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int32_t, int32_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, int64_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int64_t, int8_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int16_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int32_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int64_t, int64_t, int64_t, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<int32_t, float, float, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<float, int32_t, float, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<int32_t, double, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, int32_t, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);

template bool binary::operation<float, float, float, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<float, double, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, float, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
template bool binary::operation<double, double, double, Operation::MUL>(const tensor_t*, const tensor_t*, tensor_t*);
 
}
 