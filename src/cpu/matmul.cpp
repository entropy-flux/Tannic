#include <cstddef>
#include <type_traits>
#include <vector>

#include "cpu/matmul.hpp"

namespace cpu {

template <typename TA, typename TB, typename TC>
void gemm(
    const TA* A, const TB* B, TC* C,
    size_t M, size_t N, size_t K,
    const size_t* A_strides, const size_t* B_strides, const size_t* C_strides,
    bool A_transposed, bool B_transposed,
    size_t A_rank, size_t B_rank, size_t C_rank) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            TC sum = TC(0);
            for (size_t k = 0; k < K; ++k) {
                size_t A_off = A_transposed
                    ? k * A_strides[A_rank - 2] + m * A_strides[A_rank - 1]
                    : m * A_strides[A_rank - 2] + k * A_strides[A_rank - 1];

                size_t B_off = B_transposed
                    ? n * B_strides[B_rank - 2] + k * B_strides[B_rank - 1]
                    : k * B_strides[B_rank - 2] + n * B_strides[B_rank - 1];

                sum += static_cast<TC>(A[A_off]) * static_cast<TC>(B[B_off]);
            }
            size_t C_off = m * C_strides[C_rank - 2] + n * C_strides[C_rank - 1];
            C[C_off] = sum;
        }
    }
}
 
#ifdef OPENBLAS
#include <cblas.h>
 
#endif

inline void unravel_index(size_t idx, const size_t* shape, size_t rank, size_t* out_indices) {
    for (int i = int(rank) - 1; i >= 0; --i) {
        out_indices[i] = idx % shape[i];
        idx /= shape[i];
    }
}

inline size_t compute_batch_offset(const tensor_t* t, const size_t* idx, size_t batch_rank) {
    size_t offset = 0;
    for (size_t i = 0; i < batch_rank; ++i) {
        if (t->rank > 2)
            offset += (t->shape[i] == 1 ? 0 : idx[i]) * t->strides[i];
    }
    return offset;
}

template <typename TA, typename TB, typename TC>
void matmul_op(const tensor_t* A, const tensor_t* B, tensor_t* C, bool A_transposed, bool B_transposed) {
    const TA* A_data = reinterpret_cast<const TA*>(A->address) + A->offset;
    const TB* B_data = reinterpret_cast<const TB*>(B->address) + B->offset;
    TC* C_data = reinterpret_cast<TC*>(C->address) + C->offset;

    const size_t M = C->shape[C->rank - 2];
    const size_t N = C->shape[C->rank - 1];
    const size_t K = A_transposed ? A->shape[A->rank - 2] : A->shape[A->rank - 1];

    const size_t batch_rank = (C->rank > 2) ? C->rank - 2 : 0;

    size_t batch_size = 1;
    for (size_t i = 0; i < batch_rank; ++i)
        batch_size *= C->shape[i];

    std::vector<size_t> c_batch_shape(batch_rank);
    for (size_t i = 0; i < batch_rank; ++i)
        c_batch_shape[i] = C->shape[i];

    std::vector<size_t> a_idx(batch_rank, 0);
    std::vector<size_t> b_idx(batch_rank, 0);
    std::vector<size_t> c_idx(batch_rank, 0);

    for (size_t batch_i = 0; batch_i < batch_size; ++batch_i) {
        if (batch_rank > 0)
            unravel_index(batch_i, c_batch_shape.data(), batch_rank, c_idx.data());

        for (size_t i = 0; i < batch_rank; ++i) {
            a_idx[i] = (A->rank > 2 && A->shape[i] == 1) ? 0 : c_idx[i];
            b_idx[i] = (B->rank > 2 && B->shape[i] == 1) ? 0 : c_idx[i];
        }

        size_t offset_A = compute_batch_offset(A, a_idx.data(), batch_rank);
        size_t offset_B = compute_batch_offset(B, b_idx.data(), batch_rank);
        size_t offset_C = compute_batch_offset(C, c_idx.data(), batch_rank);

        const TA* A_batch = A_data + offset_A;
        const TB* B_batch = B_data + offset_B;
        TC* C_batch = C_data + offset_C;

        gemm<TA, TB, TC>(
            A_batch, B_batch, C_batch,
            M, N, K,
            A->strides, B->strides, C->strides,
            A_transposed, B_transposed,
            A->rank, B->rank, C->rank
        );
    }
}

} // namespace cpu



// Explicit template instantiations using bool for the last two parameters
template void cpu::matmul_op<int8_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int8_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int8_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int8_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<int16_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int16_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int16_t, int32_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int16_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<int32_t, int8_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int32_t, int16_t, int32_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int32_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int32_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<int64_t, int8_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int64_t, int16_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int64_t, int32_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int64_t, int64_t, int64_t>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<float, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<float, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<double, float, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<double, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

template void cpu::matmul_op<int32_t, float, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<float, int32_t, float>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<int32_t, double, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);
template void cpu::matmul_op<double, int32_t, double>(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);