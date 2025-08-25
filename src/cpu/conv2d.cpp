#include <array> 
#include <cstdlib>
#include "cpu/conv.hpp"

namespace {

/* 
template<typename S, typename Wt, typename D>
void conv2dKernel(
    const S* x_ptr, const Wt* w_ptr, D* y_ptr,
    size_t N, size_t C_in, size_t H_in, size_t W_in,
    size_t C_out, size_t K_h, size_t K_w,
    size_t H_out, size_t W_out,
    const size_t pad[2], const size_t stride[2],
    const strides_t& xs, const strides_t& ws, const strides_t& ys
) {
    auto offs4 = [](const strides_t& s, size_t n, size_t c, size_t h, size_t w) {
        return n * s.sizes[0] + c * s.sizes[1] + h * s.sizes[2] + w * s.sizes[3];
    };

    size_t pad_h = pad[0], pad_w = pad[1];
    size_t stride_h = stride[0], stride_w = stride[1];

    for (size_t n = 0; n < N; ++n) {
        for (size_t co = 0; co < C_out; ++co) {
            for (size_t oh = 0; oh < H_out; ++oh) {
                for (size_t ow = 0; ow < W_out; ++ow) {
                    D acc = D(0);

                    ptrdiff_t base_h = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_h);
                    ptrdiff_t base_w = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_w);

                    for (size_t ci = 0; ci < C_in; ++ci) {
                        for (size_t kh = 0; kh < K_h; ++kh) {
                            ptrdiff_t ih = base_h + static_cast<ptrdiff_t>(kh);
                            if (ih < 0 || ih >= static_cast<ptrdiff_t>(H_in)) continue;

                            for (size_t kw = 0; kw < K_w; ++kw) {
                                ptrdiff_t iw = base_w + static_cast<ptrdiff_t>(kw);
                                if (iw < 0 || iw >= static_cast<ptrdiff_t>(W_in)) continue;

                                size_t x_off = offs4(xs, n, ci, ih, iw);
                                size_t w_off = offs4(ws, co, ci, kh, kw);
                                acc += static_cast<D>(x_ptr[x_off]) * static_cast<D>(w_ptr[w_off]);
                            }
                        }
                    }

                    size_t y_off = offs4(ys, n, co, oh, ow);
                    y_ptr[y_off] = acc;
                }
            }
        }
    }
}

template<typename S, typename Wt, typename D>
status launchConv2DKernel(
    const tensor_t* x, const tensor_t* w, tensor_t* y,
    const size_t pad[2], const size_t stride[2]
) {
    if (x->rank != 4 || w->rank != 4 || y->rank != 4) {
        return ERROR;
    }

    size_t N     = x->shape.sizes[0];
    size_t C_in  = x->shape.sizes[1];
    size_t H_in  = x->shape.sizes[2];
    size_t W_in  = x->shape.sizes[3];

    size_t C_out = w->shape.sizes[0];
    size_t K_in  = w->shape.sizes[1];
    size_t K_h   = w->shape.sizes[2];
    size_t K_w   = w->shape.sizes[3];

    if (C_in != K_in) return ERROR;

    size_t H_out = (H_in + 2 * pad[0] - K_h) / stride[0] + 1;
    size_t W_out = (W_in + 2 * pad[1] - K_w) / stride[1] + 1;

    if (y->shape.sizes[0] != N || y->shape.sizes[1] != C_out ||
        y->shape.sizes[2] != H_out || y->shape.sizes[3] != W_out) {
            return ERROR;
    }

    conv2dKernel<S,Wt,D>(
        static_cast<const S*>(x->address),
        static_cast<const Wt*>(w->address),
        static_cast<D*>(y->address),
        N, C_in, H_in, W_in, C_out, K_h, K_w, H_out, W_out,
        pad, stride,
        x->strides, w->strides, y->strides
    );

    return SUCCESS;
}
*/  

template<typename S0, typename S1, typename D>
void gemmKernel( 
    bool A_trans, bool B_trans,
    const S0* A_ptr,
    const S1* B_ptr,
    D* C_ptr,
    size_t M, size_t N, size_t K,
    size_t A_ld, size_t B_ld, size_t C_ld,
    D alpha
) {   
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            D sum = D(0);
            for (int k = 0; k < K; ++k) {
                size_t A_idx = A_trans ? k * A_ld + i : i * A_ld + k;
                size_t B_idx = B_trans ? j * B_ld + k : k * B_ld + j;
                sum += static_cast<D>(A_ptr[A_idx]) * static_cast<D>(B_ptr[B_idx]);
            }
            C_ptr[i * C_ld + j] = alpha *sum;
        }
    }
}

#ifdef BLAS
#include <cblas.h>  

template <>
void gemmKernel<float, float, float>(
    bool A_trans, bool B_trans,
    const float* A_ptr,
    const float* B_ptr,
    float* C_ptr,
    size_t M, size_t N, size_t K,
    size_t A_ld, size_t B_ld, size_t C_ld,
    float alpha
) {   
    cblas_sgemm(
        CblasRowMajor, 
        A_trans ? CblasTrans : CblasNoTrans, 
        B_trans ? CblasTrans : CblasNoTrans,
        M, N, K,
        alpha, A_ptr, A_ld, B_ptr, B_ld,
        0.0, C_ptr, C_ld 
    );  
}


template <>
void gemmKernel<double, double, double>(
    bool A_trans, bool B_trans,
    const double* A_ptr,
    const double* B_ptr,
    double* C_ptr,
    size_t M, size_t N, size_t K,
    size_t A_ld, size_t B_ld, size_t C_ld, 
    double alpha
) {   
    cblas_dgemm(
        CblasRowMajor, 
        A_trans ? CblasTrans : CblasNoTrans, 
        B_trans ? CblasTrans : CblasNoTrans,
        M, N, K,
        alpha, A_ptr, A_ld, B_ptr, B_ld,
        0.0, C_ptr, C_ld
    );
}

#endif  

template<typename S, typename Wt, typename D>
status launchConv2DKernel(
    const tensor_t* x, const tensor_t* w, tensor_t* y,
    const size_t pad[2], const size_t stride[2]
) {
    if (x->rank != 4 || w->rank != 4 || y->rank != 4) {
        return ERROR;
    }

    size_t batch_size = x->shape.sizes[0];  // Renamed from N to avoid conflict
    size_t C_in  = x->shape.sizes[1];
    size_t H_in  = x->shape.sizes[2];
    size_t W_in  = x->shape.sizes[3];

    size_t C_out = w->shape.sizes[0];
    size_t K_in  = w->shape.sizes[1];
    size_t K_h   = w->shape.sizes[2];
    size_t K_w   = w->shape.sizes[3];

    if (C_in != K_in) return ERROR;

    size_t H_out = (H_in + 2 * pad[0] - K_h) / stride[0] + 1;
    size_t W_out = (W_in + 2 * pad[1] - K_w) / stride[1] + 1;

    if (y->shape.sizes[0] != batch_size || y->shape.sizes[1] != C_out ||
        y->shape.sizes[2] != H_out || y->shape.sizes[3] != W_out) {
        return ERROR;
    }

    // Calculate im2col buffer size
    size_t col_height = K_h * K_w * C_in;
    size_t col_width = H_out * W_out;
    size_t total_col_size = batch_size * col_height * col_width;
    
    // Allocate single buffer for im2col data
    D* col_buffer = (D*)malloc(total_col_size * sizeof(D));
    if (!col_buffer) return ERROR;

    // Perform im2col transformation
    size_t pad_h = pad[0], pad_w = pad[1];
    size_t stride_h = stride[0], stride_w = stride[1];
    
    D* col_ptr = col_buffer;
    
    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t oh = 0; oh < H_out; ++oh) {
            for (size_t ow = 0; ow < W_out; ++ow) {
                ptrdiff_t base_h = static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_h);
                ptrdiff_t base_w = static_cast<ptrdiff_t>(ow * stride_w) - static_cast<ptrdiff_t>(pad_w);
                
                for (size_t kh = 0; kh < K_h; ++kh) {
                    for (size_t kw = 0; kw < K_w; ++kw) {
                        for (size_t ci = 0; ci < C_in; ++ci) {
                            ptrdiff_t ih = base_h + static_cast<ptrdiff_t>(kh);
                            ptrdiff_t iw = base_w + static_cast<ptrdiff_t>(kw);
                            
                            if (ih >= 0 && ih < static_cast<ptrdiff_t>(H_in) && 
                                iw >= 0 && iw < static_cast<ptrdiff_t>(W_in)) {
                                size_t x_off = n * x->strides.sizes[0] + 
                                              ci * x->strides.sizes[1] + 
                                              ih * x->strides.sizes[2] + 
                                              iw * x->strides.sizes[3];
                                *col_ptr++ = static_cast<D>(static_cast<const S*>(x->address)[x_off]);
                            } else {
                                *col_ptr++ = D(0);
                            }
                        }
                    }
                }
            }
        }
    }

    size_t weight_cols = K_h * K_w * C_in;
    
    size_t gemm_M = H_out * W_out;
    size_t gemm_N = C_out;
    size_t gemm_K = weight_cols;
    
    for (size_t n = 0; n < batch_size; ++n) {
        D* batch_col_buffer = col_buffer + n * col_height * col_width;
        D* batch_y_ptr = static_cast<D*>(y->address) + n * C_out * H_out * W_out;
        
        gemmKernel<D, Wt, D>(
            false,  
            true,  
            batch_col_buffer,                  
            static_cast<const Wt*>(w->address),
            batch_y_ptr,                    
            gemm_M, gemm_N, gemm_K,
            gemm_K,    
            gemm_K,    
            gemm_N,  
            D(1)   
        );
    }

    // Free the im2col buffer
    free(col_buffer);
    
    return SUCCESS;
}


constexpr static inline int index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}

constexpr static status launchDefaultConv2DKernel(
    const tensor_t*, const tensor_t*, tensor_t*, 
    const size_t[2], const size_t[2]
) {
    return UNSUPPORTED_DTYPE;
}

using Conv2DKernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*, const size_t[2], const size_t[2]);

constexpr auto dispatchConv2D = []() {
    std::array<Conv2DKernel, index(TYPES, TYPES)> table;
    table.fill(launchDefaultConv2DKernel);
    table[index(int8, int8)] = launchConv2DKernel<int8_t, int8_t, int8_t>;
    table[index(int16, int16)] = launchConv2DKernel<int16_t, int16_t, int16_t>;
    table[index(int32, int32)] = launchConv2DKernel<int32_t, int32_t, int32_t>;
    table[index(int64, int64)] = launchConv2DKernel<int64_t, int64_t, int64_t>;
    table[index(float32, float32)] = launchConv2DKernel<float, float, float>;
    table[index(float64, float64)] = launchConv2DKernel<double, double, double>;
    return table;
}();

} namespace cpu {

status conv2d(const tensor_t* signal, const tensor_t* kernel, tensor_t* dst, const size_t pad[2], const size_t stride[2]) {
    return dispatchConv2D[index(signal->dtype, kernel->dtype)](signal, kernel, dst, pad, stride);
}

}