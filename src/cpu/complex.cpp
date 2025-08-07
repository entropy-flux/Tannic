#include <stdexcept>
#include <vector>
#include <array>
#include <cmath>  
#include "cpu/complex.hpp"

namespace {

template<typename S0, typename S1, typename C>
void scalarViewAsCartesianKernel(
    const S0* re_ptr, const S1* im_ptr, C* dst_ptr
) {
    dst_ptr[0] = *re_ptr;
    dst_ptr[1] = *im_ptr;
}      

template<typename Rho, typename Theta, typename D>
void scalarViewAsPolarKernel(
    const Rho* rho_ptr, const Theta* theta_ptr, D* dst_ptr
) {
    Rho rho = *rho_ptr;
    Theta theta = *theta_ptr;

    dst_ptr[0] = rho * cos(theta);
    dst_ptr[1] = rho * sin(theta);
}

template<typename S0, typename S1, typename C>
void stridedViewAsCartesianKernel(
    const S0* re_ptr, const shape_t& re_shape, const strides_t& re_ne,
    const S1* im_ptr, const shape_t& im_shape, const strides_t& im_ne,
    C* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank
) {  
    size_t cnt[8] = {0}; 
    for (size_t idx = 0;; ++idx) {
        size_t re_offs = 0, im_offs = 0;

        for (uint8_t i = 0; i < rank; ++i) { 
            size_t idx_re = (re_shape.sizes[i] == 1) ? 0 : cnt[i];
            size_t idx_im = (im_shape.sizes[i] == 1) ? 0 : cnt[i];
            
            re_offs += idx_re * re_ne.sizes[i];
            im_offs += idx_im * im_ne.sizes[i];
        }

        dst_ptr[2 * idx]     = re_ptr[re_offs];
        dst_ptr[2 * idx + 1] = im_ptr[im_offs];

        bool done = false;
        for (int i = rank - 1; i >= 0; --i) {
            if (++cnt[i] < dst_shape.sizes[i])
                break;
            if (i == 0)
                done = true;
            cnt[i] = 0;
        }

        if (done) break;
    }
}  

template<typename Rho, typename Theta, typename C>
void stridedViewAsPolarKernel(
    const Rho* rho_ptr, const shape_t& rho_shape, const strides_t& rho_ne,
    const Theta* theta_ptr, const shape_t& theta_shape, const strides_t& theta_ne,
    C* dst_ptr, const shape_t& dst_shape, const strides_t& dst_strides,
    uint8_t rank
) {
    size_t cnt[8] = {0};
    for (size_t idx = 0;; ++idx) {
        size_t rho_offs = 0, theta_offs = 0;

        for (uint8_t i = 0; i < rank; ++i) {
            size_t idx_rho = (rho_shape.sizes[i] == 1) ? 0 : cnt[i];
            size_t idx_theta = (theta_shape.sizes[i] == 1) ? 0 : cnt[i];

            rho_offs += idx_rho * rho_ne.sizes[i];
            theta_offs += idx_theta * theta_ne.sizes[i];
        }

        Rho rho = rho_ptr[rho_offs];
        Theta theta = theta_ptr[theta_offs];

        dst_ptr[2 * idx]     = rho * std::cos(theta);  // real part
        dst_ptr[2 * idx + 1] = rho * std::sin(theta);  // imag part

        bool done = false;
        for (int i = rank - 1; i >= 0; --i) {
            if (++cnt[i] < dst_shape.sizes[i])
                break;
            if (i == 0)
                done = true;
            cnt[i] = 0;
        }

        if (done) break;
    }
}
 
template<typename S0, typename S1, typename C>
status launchViewAsCartesianKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    if (dst->rank == 0) {
        scalarViewAsCartesianKernel<S0, S1, C>(
            (const S0*)(src0->address), 
            (const S1*)(src1->address),  
            (C*)(dst->address)
        ); 
    } 
    
    else {     
        stridedViewAsCartesianKernel<S0, S1, C>(
            (const S0*)(src0->address), src0->shape, src0->strides,
            (const S1*)(src1->address), src1->shape, src1->strides,
            (C*)(dst->address), dst->shape, dst->strides,
            dst->rank
        ); 
    } 
    return SUCCESS;
}       

template<typename Rho, typename Theta, typename C>
status launchViewAsPolarKernel(const tensor_t* src0, const tensor_t* src1, tensor_t* dst) {
    if (dst->rank == 0) {
        scalarViewAsPolarKernel<Rho, Theta, C>(
            (const Rho*)(src0->address), 
            (const Theta*)(src1->address),  
            (C*)(dst->address)
        ); 
    } 
    else {     
        stridedViewAsPolarKernel<Rho, Theta, C>(
            (const Rho*)(src0->address), src0->shape, src0->strides,
            (const Theta*)(src1->address), src1->shape, src1->strides,
            (C*)(dst->address), dst->shape, dst->strides,
            dst->rank
        ); 
    } 
    return SUCCESS;
}

using Kernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*);      

constexpr static status launchDefaultKernel(const tensor_t*, const tensor_t*, tensor_t*) {
    return UNSUPPORTED_DTYPE;
}; 

constexpr static inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}  

constexpr auto dispatchViewAsCartesian = []() {
    std::array<Kernel, index(TYPES, TYPES)> table; table.fill(launchDefaultKernel); 
    table[index(int8, int8)]     = launchViewAsCartesianKernel<int8_t, int8_t,  float>;
    table[index(int8, int16)]    = launchViewAsCartesianKernel<int8_t, int16_t, float>;
    table[index(int8, int32)]    = launchViewAsCartesianKernel<int8_t, int32_t, float>;
    table[index(int8, int64)]    = launchViewAsCartesianKernel<int8_t, int64_t, float>;

    table[index(int16, int8)]    = launchViewAsCartesianKernel<int16_t, int8_t,  float>;
    table[index(int16, int16)]   = launchViewAsCartesianKernel<int16_t, int16_t, float>;
    table[index(int16, int32)]   = launchViewAsCartesianKernel<int16_t, int32_t, float>;
    table[index(int16, int64)]   = launchViewAsCartesianKernel<int16_t, int64_t, float>;

    table[index(int32, int8)]    = launchViewAsCartesianKernel<int32_t, int8_t,  float>;
    table[index(int32, int16)]   = launchViewAsCartesianKernel<int32_t, int16_t, float>;
    table[index(int32, int32)]   = launchViewAsCartesianKernel<int32_t, int32_t, float>;
    table[index(int32, int64)]   = launchViewAsCartesianKernel<int32_t, int64_t, float>;

    table[index(int64, int8)]    = launchViewAsCartesianKernel<int64_t, int8_t,  float>;
    table[index(int64, int16)]   = launchViewAsCartesianKernel<int64_t, int16_t, float>;
    table[index(int64, int32)]   = launchViewAsCartesianKernel<int64_t, int32_t, float>;
    table[index(int64, int64)]   = launchViewAsCartesianKernel<int64_t, int64_t, float>;

    table[index(int32, float32)] = launchViewAsCartesianKernel<int32_t, float,   float>;
    table[index(float32, int32)] = launchViewAsCartesianKernel<float, int32_t,   float>;
    table[index(int32, float64)] = launchViewAsCartesianKernel<int32_t, double,  double>;
    table[index(float64, int32)] = launchViewAsCartesianKernel<double, int32_t,  double>;

    table[index(float32, float32)] = launchViewAsCartesianKernel<float,  float,  float>;
    table[index(float32, float64)] = launchViewAsCartesianKernel<float,  double, double>;
    table[index(float64, float32)] = launchViewAsCartesianKernel<double, float,  double>;
    table[index(float64, float64)] = launchViewAsCartesianKernel<double, double, double>;
    return table;
}(); 

constexpr auto dispatchViewAsPolar = []() {
    std::array<Kernel, index(TYPES, TYPES)> table; 
    table.fill(launchDefaultKernel); 
    table[index(int8, int8)]     = launchViewAsPolarKernel<int8_t, int8_t,  float>;
    table[index(int8, int16)]    = launchViewAsPolarKernel<int8_t, int16_t, float>;
    table[index(int8, int32)]    = launchViewAsPolarKernel<int8_t, int32_t, float>;
    table[index(int8, int64)]    = launchViewAsPolarKernel<int8_t, int64_t, float>;

    table[index(int16, int8)]    = launchViewAsPolarKernel<int16_t, int8_t,  float>;
    table[index(int16, int16)]   = launchViewAsPolarKernel<int16_t, int16_t, float>;
    table[index(int16, int32)]   = launchViewAsPolarKernel<int16_t, int32_t, float>;
    table[index(int16, int64)]   = launchViewAsPolarKernel<int16_t, int64_t, float>;

    table[index(int32, int8)]    = launchViewAsPolarKernel<int32_t, int8_t,  float>;
    table[index(int32, int16)]   = launchViewAsPolarKernel<int32_t, int16_t, float>;
    table[index(int32, int32)]   = launchViewAsPolarKernel<int32_t, int32_t, float>;
    table[index(int32, int64)]   = launchViewAsPolarKernel<int32_t, int64_t, float>;

    table[index(int64, int8)]    = launchViewAsPolarKernel<int64_t, int8_t,  float>;
    table[index(int64, int16)]   = launchViewAsPolarKernel<int64_t, int16_t, float>;
    table[index(int64, int32)]   = launchViewAsPolarKernel<int64_t, int32_t, float>;
    table[index(int64, int64)]   = launchViewAsPolarKernel<int64_t, int64_t, float>;

    table[index(int32, float32)] = launchViewAsPolarKernel<int32_t, float,   float>;
    table[index(float32, int32)] = launchViewAsPolarKernel<float, int32_t,   float>;
    table[index(int32, float64)] = launchViewAsPolarKernel<int32_t, double,  double>;
    table[index(float64, int32)] = launchViewAsPolarKernel<double, int32_t,  double>;

    table[index(float32, float32)] = launchViewAsPolarKernel<float,  float,  float>;
    table[index(float32, float64)] = launchViewAsPolarKernel<float,  double, double>;
    table[index(float64, float32)] = launchViewAsPolarKernel<double, float,  double>;
    table[index(float64, float64)] = launchViewAsPolarKernel<double, double, double>; 
    return table;
}(); 

} namespace cpu {

status view_as_cartesian(const tensor_t* re, const tensor_t* im, tensor_t* c) {
    return dispatchViewAsCartesian[index(re->dtype, im->dtype)](re, im, c);
}

status view_as_polar(const tensor_t* rho, const tensor_t* theta, tensor_t* c) {
    return dispatchViewAsPolar[index(rho->dtype, theta->dtype)](rho, theta, c);
}

}