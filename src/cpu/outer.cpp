#include <array>
#include "cpu/outer.hpp"
#ifndef HAS_FLOAT16
    #if defined(__STDCPP_FLOAT16_T__) && __STDCPP_FLOAT16_T__
        #include <stdfloat>
        using half = std::float16_t;
        #define HAS_FLOAT16 1
    #else 
        #define HAS_FLOAT16 0 
        struct half_placeholder { float value; };
        using half = half_placeholder;
    #endif
#endif
 
namespace {  

template<typename S1, typename S2, typename D>
void vectorOuterKernel(
    const S1* src1_ptr, size_t n1,
    const S2* src2_ptr, size_t n2,
    D* dst_ptr
) { 
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            dst_ptr[i * n2 + j] = src1_ptr[i] * src2_ptr[j];
        }
    }
}

template<typename S1, typename S2, typename D>
status launchOuterKernel(const tensor_t* src1, const tensor_t* src2, tensor_t* dst) {  
    size_t n1 = src1->shape.sizes[0];
    size_t n2 = src2->shape.sizes[0];

    vectorOuterKernel<S1, S2, D>(
        (const S1*)(src1->address), n1,
        (const S2*)(src2->address), n2,
        (D*)(dst->address)
    );

    return SUCCESS;
} 

using Kernel = status(*)(const tensor_t*, const tensor_t*, tensor_t*);      

constexpr static status launchDefaultKernel(const tensor_t*, const tensor_t*, tensor_t*) {
    return UNSUPPORTED_DTYPE;
}; 

constexpr static inline auto index(type first, type second) {
    return static_cast<int>(first) + static_cast<int>(TYPES) * static_cast<int>(second);
}  

constexpr auto dispatchOuter = []() {
    std::array<Kernel, index(TYPES, TYPES)> table; 
    table.fill(launchDefaultKernel); 
 
    table[index(int8, int8)]   = launchOuterKernel<int8_t, int8_t, int32_t>;
    table[index(int8, int16)]  = launchOuterKernel<int8_t, int16_t, int32_t>;
    table[index(int8, int32)]  = launchOuterKernel<int8_t, int32_t, int32_t>;
    table[index(int8, int64)]  = launchOuterKernel<int8_t, int64_t, int64_t>;
 
    table[index(int16, int8)]   = launchOuterKernel<int16_t, int8_t, int32_t>;
    table[index(int16, int16)]  = launchOuterKernel<int16_t, int16_t, int32_t>;
    table[index(int16, int32)]  = launchOuterKernel<int16_t, int32_t, int32_t>;
    table[index(int16, int64)]  = launchOuterKernel<int16_t, int64_t, int64_t>;
 
    table[index(int32, int8)]   = launchOuterKernel<int32_t, int8_t, int32_t>;
    table[index(int32, int16)]  = launchOuterKernel<int32_t, int16_t, int32_t>;
    table[index(int32, int32)]  = launchOuterKernel<int32_t, int32_t, int64_t>;
    table[index(int32, int64)]  = launchOuterKernel<int32_t, int64_t, int64_t>;
 
    table[index(int64, int8)]   = launchOuterKernel<int64_t, int8_t, int64_t>;
    table[index(int64, int16)]  = launchOuterKernel<int64_t, int16_t, int64_t>;
    table[index(int64, int32)]  = launchOuterKernel<int64_t, int32_t, int64_t>;
    table[index(int64, int64)]  = launchOuterKernel<int64_t, int64_t, int64_t>;
 
#ifdef HAS_FLOAT16
    table[index(int32, float16)] = launchOuterKernel<int32_t, half, float>;
    table[index(float16, int32)] = launchOuterKernel<half, int32_t, float>; 

    table[index(float16, float16)] = launchOuterKernel<half, half, float>;
    table[index(float16, float32)] = launchOuterKernel<half, float, float>;
    table[index(float16, float64)] = launchOuterKernel<half, double, double>;
    table[index(float64, float16)] = launchOuterKernel<double, half, double>;
#endif

    table[index(float32, float16)] = launchOuterKernel<float, half, float>;
    table[index(float32, float32)] = launchOuterKernel<float, float, float>;
    table[index(float32, float64)] = launchOuterKernel<float, double, double>;

    table[index(int32, float32)] = launchOuterKernel<int32_t, float, float>;
    table[index(float32, int32)] = launchOuterKernel<float, int32_t, float>;
    table[index(int32, float64)] = launchOuterKernel<int32_t, double, double>;
    table[index(float64, int32)] = launchOuterKernel<double, int32_t, double>;
 
    table[index(float64, float32)] = launchOuterKernel<double, float, double>;
    table[index(float64, float64)] = launchOuterKernel<double, double, double>;

    return table;
}();


} namespace cpu {

status outer(tensor_t const* src0, tensor_t const* src1, tensor_t* dst) {
    return dispatchOuter[index(src0->dtype, src1->dtype)](src0, src1, dst);
}

}