#include "types.hpp"
#include "limits.hpp"
#include "tensor.hpp"
#include <cstring> 
#include <limits>

namespace tannic::expression {

Tensor Zero::forward(Context const& context) const {
    Tensor result(dtype_, shape_);
    result.initialize();

    switch (dtype_) {
    case int8: {
        std::memset(result.bytes(), 0, result.nbytes());
        break;
    }
    case int16: {
        auto* ptr = reinterpret_cast<int16_t*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 0);
        break;
    }
    case int32: {
        auto* ptr = reinterpret_cast<int32_t*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 0);
        break;
    }
    case int64: {
        auto* ptr = reinterpret_cast<int64_t*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 0);
        break;
    }
    case float16: {
        auto* ptr = reinterpret_cast<float16_t*>(result.bytes());
        float16_t zero(0.0f);
        std::fill(ptr, ptr + result.nelements(), zero);
        break;
    }
    case bfloat16: {
        auto* ptr = reinterpret_cast<bfloat16_t*>(result.bytes());
        bfloat16_t zero(0.0f);
        std::fill(ptr, ptr + result.nelements(), zero);
        break;
    }
    case float32: {
        auto* ptr = reinterpret_cast<float*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 0.0f);
        break;
    }
    case float64: {
        auto* ptr = reinterpret_cast<double*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 0.0);
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype in Zero::forward");
    }

    return result;
}

Tensor One::forward(Context const& context) const {
    Tensor result(dtype_, shape_);
    result.initialize();

    switch (dtype_) {
    case int8: {
        auto* ptr = reinterpret_cast<int8_t*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 1);
        break;
    }
    case int16: {
        auto* ptr = reinterpret_cast<int16_t*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 1);
        break;
    }
    case int32: {
        auto* ptr = reinterpret_cast<int32_t*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 1);
        break;
    }
    case int64: {
        auto* ptr = reinterpret_cast<int64_t*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 1);
        break;
    }
    case float16: {
        auto* ptr = reinterpret_cast<float16_t*>(result.bytes());
        float16_t one(1.0f);
        std::fill(ptr, ptr + result.nelements(), one);
        break;
    }
    case bfloat16: {
        auto* ptr = reinterpret_cast<bfloat16_t*>(result.bytes());
        bfloat16_t one(1.0f);
        std::fill(ptr, ptr + result.nelements(), one);
        break;
    }
    case float32: {
        auto* ptr = reinterpret_cast<float*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 1.0f);
        break;
    }
    case float64: {
        auto* ptr = reinterpret_cast<double*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), 1.0);
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype in One::forward");
    }

    return result;
}

template<>
Tensor Infinity<Sign::Positive>::forward(Context const& context) const {
    Tensor result(dtype_, shape_);
    result.initialize();

    switch (dtype_) { 

        case float16: {
            auto* ptr = reinterpret_cast<float16_t*>(result.bytes());
            float16_t inf(std::numeric_limits<float>::infinity());
            std::fill(ptr, ptr + result.nelements(), inf);
            break;
        }
        case bfloat16: {
            auto* ptr = reinterpret_cast<bfloat16_t*>(result.bytes());
            bfloat16_t inf(std::numeric_limits<float>::infinity());
            std::fill(ptr, ptr + result.nelements(), inf);
            break;
        }
        case float32: {
            auto* ptr = reinterpret_cast<float*>(result.bytes());
            std::fill(ptr, ptr + result.nelements(), std::numeric_limits<float>::infinity());
            break;
        }
        case float64: {
            auto* ptr = reinterpret_cast<double*>(result.bytes());
            std::fill(ptr, ptr + result.nelements(), std::numeric_limits<double>::infinity());
            break;
        }
        default:
            throw Exception("Unsupported dtype in Infinity<Positive>::forward");
    }

    return result;
} 

template<>
Tensor Infinity<Sign::Negative>::forward(Context const& context) const {
    Tensor result(dtype_, shape_);
    result.initialize();

    switch (dtype_) { 

    case float16: {
        auto* ptr = reinterpret_cast<float16_t*>(result.bytes());
        float16_t inf(-std::numeric_limits<float>::infinity());
        std::fill(ptr, ptr + result.nelements(), inf);
        break;
    }
    case bfloat16: {
        auto* ptr = reinterpret_cast<bfloat16_t*>(result.bytes());
        bfloat16_t inf(-std::numeric_limits<float>::infinity());
        std::fill(ptr, ptr + result.nelements(), inf);
        break;
    }
    case float32: {
        auto* ptr = reinterpret_cast<float*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), -std::numeric_limits<float>::infinity());
        break;
    }
    case float64: {
        auto* ptr = reinterpret_cast<double*>(result.bytes());
        std::fill(ptr, ptr + result.nelements(), -std::numeric_limits<double>::infinity());
        break;
    }
    default:
        throw Exception("Unsupported dtype in Infinity<Negative>::forward");
    }

    return result;
}

template class Infinity<Sign::Positive>;
template class Infinity<Sign::Negative>;

} // namespace tannic::expression

 