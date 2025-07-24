#include "Functions.hpp"
#include "runtime/tensor.h"
#include "cpu/funcs.hpp"  

namespace tannic { 

static inline tensor_t c_tensor_t(Tensor const& tensor) {
    return tensor_t{
        .rank = tensor.rank(),
        .address = static_cast<void*>(tensor.bytes()),
        .shape = tensor.shape().address(),
        .strides = tensor.strides().address(), 
        .dtype = tensor.dtype()
    };
}

void expression::Log::operator()(Tensor const& input, Tensor& output) const {  
    output.initialize();
    tensor_t src = c_tensor_t(input);
    tensor_t dst = c_tensor_t(output);
    cpu::log(&src, &dst);
}

void expression::Exp::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::exp[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Exp operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void expression::Sqrt::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::sqrt[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Sqrt operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void expression::Abs::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::abs[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Abs operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void expression::Sin::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::sin[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Sin operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void expression::Cos::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::cos[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Cos operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void expression::Tan::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::tan[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Tan operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void expression::Sinh::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::sinh[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Sinh operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void expression::Cosh::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::cosh[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Cosh operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}

void expression::Tanh::operator()(Tensor const& input, Tensor& output) const {
    output.initialize();
    tensor_t source = c_tensor_t(input);
    tensor_t target = c_tensor_t(output);
    bool status = cpu::tanh[cpu::index(input.dtype())](&source, &target);
    if(!status) {
        throw std::runtime_error(
            "Tanh operation failed for dtype: " + dnameof(input.dtype())
        );
    }
}
 
} // namespace tannic