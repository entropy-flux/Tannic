#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 

using namespace tannic;

namespace cpu { 
 
void log(tensor_t const*, tensor_t*); 
void exp(tensor_t const*, tensor_t*); 
void sqrt(tensor_t const*, tensor_t*); 
void abs(tensor_t const*, tensor_t*); 
void sin(tensor_t const*, tensor_t*); 
void cos(tensor_t const*, tensor_t*); 
void tan(tensor_t const*, tensor_t*); 
void sinh(tensor_t const*, tensor_t*); 
void cosh(tensor_t const*, tensor_t*); 
void tanh(tensor_t const*, tensor_t*);  

void neg(tensor_t const*, tensor_t*);
void add(tensor_t const*, tensor_t const*, tensor_t*);
void sub(tensor_t const*, tensor_t const*, tensor_t*); 
void mul(tensor_t const*, tensor_t const*, tensor_t*);  

void gemm(const tensor_t*, const tensor_t*, tensor_t*, bool, bool);

void argmax(tensor_t const*, tensor_t*, uint8_t axis);
void argmin(tensor_t const*, tensor_t*, uint8_t axis);

} // namespace cpu