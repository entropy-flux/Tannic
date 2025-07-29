#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 

using namespace tannic;

namespace cpu { 
 
void log(tensor_t  const*, tensor_t*); 
void exp(tensor_t  const*, tensor_t*); 
void sqrt(tensor_t const*, tensor_t*); 
void abs(tensor_t  const*, tensor_t*); 
void sin(tensor_t  const*, tensor_t*); 
void cos(tensor_t  const*, tensor_t*); 
void tan(tensor_t  const*, tensor_t*); 
void sinh(tensor_t const*, tensor_t*); 
void cosh(tensor_t const*, tensor_t*); 
void tanh(tensor_t const*, tensor_t*);   

} // namespace cpu