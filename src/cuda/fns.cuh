#pragma once
#include "runtime/tensor.h"
#include "runtime/resources.h"

using namespace tannic; 

namespace cuda {
 
void log(device_t  const*, tensor_t const*, tensor_t*);
void exp(device_t  const*, tensor_t const*, tensor_t*);
void sqrt(device_t const*, tensor_t const*, tensor_t*);
void abs(device_t  const*, tensor_t const*, tensor_t*);
void sin(device_t  const*, tensor_t const*, tensor_t*);
void cos(device_t  const*, tensor_t const*, tensor_t*);
void tan(device_t  const*, tensor_t const*, tensor_t*);
void sinh(device_t const*, tensor_t const*, tensor_t*);
void cosh(device_t const*, tensor_t const*, tensor_t*);
void tanh(device_t const*, tensor_t const*, tensor_t*);

}