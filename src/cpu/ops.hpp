#pragma once 
#include "runtime/types.h"
#include "runtime/tensor.h" 
#include "runtime/status.h"

using namespace tannic;

namespace cpu {  
     
status cpy( const tensor_t*, tensor_t*); 
status log( const tensor_t*, tensor_t*); 
status exp( const tensor_t*, tensor_t*); 
status sqrt( const tensor_t*, tensor_t*); 
status rsqrt( const tensor_t*, tensor_t*, float); 
status abs( const tensor_t*, tensor_t*); 
status sin( const tensor_t*, tensor_t*); 
status cos( const tensor_t*, tensor_t*); 
status tan( const tensor_t*, tensor_t*); 
status sinh( const tensor_t*, tensor_t*); 
status cosh( const tensor_t*, tensor_t*); 
status tanh( const tensor_t*, tensor_t*);

status neg(const tensor_t*, tensor_t*);
status add(const tensor_t*, const tensor_t*, tensor_t*);
status sub(const tensor_t*, const tensor_t*, tensor_t*); 
status mul(const tensor_t*, const tensor_t*, tensor_t*);    
status pow(const tensor_t*, const tensor_t*, tensor_t*);

status scale(const tensor_t*, const scalar_t*, tensor_t*);     

} // namespace cpu