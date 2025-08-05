#pragma once

#include "Resources.hpp" 
#include "Tensor.hpp"
#include "runtime/tensor.h"
#include "runtime/resources.h"
#include "runtime/streams.h"
#include "runtime/status.h"

using namespace tannic;

tensor_t structure(Tensor const&);
host_t structure(Host const&);
device_t structure(Device const&);
allocator_t structure(Allocator const&);  
stream_t pop_stream(const device_t*);
void put_stream(const device_t*, stream_t);  
status resolve(const allocator_t*, const allocator_t*, allocator_t*);
