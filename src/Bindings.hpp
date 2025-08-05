#pragma once

#include "Resources.hpp" 
#include "Tensor.hpp"
#include "runtime/tensor.h"
#include "runtime/resources.h"
#include "runtime/streams.h"

using namespace tannic;

tensor_t structure(Tensor const& tensor);
host_t structure(Host const& resource);
device_t structure(Device const& resource);
allocator_t structure(Allocator const& allocator);

stream_t pop_stream(const device_t*);
void put_stream(const device_t*, stream_t);