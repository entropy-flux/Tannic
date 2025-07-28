#pragma once

#include "Resources.hpp" 
#include "Tensor.hpp"
#include "runtime/tensor.h"
#include "runtime/resources.h"

using namespace tannic;

tensor_t structure(Tensor const& tensor);
host_t structure(Host const& resource);
device_t structure(Device const& resource);
allocator_t structure(Allocator const& allocator);