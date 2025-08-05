// Copyright 2025 Eric Cardozo
//
// This file is part of the Tannic Tensor Library.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#ifndef TENSOR_H
#define TENSOR_H 
 
#include "types.h"   

#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
#include "runtime/resources.h"

namespace tannic {
extern "C" {

#else
#include <stdint.h>
#include <stddef.h>
#include "runtime/resources.h"
#endif 

#define DIMENSIONS 8

struct shape_t {
    union {
        const uint32_t* address;
        uint32_t sizes[DIMENSIONS];
    };
};

struct strides_t {
    union {
        const uint64_t* address;
        uint64_t sizes[DIMENSIONS];
    };
};

struct tensor_t {
    void* address;
    uint8_t rank;
    struct shape_t shape;
    struct strides_t strides;  
    enum type dtype; 
    struct allocator_t allocator;
};    

#ifdef __cplusplus
}
} // namespace tannic
#endif 

#endif // TENSOR_H