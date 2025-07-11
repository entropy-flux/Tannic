// Copyright 2025 Eric Cardozo
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

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include "core/types.h" 
#include "core/resources.h"

typedef struct scalar_t {
    storage_t storage;
    type dtype; 
} scalar_t;

typedef struct tensor_t {
    uint8_t rank;
    size_t* shape;
    size_t* strides;
    storage_t* storage;
    ptrdiff_t offset;
    type dtype; 
} tensor_t; 

#ifdef __cplusplus
}
#endif 

#endif // TENSOR_H