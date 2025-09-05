// Copyright 2025 Eric Hermosis
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

#ifndef TYPES_H
#define TYPES_H 

#ifdef __cplusplus
#include <cstdint> 
namespace tannic {
extern "C" {
#else
#include <stdint.h> 
#endif 

enum type {  
    any,
    boolean, 
    int8,
    int16,
    int32,
    int64,
    float16,
    bfloat16,
    float32,
    float64,
    complex64,   
    complex128,  
    unknown,
    TYPES
};
 
#ifdef __cplusplus
}
} // namespace tannic
#endif
 
#endif // TYPES_H