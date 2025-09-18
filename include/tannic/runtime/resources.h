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

#ifndef RESOURCES_H
#define RESOURCES_H 

#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
#include "runtime/status.h"

namespace tannic {
extern "C" {

#else
#include <stdint.h>
#include <stddef.h>
#include "runtime/status.h"
#endif 

enum environment {
    HOST,
    DEVICE
};

enum host {
    PAGEABLE = 1 << 0, 
    PINNED   = 1 << 1  
}; 
 
struct host_t {
    enum host traits;
};

struct device_t {
    int id; 
}; 

struct environment_t { 
    enum environment kind;
    union {
        struct host_t host;
        struct device_t device;
    } resource;
};
  
status resolve_two_environment(const environment_t*, const environment_t*, environment_t*); 
status resolve_three_environment(const environment_t* a, const environment_t* b, const environment_t* c, environment_t* result_out);

#ifdef __cplusplus
}
} //namesmpace tannic
#endif 

#endif // RESOURCES_H