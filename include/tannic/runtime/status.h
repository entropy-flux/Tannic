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

#ifndef STATUS_H
#define STATUS_H

#ifdef __cplusplus
namespace tannic {
extern "C" {
#endif 

enum status { 
    SUCCESS, 
    ERROR,
    UNSUPORTED_DTYPE,
    NULL_ALLOCATOR,
    INCOMPATIBLE_DEVICES,  
};

#ifdef __cplusplus
}
} // namespace tannic
#endif
 
#endif // STATUS_H