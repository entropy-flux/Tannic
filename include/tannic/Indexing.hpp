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

#ifndef INDEXING_HPP
#define INDEXING_HPP

#include <cassert>
 
#include "Concepts.hpp"

namespace tannic::indexing {
  
struct Range {
    int start;
    int stop;
};

template<Integral Index, Integral Size>
constexpr inline Index normalize(Index index, Size bound) {
    if (index < 0) index += bound;
    assert(index >= 0 && index < bound && "Index out of bounds");
    return index;
}  

template<Integral Size>
constexpr inline Range normalize(Range range, Size size) {
    int start = range.start < 0 ? size + range.start : range.start;
    int stop = range.stop < 0 ? size + range.stop + 1 : range.stop;
    return {start, stop};
}  
 
} // namespace TANNIC::indexing

#endif // INDEXING_H