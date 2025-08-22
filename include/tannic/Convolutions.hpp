// Copyright 2025 Eric Hermosis
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

#ifndef CONVOLUTIONS_HPP
#define CONVOLUTIONS_HPP 

/**
 * @file Convolutions.hpp
 * @author Eric Hermosis
 * @date 2025
 * @brief Defines tensor convolutional operations.
 *
 * This header provides tensor convolution operations implemented as expression templates. 
 *
 * Part of the Tannic Tensor Library.
 */

#include <tuple>
#include <array>
#include <vector>
#include <cassert>

#include "Concepts.hpp"
#include "Types.hpp"
#include "Traits.hpp"
#include "Shape.hpp" 
#include "Tensor.hpp" 
#include "Exceptions.hpp"
#include "Transformations.hpp"

namespace tannic {

class Tensor;  

namespace transformation {
 

template<auto Dimension>
class Convolution;

template<>
class Convolution<1> {
public: 
    std::array<std::size_t, 1> strides;
    std::array<std::size_t, 1> padding;

    constexpr type promote(type signal, type kernel) const {
        if (signal != kernel)
            throw Exception("Dtypes must match in convolutions.");
        return signal;
    }

    constexpr Shape transform(Shape const& signal, Shape const& kernel) const {  
        if (signal.rank() != 3 || kernel.rank() != 3)
            throw Exception("Only rank 3 tensors supported for Conv1D."); 

        std::size_t N     = signal[0];     
        std::size_t C_in  = signal[1];     
        std::size_t L_in  = signal[2];     

        std::size_t C_out = kernel[0];      
        std::size_t K_in  = kernel[1];       
        std::size_t K_len = kernel[2];      

        if (C_in != K_in)
            throw Exception("Input channels must match kernel channels.");

        std::size_t L_out = (L_in + 2 * padding[0] - K_len) / strides[0] + 1;
        return Shape{N, C_out, L_out};
    }

    void forward(Tensor const&, Tensor const&, Tensor&) const;
};
 
template<>
class Convolution<2> {
public:      
    std::array<std::size_t, 2> strides;
    std::array<std::size_t, 2> padding; 
 
    constexpr type promote(type signal, type kernel) const {
        if (signal != kernel)
            throw Exception("Dtypes must match in convolutions.");
        return signal;
    }

    constexpr Shape transform(Shape const& signal, Shape const& kernel) const {   
        if (signal.rank() != 4 | kernel.rank() != 4) 
            throw Exception("Only rank 4 tensors supported."); 
            std::size_t N     = signal[0];           
            std::size_t C_in  = signal[1];         
            std::size_t H_in  = signal[2];         
            std::size_t W_in  = signal[3];          

            std::size_t C_out = kernel[0];          
            std::size_t K_in  = kernel[1];        
            std::size_t K_h   = kernel[2];          
            std::size_t K_w   = kernel[3];          

            if (C_in != K_in)
                throw Exception("Input channels must match kernel channels.");

            std::size_t H_out = (H_in + 2 * padding[0] - K_h) / strides[0] + 1;
            std::size_t W_out = (W_in + 2 * padding[1] - K_w) / strides[1] + 1;

            return Shape{N, C_out, H_out, W_out};
    }

    void forward(Tensor const&, Tensor const&, Tensor&) const;
};


template<auto Dimension, Expression Signal, Expression Kernel>
constexpr auto convolve(Signal&& signal, Kernel&& kernel, std::size_t strides, std::size_t padding) {
    if (strides == 0 | strides == 0) 
        throw Exception("Strides should be non zero.");
        
    return Transformation<Convolution<Dimension>, Signal, Kernel>(
        {{strides}, {padding}}, 
        std::forward<Signal>(signal), 
        std::forward<Kernel>(kernel)
    );
} 
 

template<auto Dimension, Expression Signal, Expression Kernel>
constexpr auto convolve(Signal&& signal, Kernel&& kernel, std::array<std::size_t, Dimension> strides, std::array<std::size_t, Dimension> padding) {
    if (strides[0] == 0 | strides[1] == 0) 
        throw Exception("Strides should be non zero.");
        
    return Transformation<Convolution<Dimension>, Signal, Kernel>(
        {strides, padding}, 
        std::forward<Signal>(signal), 
        std::forward<Kernel>(kernel)
    );
} 
 
} // namespace transformation
 
using transformation::convolve;

} // namespace tannic

#endif // CONVOLUTIONS_HPP