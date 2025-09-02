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
 * @file convolutions.hpp
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

#include "concepts.hpp"
#include "types.hpp"
#include "traits.hpp"
#include "shape.hpp" 
#include "tensor.hpp" 
#include "exceptions.hpp"
#include "transformations.hpp"
 
namespace tannic {

class Tensor;  

} namespace tannic::transformation { 

/**
 * @brief Expression for 1D convolution operations
 *
 * Implements 1D convolution with:
 * - Input and kernel dtype validation
 * - Automatic output shape calculation
 * - Support for strides and padding 
 */ 
class Convolution1D {
public: 
    std::array<std::size_t, 1> strides;  ///< Stride values for the convolution operation
    std::array<std::size_t, 1> padding;  ///< Padding values for the convolution operation

    /**
     * @brief Type promotion for 1D convolution
     * 
     * Requires both signal and kernel to have identical data types.
     * 
     * @param signal Data type of the input signal tensor
     * @param kernel Data type of the kernel tensor
     * @return Data type (same as inputs)
     * @throws Exception if signal and kernel dtypes don't match
     */
    constexpr type promote(type signal, type kernel) const {
        if (signal != kernel)
            throw Exception("Dtypes must match in convolutions.");
        return signal;
    }

    /**
     * @brief Computes output shape for 1D convolution
     * 
     * Calculates the output shape based on input signal shape, kernel shape,
     * strides, and padding using the standard convolution formula.
     *
     * @param signal Shape of the input signal tensor (N, C_in, L_in)
     * @param kernel Shape of the kernel tensor (C_out, K_in, K_len)
     * @return Output shape after 1D convolution (N, C_out, L_out)
     * @throws Exception if input ranks are not 3 or channel dimensions don't match
     *
     * @details Expected input shapes:
     * - Signal: [batch_size, input_channels, signal_length]
     * - Kernel: [output_channels, input_channels, kernel_length]
     * - Output: [batch_size, output_channels, output_length]
     *
     * Output length formula: L_out = (L_in + 2*padding - K_len) / stride + 1
     */
    constexpr Shape transform(Shape const& signal, Shape const& kernel) const {  
        if (signal.rank() != 3 || kernel.rank() != 3)
            throw Exception("Only rank 3 tensors supported for Conv1D."); 

        std::size_t N     = signal[0];     // Batch size
        std::size_t C_in  = signal[1];     // Input channels
        std::size_t L_in  = signal[2];     // Input length

        std::size_t C_out = kernel[0];     // Output channels
        std::size_t K_in  = kernel[1];     // Kernel input channels
        std::size_t K_len = kernel[2];     // Kernel length

        if (C_in != K_in)
            throw Exception("Input channels must match kernel channels.");

        std::size_t L_out = (L_in + 2 * padding[0] - K_len) / strides[0] + 1;
        return Shape{N, C_out, L_out};
    }

    /**
     * @brief Performs the forward pass of 1D convolution
     * 
     * @param signal Input signal tensor
     * @param kernel Convolution kernel tensor
     * @param result Output tensor to store convolution result
     */
    void forward(Tensor const& signal, Tensor const& kernel, Tensor& result) const;
};

/**
 * @brief Expression for 2D convolution operations
 *
 * Implements 2D convolution with:
 * - Input and kernel dtype validation
 * - Automatic output shape calculation
 * - Support for strides and padding in both spatial dimensions 
 */ 
class Convolution2D {
public:      
    std::array<std::size_t, 2> strides;  ///< Stride values for height and width dimensions
    std::array<std::size_t, 2> padding;  ///< Padding values for height and width dimensions
  
    /**
     * @brief Type promotion for 2D convolution
     * 
     * Requires both signal and kernel to have identical data types.
     * 
     * @param signal Data type of the input signal tensor
     * @param kernel Data type of the kernel tensor
     * @return Data type (same as inputs)
     * @throws Exception if signal and kernel dtypes don't match
     */
    constexpr type promote(type signal, type kernel) const {
        if (signal != kernel)
            throw Exception("Dtypes must match in convolutions.");
        return signal;
    }

    constexpr type promote(type signal, type kernel, type bias) const {
        if (signal != kernel || signal != bias)
            throw Exception("Signal, kernel, and bias dtypes must all match.");
        return signal;
    }


    /**
     * @brief Computes output shape for 2D convolution
     * 
     * Calculates the output shape based on input signal shape, kernel shape,
     * strides, and padding using the standard 2D convolution formula.
     *
     * @param signal Shape of the input signal tensor (N, C_in, H_in, W_in)
     * @param kernel Shape of the kernel tensor (C_out, K_in, K_h, K_w)
     * @return Output shape after 2D convolution (N, C_out, H_out, W_out)
     * @throws Exception if input ranks are not 4 or channel dimensions don't match
     *
     * @details Expected input shapes:
     * - Signal: [batch_size, input_channels, height, width]
     * - Kernel: [output_channels, input_channels, kernel_height, kernel_width]
     * - Output: [batch_size, output_channels, output_height, output_width]
     *
     * Output dimension formulas:
     * - H_out = (H_in + 2*padding_h - K_h) / stride_h + 1
     * - W_out = (W_in + 2*padding_w - K_w) / stride_w + 1
     */
    constexpr Shape transform(Shape const& signal, Shape const& kernel) const {   
        if (signal.rank() != 4 | kernel.rank() != 4) 
            throw Exception("Only rank 4 tensors supported."); 
            
        std::size_t N     = signal[0];     // Batch size
        std::size_t C_in  = signal[1];     // Input channels
        std::size_t H_in  = signal[2];     // Input height
        std::size_t W_in  = signal[3];     // Input width

        std::size_t C_out = kernel[0];     // Output channels
        std::size_t K_in  = kernel[1];     // Kernel input channels
        std::size_t K_h   = kernel[2];     // Kernel height
        std::size_t K_w   = kernel[3];     // Kernel width

        if (C_in != K_in)
            throw Exception("Input channels must match kernel channels.");

        std::size_t H_out = (H_in + 2 * padding[0] - K_h) / strides[0] + 1;
        std::size_t W_out = (W_in + 2 * padding[1] - K_w) / strides[1] + 1;

        return Shape{N, C_out, H_out, W_out};
    }

    constexpr Shape transform(Shape const& signal, Shape const& kernel, Shape const& bias) const {
        Shape out = transform(signal, kernel);
        if (bias.rank() != 1 || bias[0] != out[1])
            throw Exception("Bias shape must be [C_out].");
        return out;
    }

    /**
     * @brief Performs the forward pass of 2D convolution
     * 
     * @param signal Input signal tensor
     * @param kernel Convolution kernel tensor
     * @param result Output tensor to store convolution result
     */
    void forward(Tensor const& signal, Tensor const& kernel, Tensor& result) const;
    void forward(Tensor const& signal, Tensor const& kernel, Tensor const& bias, Tensor& result) const;
};

/**
 * @brief Creates a 1D convolution expression with uniform stride and padding
 *
 * @tparam Signal Input signal expression type
 * @tparam Kernel Convolution kernel expression type
 * @param signal Input signal tensor operand
 * @param kernel Convolution kernel tensor operand
 * @param stride Stride value applied to the signal length
 * @param padding Padding value applied to the signal length
 * @return Transformation expression representing the 1D convolution operation
 * @throws Exception if stride is zero
 */
template<Expression Signal, Expression Kernel>
constexpr auto convolve1D(Signal&& signal, Kernel&& kernel, std::size_t stride, std::size_t padding) {
    if (stride == 0)
        throw Exception("Stride must be non-zero for Conv1D.");

    return Transformation<Convolution1D, Signal, Kernel>(
        {{stride}, {padding}},
        std::forward<Signal>(signal),
        std::forward<Kernel>(kernel)
    );
}

/**
 * @brief Creates a 1D convolution expression with explicit stride and padding arrays
 *
 * @tparam Signal Input signal expression type
 * @tparam Kernel Convolution kernel expression type
 * @param signal Input signal tensor operand
 * @param kernel Convolution kernel tensor operand
 * @param strides Array of stride values (size = 1)
 * @param padding Array of padding values (size = 1)
 * @return Transformation expression representing the 1D convolution operation
 * @throws Exception if stride is zero
 */
template<Expression Signal, Expression Kernel>
constexpr auto convolve1D(
    Signal&& signal, Kernel&& kernel,
    std::array<std::size_t, 1> strides,
    std::array<std::size_t, 1> padding
)   {
    if (strides[0] == 0)
        throw Exception("Stride must be non-zero for Conv1D.");

    return Transformation<Convolution1D, Signal, Kernel>(
        {strides, padding},
        std::forward<Signal>(signal),
        std::forward<Kernel>(kernel)
    );
}

/**
 * @brief Creates a 2D convolution expression with uniform stride and padding
 * 
 * @param signal Input signal tensor operand
 * @param kernel Convolution kernel tensor operand
 * @param stride Stride value applied uniformly to both spatial dimensions
 * @param padding Padding value applied uniformly to both spatial dimensions
 * @return Transformation expression representing the 2D convolution operation
 * @throws Exception if stride is zero
 */
template<Expression Signal, Expression Kernel>
constexpr auto convolve2D(Signal&& signal, Kernel&& kernel, std::size_t stride, std::size_t padding) {
    if (stride == 0)
        throw Exception("Stride must be non-zero for Conv2D.");

    return Transformation<Convolution2D, Signal, Kernel>(
        {{stride, stride}, {padding, padding}},
        std::forward<Signal>(signal),
        std::forward<Kernel>(kernel)
    );
}

/**
 * @brief Creates a 2D convolution expression with explicit stride and padding arrays
 * 
 * @param signal Input signal tensor operand
 * @param kernel Convolution kernel tensor operand
 * @param strides Array of stride values (size = 2: {stride_h, stride_w})
 * @param padding Array of padding values (size = 2: {pad_h, pad_w})
 * @return Transformation expression representing the 2D convolution operation
 * @throws Exception if any stride value is zero
 */
template<Expression Signal, Expression Kernel>
constexpr auto convolve2D(
    Signal&& signal, Kernel&& kernel,
    std::array<std::size_t, 2> strides,
    std::array<std::size_t, 2> padding
)   {
    if (strides[0] == 0 || strides[1] == 0)
        throw Exception("Stride values must be non-zero for Conv2D.");

    return Transformation<Convolution2D, Signal, Kernel>(
        {strides, padding},
        std::forward<Signal>(signal),
        std::forward<Kernel>(kernel)
    );
}

/**
 * @brief Creates a biased 2D convolution expression with uniform stride and padding
 * 
 * @param signal Input signal tensor operand
 * @param kernel Convolution kernel tensor operand
 * @param bias Bias tensor operand (shape = [C_out])
 * @param stride Stride value applied uniformly to both spatial dimensions
 * @param padding Padding value applied uniformly to both spatial dimensions
 * @return Transformation expression representing the biased 2D convolution operation
 * @throws Exception if stride is zero
 */
template<Expression Signal, Expression Kernel, Expression Bias>
constexpr auto convolve2D(Signal&& signal, Kernel&& kernel, Bias&& bias, std::size_t stride, std::size_t padding) {
    if (stride == 0)
        throw Exception("Stride must be non-zero for Conv2D.");

    return Transformation<Convolution2D, Signal, Kernel, Bias>(
        {{stride, stride}, {padding, padding}},
        std::forward<Signal>(signal),
        std::forward<Kernel>(kernel),
        std::forward<Bias>(bias)
    );
}

/**
 * @brief Creates a biased 2D convolution expression with explicit stride and padding arrays
 * 
 * @param signal Input signal tensor operand
 * @param kernel Convolution kernel tensor operand
 * @param bias Bias tensor operand (shape = [C_out])
 * @param strides Array of stride values (size = 2: {stride_h, stride_w})
 * @param padding Array of padding values (size = 2: {pad_h, pad_w})
 * @return Transformation expression representing the biased 2D convolution operation
 * @throws Exception if any stride value is zero
 */
template<Expression Signal, Expression Kernel, Expression Bias>
constexpr auto convolve2D(
    Signal&& signal, Kernel&& kernel, Bias&& bias,
    std::array<std::size_t, 2> strides,
    std::array<std::size_t, 2> padding
) {
    if (strides[0] == 0 || strides[1] == 0)
        throw Exception("Stride values must be non-zero for Conv2D.");

    return Transformation<Convolution2D, Signal, Kernel, Bias>(
        {strides, padding},
        std::forward<Signal>(signal),
        std::forward<Kernel>(kernel),
        std::forward<Bias>(bias)
    );
}


 
} namespace tannic {
 
using transformation::convolve1D; 
using transformation::convolve2D; 

} // namespace tannic

#endif // CONVOLUTIONS_HPP