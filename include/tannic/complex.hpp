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
 
#ifndef COMPLEX_HPP
#define COMPLEX_HPP

#include "concepts.hpp"
#include "types.hpp"
#include "shape.hpp"
#include "strides.hpp"
#include "traits.hpp"  
#include <cassert>

namespace tannic {
 
class Tensor; 

} namespace tannic::expression { 
 
struct Cartesian {
    static void forward(Tensor const&, Tensor const&, Tensor&);
};
 
struct Polar {  
    static void forward(Tensor const&, Tensor const&, Tensor&);
}; 

 
template<class Coordinates, Composable ... Sources>
class Complexification;
 
template<class Coordinates, Composable Source>
class Complexification<Coordinates, Source> {
public:
    typename Trait<Source>::Reference source;

    constexpr Complexification(Trait<Source>::Reference source)
    :   source(source) {
        switch (source.dtype()) {
            case float32: dtype_ = complex64; break;
            case float64: dtype_ = complex128; break;
            default:
                throw Exception("Complex view error: source tensor dtype must be float32 or float64");
        }

        if (source.shape().back() != 2) {
            throw Exception("Complex view error: last dimension must be size 2 (real + imag).");
        }
 
        shape_   = Shape(source.shape().begin(), source.shape().end() - 1); 
        strides_ = Strides(source.strides().begin(), source.strides().end() - 1);

        for (int dimension = 0; dimension < strides_.rank(); ++dimension) {
            strides_[dimension] /= 2;
        } 
        
        strides_[-1] = 1;  
    }

    constexpr type dtype() const {
        return dtype_;
    }
    
    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides const& strides() const {
        return strides_;
    }

    std::ptrdiff_t offset() const {
        return source.offset();
    } 
 
    Tensor forward(Context const& context) const;
    
private:
    type dtype_; 
    Shape shape_;
    Strides strides_;
};
 
template<class Coordinates, Composable Real, Composable Imaginary>
class Complexification<Coordinates, Real, Imaginary> {
public:
    typename Trait<Real>::Reference real;
    typename Trait<Imaginary>::Reference imaginary;
  
    constexpr Complexification(typename Trait<Real>::Reference real, typename Trait<Imaginary>::Reference imaginary)
    :   real(real)
    ,   imaginary(imaginary)
    {  
        if (real.shape() != imaginary.shape() | real.strides() != imaginary.strides()) 
            throw Exception("Complexification error: real and imaginary part layouts must match");

        if (real.dtype() == float64 || imaginary.dtype() == float64) {
            dtype_ = complex128;
        } 
        
        else {
            dtype_ = complex64;
        } 
    } 
 
    constexpr type dtype() const {
        return dtype_;
    } 
 
    constexpr Shape const& shape() const {
        return real.shape();
    }
 
    constexpr Strides const& strides() const {
        return real.strides();
    } 

    std::ptrdiff_t offset() const {
        return 0;
    } 
 
    Tensor forward(Context const& context) const;
    
private:
    type dtype_;  
};
 
template<Composable Source>
class Realification {
public:
    typename Trait<Source>::Reference source;
 
    constexpr Realification(Trait<Source>::Reference source)
    :   source(source) { 
        switch (source.dtype()) {
            case complex64:  dtype_ = float32; break;
            case complex128: dtype_ = float64; break;
            default:
                throw Exception("Real view error: source tensor dtype must be complex64 or complex128");
        }
 
        if (source.strides()[-1] != 1) {
            throw Exception("Real view error: source tensor is not in interleaved real/imag format");
        }
 
        shape_ = Shape(source.shape().begin(), source.shape().end());
        shape_.expand(2);
 
        strides_ = Strides(source.strides().begin(), source.strides().end());
        strides_.expand(1);  
        for (auto dimension = 0; dimension < strides_.rank() - 1; ++dimension) {
            strides_[dimension] *= 2;   
        }
    } 

    constexpr type dtype() const { 
        return dtype_; 
    }
 
    constexpr Shape const& shape() const { 
        return shape_; 
    }
 
    constexpr Strides const& strides() const { 
        return strides_; 
    }

    std::ptrdiff_t offset() const { 
        return source.offset(); 
    }

    Tensor forward(Context const& context) const;

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
};   


/****************************************************************************/

/**
 * @brief Creates a complex tensor view from interleaved real/imaginary data
 * @param real Tensor with shape [...,2N] containing N real/imaginary pairs
 * @return Complex tensor view with shape [...,N] and dtype complex64/128
 *
 * #### Requirements:
 * - Last dimension must be even-sized (pairs of values)
 * - Elements must be contiguous (stride 1 in last dimension)
 * - Dtype must be float32 or float64
 *
 * #### Example:
 * ```cpp 
 * // float32 input with shape [4]
 * Tensor real = {1.0f, 2.0f, 3.0f, 4.0f};
 * 
 * // complex64 output with shape [2]
 * Tensor cplx = complexify(real); 
 * // cplx = [1+2i, 3+4i]
 * ```
 */
template<Composable Real>
constexpr auto complexify(Real&& real) {
    return Complexification<Cartesian, Real>{std::forward<Real>(real)};
}

/**
 * @brief Creates complex tensor from separate real and imaginary tensors  
 * @param real Tensor containing real components
 * @param imaginary Tensor containing imaginary components
 * @return Complex tensor combining both components
 *
 * #### Requirements:
 * - Both tensors must have identical shapes and strides
 * - Dtypes must match (float32→complex64, float64→complex128)
 *
 * #### Example:
 * ```cpp
 * Tensor r = {1.0, 2.0};  // real parts
 * Tensor i = {3.0, 4.0};  // imag parts
 * 
 * Tensor c = complex(r, i);
 * // c = [1+3i, 2+4i] 
 * ```
 */
template<Composable Real, Composable Imaginary>
constexpr auto complex(Real&& real, Imaginary&& imaginary) {
    return Complexification<Cartesian, Real, Imaginary>{
        std::forward<Real>(real), 
        std::forward<Imaginary>(imaginary)
    };
}

/**
 * @brief Creates complex tensor from polar coordinates (magnitude/angle)
 * @param rho Tensor containing magnitudes
 * @param theta Tensor containing angles in radians
 * @return Complex tensor in Cartesian form
 *
 * #### Requirements:
 * - Both tensors must have identical shapes
 * - Angles must be in radians
 *
 * #### Example:
 * ```cpp
 * Tensor mag = {1.0, 2.0};    // magnitudes
 * Tensor ang = {0.0, M_PI/2}; // angles 
 * 
 * Tensor c = polar(mag, ang);
 * // c = [1+0i, 0+2i]  // cos(0)=1, sin(π/2)=1
 * ```
 */
template<Composable Magnitude, Composable Angle>
constexpr auto polar(Magnitude&& rho, Angle&& theta) {
    return Complexification<Polar, Magnitude, Angle>{
        std::forward<Magnitude>(rho), 
        std::forward<Angle>(theta)
    };
}

/**
 * @brief Creates a real-valued view of complex tensor data
 * @param complex Complex tensor to reinterpret
 * @return Real tensor view exposing [real, imag] components
 *
 * #### Transformation Rules:
 * - Dtype: complex64 → float32, complex128 → float64
 * - Shape: [...,N] → [...,N,2] (adds dimension for components)
 * - Memory: Maintains same storage with adjusted strides
 *
 * #### Requirements:
 * - Input must be complex64 or complex128
 * - Must have stride 1 in last dimension (contiguous complex pairs)
 *
 * #### Example:
 * ```cpp
 * // complex64 input with shape [2]
 * Tensor cplx = {1+2i, 3+4i};
 * 
 * // float32 output with shape [2,2] 
 * Tensor real_view = realify(cplx);
 * // real_view = [[1, 2],  // real, imag components
 * //              [3, 4]] 
 * ```
 */
template<Composable Complex>
constexpr auto realify(Complex&& complex) {
    return Realification<Complex>{std::forward<Complex>(complex)};
} 

} namespace tannic {
    
using expression::complex;
using expression::complexify;
using expression::realify;
using expression::polar;
 
} // namespace tannic

#endif // COMPLEX_HPP 