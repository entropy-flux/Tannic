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
 
#ifndef COMPLEX_HPP
#define COMPLEX_HPP

#include "Concepts.hpp"
#include "Types.hpp"
#include "Shape.hpp"
#include "Strides.hpp"
#include "Traits.hpp"  
#include <cassert>

namespace tannic {
 
class Tensor; 
 
namespace expression { 

struct Cartesian {
    static void forward(Tensor const&, Tensor const&, Tensor&);
};

/*
struct Polar {  
    static void forward(Tensor const&, Tensor const&, Tensor&);
};
*/


template<class Coordinates, Expression ... Sources>
class Complexification;

template<class Coordinates, Expression Source>
class Complexification<Coordinates, Source> {
public:
    typename Trait<Source>::Reference source;

    constexpr Complexification(Trait<Source>::Reference source)
    :   source(source) 
    { 
        switch (source.dtype()) {
            case float32: dtype_ = complex64; break;
            case float64: dtype_ = complex128; break;
            default: assert(false && "Complex view error: source tensor dtype must be float32 or float64");
        }  

        if (source.strides()[-1] == 1 && source.strides()[-2] == 2) { 
        assert(source.shape().back() == 2 && 
            "Complex view error: last dimension must be size 2 (real + imag).");
            shape_ = Shape(source.shape().begin(), source.shape().end() - 1);
            strides_ = Strides(source.strides().begin(), source.strides().end() - 1); 
            strides_[-1] = 1; 
        } else { 
            assert(false &&
                "Complex view error: source tensor is not contiguous in last two dimensions. "
                "Cannot create complex view safely."
            );
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
 
    Tensor forward() const;
    
private:
    type dtype_; 
    Shape shape_;
    Strides strides_;
};

template<class Coordinates, Expression Real, Expression Imaginary>
class Complexification<Coordinates, Real, Imaginary> {
public:
    typename Trait<Real>::Reference real;
    typename Trait<Imaginary>::Reference imaginary;

    constexpr Complexification(typename Trait<Real>::Reference real, typename Trait<Imaginary>::Reference imaginary)
    :   real(real)
    ,   imaginary(imaginary)
    {  
        assert(real.shape() == imaginary.shape() &&     "Complexification error: real and imaginary part shapes must match");  
        assert(real.strides() == imaginary.strides() && "Complexification error: real and imaginary part shapes must match");
        if (real.dtype() == float64 || imaginary.dtype() == float64) {
            dtype_ = complex128;
        } else {
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
 
    Tensor forward() const;
    
private:
    type dtype_;  
};


template<Expression Source>
class Realification {
public:
    typename Trait<Source>::Reference source;

    constexpr Realification(Trait<Source>::Reference source)
    :   source(source) 
    { 
        switch (source.dtype()) {
            case complex64:  dtype_ = float32; break;
            case complex128: dtype_ = float64; break;
            default:
                assert(false && 
                       "Real view error: source tensor dtype must be complex64 or complex128");
        }
 
        if (source.strides()[-1] == 1) {  
            shape_ = Shape(source.shape().begin(), source.shape().end());
            shape_.expand(2);  

            strides_ = Strides(source.strides().begin(), source.strides().end());
            strides_.expand(1);  
            strides_[-2] = 2;
        } else {
            assert(false && "Real view error: source tensor is not in interleaved real/imag format");
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

    Tensor forward() const;

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
};  

template<Expression Real>
constexpr auto complex(Real&& real) {
    return Complexification<Cartesian, Real>{std::forward<Real>(real)};
} 

template<Expression Real, Expression Imaginary>
constexpr auto complex(Real&& real, Imaginary&& imaginary) {
    return Complexification<Cartesian, Real, Imaginary>{std::forward<Real>(real), std::forward<Imaginary>(imaginary)};
} 

template<Expression Complex>
constexpr auto real(Complex&& complex) {
    return Realification<Complex>{std::forward<Complex>(complex)};
}

} // namespace expression

using expression::complex;
using expression::real;
 
} // namespace tannic

#endif // COMPLEX_HPP 