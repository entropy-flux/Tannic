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

#ifndef COMPARISONS_HPP
#define COMPARISONS_HPP

/**
 * @file comparisons.hpp
 * @author Eric Hermosis
 * @date 2025
 * @brief Defines element-wise comparison operations for tensor expressions.
 *
 * This header provides **lazy-evaluated comparison operators** for `Tensor` 
 * and expression types. All comparisons are element-wise and produce 
 * boolean tensors of the same shape as the operands.  
 *
 * Supported operators:
 * - Equality and inequality:
 *   * `==` (equal)
 *   * `!=` (not equal)
 * - Relational comparisons:
 *   * `<`  (less than)
 *   * `<=` (less than or equal)
 *   * `>`  (greater than)
 *   * `>=` (greater than or equal)
 *
 * ## Example
 * ```cpp
 * #include <iostream>
 * #include <tannic.hpp>
 * #include <tannic/comparisons.hpp>
 *
 * using namespace tannic;
 *
 * int main() {
 *     Tensor A = {1, 2, 3, 4, 5};
 *     Tensor B = {5, 4, 3, 2, 1};
 *
 *     std::cout << "A == B: " << (A == B) << std::endl;
 *     std::cout << "A != B: " << (A != B) << std::endl;
 *     std::cout << "A >  B: " << (A >  B) << std::endl;
 *     std::cout << "A >= B: " << (A >= B) << std::endl;
 *     std::cout << "A <  B: " << (A <  B) << std::endl;
 *     std::cout << "A <= B: " << (A <= B) << std::endl;
 * }
 * ```
 *
 * Part of the Tannic Tensor Library.
 */

#include "concepts.hpp"
#include "expressions.hpp"
#include "shape.hpp"
#include "tensor.hpp"

namespace tannic::expression {
 
/**
 * @brief Expression template for element-wise tensor comparisons.
 *
 * Represents a lazy comparison between two tensor expressions.
 * The actual boolean tensor is only materialized when assigned
 * to a `Tensor`.
 *
 * @tparam Criteria The comparison functor (e.g., `EQ`, `LT`)
 * @tparam First    The left-hand expression type
 * @tparam Second   The right-hand expression type
 */
template<class Criteria, Composable First, Composable Second>
class Comparison : public Expression<Criteria, First, Second> {
public:   
    /**
     * @brief Constructs a comparison expression.
     * @throws Exception if the tensor shapes differ.
     */
    constexpr Comparison(Criteria criteria, typename Trait<First>::Reference first, typename Trait<Second>::Reference second) 
    :   Expression<Criteria, First, Second>(criteria, first, second)
    ,   shape_(first.shape())
    ,   strides_(shape_)
    {
        if(first.shape() != second.shape()) 
            throw Exception("Cannot compare tensors of different shape"); 
    }

    constexpr type dtype() const {
        return boolean;
    }
    
    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides  const& strides() const {
        return strides_;
    }

    constexpr std::ptrdiff_t offset() const {
        return 0;
    }
    
    Tensor forward() const {
        Tensor result(boolean, shape_, strides_, 0);
        this->operation.forward(std::get<0>(this->operands), std::get<1>(this->operands), result);
        return result;
    }


private:
    Shape shape_;
    Strides strides_;

};

} namespace tannic::comparison {

using expression::Comparison;

struct EQ {
    void forward(Tensor const&, Tensor const&, Tensor&) const;
};

struct NE { 
    void forward(Tensor const&, Tensor const&, Tensor&) const;
};

struct GT {
    void forward(Tensor const&, Tensor const&, Tensor&) const;
};

struct GE {
    void forward(Tensor const&, Tensor const&, Tensor&) const;
};

struct LT {
    void forward(Tensor const&, Tensor const&, Tensor&) const;
};

struct LE { 
    void forward(Tensor const&, Tensor const&, Tensor&) const;
};

template<Composable First, Composable Second>
constexpr auto operator==(First&& lhs, Second&& rhs) {
    return Comparison<EQ, First, Second>({}, lhs, rhs);
}

template<Composable First, Composable Second>
constexpr auto operator!=(First&& lhs, Second&& rhs) {
    return Comparison<NE, First, Second>({}, lhs, rhs);
}

template<Composable First, Composable Second>
constexpr auto operator<(First&& lhs, Second&& rhs) {
    return Comparison<LT, First, Second>({}, lhs, rhs);
}

template<Composable First, Composable Second>
constexpr auto operator<=(First&& lhs, Second&& rhs) {
    return Comparison<LE, First, Second>({}, lhs, rhs);
}

template<Composable First, Composable Second>
constexpr auto operator>(First&& lhs, Second&& rhs) {
    return Comparison<GT, First, Second>({}, lhs, rhs);
}

template<Composable First, Composable Second>
constexpr auto operator>=(First&& lhs, Second&& rhs) {
    return Comparison<GE, First, Second>({}, lhs, rhs);
}

/**
 * @brief Determine whether two tensors are element-wise equal within a tolerance.
 *
 * This function checks whether all elements of two tensors are close to each other,
 * within a relative tolerance (`rtol`) and an absolute tolerance (`atol`). 
 *
 * @param first   The first tensor to compare.
 * @param second  The second tensor to compare.
 * @param rtol    Relative tolerance. Default = 1e-5.
 *                The allowable difference grows with the magnitude of the values.
 * @param atol    Absolute tolerance. Default = 1e-8.
 *                The minimum absolute tolerance for small values.
 *
 * @return true if all corresponding elements of @p first and @p second
 *         satisfy the condition:
 *         \f$ |a - b| \leq \text{atol} + \text{rtol} \times |b| \f$,
 *         false otherwise.
 *
 * @throw tannic::Exception if the tensors have mismatched shapes
 *        or incompatible environments.
 */
bool allclose(Tensor const& first, Tensor const& second, double rtol = 1e-5f, double atol = 1e-8f);

} namespace tannic {
    
using comparison::operator==;
using comparison::operator!=;
using comparison::operator< ;
using comparison::operator<=;
using comparison::operator> ;
using comparison::operator>=;
using comparison::allclose  ;

} // namespace tannic
 
#endif // COMPARISONS_HPP