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

#ifndef SLICES_HPP
#define SLICES_HPP

/**
 * @file Slices.hpp 
 * @author Eric Hermosis
 * @date 2025  
 * @brief Implements tensor slicing for expression templates in the Tannic Tensor Library.
 *
 * This header defines the `Slice` class template, which represents a view or subrange
 * of a tensor without copying its data. It supports chained indexing, assignment,
 * and comparison operations while preserving the underlying expression structure.
 *
 * Key features:
 * - Zero-copy view into a tensor or tensor expression.
 * - Supports integer indexing and range slicing.
 * - Preserves shape, strides, and data type information.
 * - Works as part of the Tannic expression template framework.
 *
 * Example usage:
 * 
 * ```cpp
 * Tensor X(float32, {2,2,2}); X.initialize(); 
 * X[0] = 1;    //arbitrary types assignment support.
 * X[1] = 2.f;
 * std::cout << X  << std::endl; // Tensor([[[1, 1], [1, 1]], 
 *                              //          [[2, 2], [2, 2]]] dtype=float32, shape=(2, 2, 2))
 * Tensor Y = X[1]; 
 * std::cout << Y << std::endl; // Tensor([[2, 2], 
 *                             //          [2, 2]] dtype=float32, shape=(2, 2))
 * std::cout << Y[0] << std::endl; // Tensor([2, 2] dtype=float32, shape=(2))
 * ```
 */ 

#include <tuple>
#include <utility>
#include <cstddef>
#include <vector>
 
#include "types.hpp" 
#include "traits.hpp"
#include "shape.hpp"
#include "strides.hpp"
#include "indexing.hpp"  

namespace tannic {
class Tensor;
}

namespace tannic::expression {   

/**
 * @class Slice
 * @brief Expression template representing a tensor slice or subview.
 *
 * @tparam Source  The expression type from which the slice is taken (e.g., a Tensor).
 * @tparam Indexes Variadic list of index types (integer or `indexing::Range`).
 *
 * A `Slice` stores:
 * - The source expression reference.
 * - A tuple of index specifications.
 * - The computed shape, strides, and offset for the slice.
 *
 * It supports further slicing through `operator[]`, element assignment, and scalar
 * comparison for rank-0 slices.
 *
 * This class does not own the underlying data; it merely refers to a subset of it.
 */
template <Expression Source, class... Indexes>
class Slice {  
public:    

    /**
     * @brief Create a slice from a source expression and an index tuple.
     * @param source   Reference to the source expression.
     * @param indexes  Tuple containing integer indices or ranges.
     *
     * The constructor computes the resulting shape, strides, and byte offset
     * based on the given indexes.
     */
    constexpr Slice(typename Trait<Source>::Reference source, std::tuple<Indexes...> indexes)
    :   dtype_(source.dtype())  
    ,   source_(source)
    ,   indexes_(indexes)
    {   
        if (source.rank() == 0) { 
            shape_ = Shape{};
            strides_ = Strides{};
            offset_ = 0;  
        }

        else {
            std::array<Shape::size_type, Shape::limit> shape{};
            std::array<Strides::size_type, Strides::limit> strides{};
            Shape::rank_type dimension = 0;
            Shape::rank_type rank = 0;   
            offset_ = 0;
            auto process = [&](const auto& argument) {
                using Argument = std::decay_t<decltype(argument)>;
                if constexpr (std::is_same_v<Argument, indexing::Range>) { 
                    auto range = normalize(argument, source.shape()[dimension]);
                    auto size = range.stop - range.start; 
                    shape[rank] = size;
                    strides[rank] = source.strides()[dimension]; 
                    offset_ += range.start * source.strides()[dimension] * dsizeof(dtype_); 
                    rank++; dimension++;
                } 
                
                else if constexpr (std::is_integral_v<Argument>) { 
                    auto index = indexing::normalize(argument, source.shape()[dimension]);
                    offset_ += index * source.strides()[dimension] * dsizeof(dtype_);
                    dimension++;
                } 
                
                else {         
                    throw Exception("Unknown index type"); 
                } 
            };

            std::apply([&](const auto&... arguments) {
                (process(arguments), ...);
            }, indexes);

            while (dimension < source.rank()) {
                shape[rank] = source.shape()[dimension];
                strides[rank] = source.strides()[dimension];
                rank++; dimension++;
            } 

            shape_ = Shape(shape.begin(), shape.begin() + rank);
            strides_ = Strides(strides.begin(), strides.begin() + rank); 
        } 
    }    

    /**
     * @brief Index into the slice with an integer.
     * @param index The index into the current slice's first dimension.
     * @return A new `Slice` object with the updated index tuple.
     */
    template<Integral Index>
    constexpr auto operator[](Index index) const { 
        return Slice<Source, Indexes..., Index>(source_, std::tuple_cat(indexes_, std::make_tuple(index)));
    } 

    /**
     * @brief Index into the slice with a range.
     * @param range The range to select from the current slice's first dimension.
     * @return A new `Slice` object with the updated index tuple.
     */
    constexpr auto operator[](indexing::Range range) const {    
        return Slice<Source, Indexes..., indexing::Range>(source_, std::tuple_cat(indexes_, std::make_tuple(range)));
    }  

    /**
     * @brief Assigns a scalar value to all elements in the slice.
     *
     * This method writes the given value into every element referenced by the slice.
     * If the slice has rank 0 (scalar), it directly assigns the value to the single element.
     * If the slice has higher rank, it iterates over all elements in the slice and assigns the value.
     *
     * @tparam T A type convertible to the underlying element type of the tensor.
     * @param value The scalar value to assign.
     *
     * @note The value is cast to the slice's runtime `dtype_` before assignment.
     * @note This operation modifies the underlying tensor/expression in-place.
     *
     * ```cpp
     * Tensor tensor({3, 3}); tensor.initialize();
     * tensor[{0,2}][1] = 42; // assign 42 to elements in rows 0..1, column 1
     * ```
     */
    template<typename T>
    void operator=(T value);



    /**
     * @brief Compares a scalar value to the element in a rank-0 slice.
     *
     * This method checks if the scalar stored in the slice equals the given value.
     * It only works for rank-0 (scalar) slices; otherwise, an assertion will fail.
     *
     * @tparam T A type convertible to the underlying element type of the tensor.
     * @param value The scalar value to compare against.
     * @return `true` if the value matches the element in the slice, `false` otherwise.
     *
     * @warning Will assert if called on a slice with `rank() > 0`.
     *
     * ```cpp 
     * if (tensor[1][2] == 5) {
     *     // Element at (1, 2) is equal to 5
     * }
     * 
     * else if(tensor[1,1] == 3) {
     *      // Element at (1,1) is equal to 3
     * }
     * ```
     */
    template<typename T>
    bool operator==(T value) const;  

    /**
     * @brief Returns the runtime data type of elements in this slice.
     *
     * The data type (`dtype_`) is determined from the source expression or tensor
     * at construction and remains constant for the lifetime of the slice.
     *
     * @return The data type of the elements as a `type` enum.
     *
     * @note This is a runtime property, not a compile-time type.
     */
    constexpr auto dtype() const {
        return dtype_;
    } 

    /**
     * @brief Returns the number of dimensions in this slice.
     *
     * The rank is computed after applying all indexing operations
     * (ranges and/or integer indices) to the source tensor. Integral
     * index removes the corresponding dimension from the rank while 
     * range based slices may preserve it.
     * 
     * @return The rank (number of dimensions) of the slice.
     *
     * @note Scalar slices have rank 0.
     */
    constexpr auto rank() const {
        return shape_.rank();
    } 

    /**
     * @brief Returns the shape (size in each dimension) of this slice.
     *
     * The shape is derived from the source tensor after applying the
     * given indexes according to these rules:
     * 1. For each index in the index tuple:
     *    - If it's a range (`indexing::Range`):
     *      - The dimension size becomes `range.stop - range.start`
     *      - This dimension is preserved in the output shape
     *    - If it's an integer index:
     *      - The dimension is eliminated (not included in output shape)
     * 2. Any remaining dimensions from the source tensor (after processing all indices) 
     *    are preserved with their original sizes
     *
     * Example:
     * ```cpp
     * Tensor X({4,3,2});  // shape [4,3,2]
     * auto slice = X[range(1,3)][1]; 
     * // Resulting shape: [2, 2] (from range(1,3) and preserving last dimension)
     * ```
     *
     * @return A constant reference to the `Shape` object describing this slice.
     */
    constexpr Shape const& shape() const {
        return shape_;
    }


    /**
     * @brief Returns the memory strides for this slice.
     *
     * Strides represent the number of elements to skip in each dimension
     * to move to the next element along that axis. They are computed as:
     * 1. For each index in the tuple:
     *    - If it's a range:
     *      - The stride for this dimension is copied from the source tensor
     *    - If it's an integer index:
     *      - No stride is recorded (dimension is eliminated)
     * 2. Remaining dimensions keep their original strides
     *
     * The strides are always computed in elements (not bytes) and are
     * relative to the original tensor's memory layout.
     *
     * @return A constant reference to the `Strides` object for this slice.
     */
    constexpr Strides const& strides() const {
        return strides_;
    }  

    /**
     * @brief Returns the byte offset from the source tensor's data pointer.
     *
     * The offset is calculated as:
     * 1. Initialized to 0
     * 2. For each index:
     *    - If it's a range:
     *      - Add `range.start * source_stride * element_size` to offset
     *    - If it's an integer index:
     *      - Add `index * source_stride * element_size` to offset
     * 3. The offset is relative to the source tensor's data pointer
     *
     * This represents the starting memory position of the slice within
     * the original tensor's storage.
     *
     * @return The byte offset from the source tensor's data pointer.
     */
    std::ptrdiff_t offset() const {
        return offset_ + source_.offset();
    }

    std::byte* bytes() {
        return source_.bytes() + offset_;
    }

    std::byte const* bytes() const {
        return source_.bytes() + offset_;
    } 

    Tensor forward() const;
 
    void assign(std::byte const* value, std::ptrdiff_t offset); 
    void assign(bool const*, std::ptrdiff_t); 
    bool compare(std::byte const* value, std::ptrdiff_t offset) const;

private: 
    type dtype_;
    Shape shape_;
    Strides strides_;
    std::ptrdiff_t offset_;
    typename Trait<Source>::Reference source_;              
    std::tuple<Indexes...> indexes_;    
};   

template<typename T>
inline std::byte const* tobytes(T const& reference) { 
    return reinterpret_cast<std::byte const*>(&reference);
}  

template <Expression Source, class... Indexes>
template <typename T>
void Slice<Source, Indexes...>::operator=(T value) {    
    auto copy = [this](std::byte const* value, std::ptrdiff_t offset) {
        if(rank() == 0) { 
            assign(value, offset);
            return;
        } 
        std::vector<std::size_t> indexes(rank(), 0);
        bool done = false;  
        
        while (!done) {
            std::size_t position = offset;
            for (auto dimension = 0; dimension < rank(); ++dimension) {
                position += indexes[dimension] * strides_[dimension] * dsizeof(dtype_);
            } 
  
            assign(value, position); 
            done = true;
            for (int dimension = rank() - 1; dimension >= 0; --dimension) {
                if (++indexes[dimension] < shape_[dimension]) {
                    done = false;
                    break;
                }
                indexes[dimension] = 0;
            }
        }
    };

    switch (dtype_) {
        case int8:  {   int8_t casted = value; copy(tobytes(casted), offset()); break; }
        case int16: {  int16_t casted = value; copy(tobytes(casted), offset()); break; }
        case int32: {  int32_t casted = value; copy(tobytes(casted), offset()); break; }
        case int64: {  int64_t casted = value; copy(tobytes(casted), offset()); break; }
        case float32: {  float casted = value; copy(tobytes(casted), offset()); break; }
        case float64: { double casted = value; copy(tobytes(casted), offset()); break; } 
        default: throw Exception("Unsupported dtype for assignment");
    } 
}

template <Expression Source, class... Indexes>
template <typename T>
bool Slice<Source, Indexes...>::operator==(T value) const {    
    if (rank() != 0)
        throw Exception("Cannot compare an scalar to a non scalar slice");
 
    switch (dtype_) { 
        case int8:  {    int8_t casted = value; return compare(tobytes(casted), offset()); }
        case int16: {   int16_t casted = value; return compare(tobytes(casted), offset()); }
        case int32: {   int32_t casted = value; return compare(tobytes(casted), offset()); }
        case int64: {   int64_t casted = value; return compare(tobytes(casted), offset()); }
        case float32:  {  float casted = value; return compare(tobytes(casted), offset()); }
        case float64:  { double casted = value; return compare(tobytes(casted), offset()); }  
        default: throw Exception("Unsupported dtype for comparison");
    }
}   
 
} // namespace tannic::expression
 
#endif // SLICES_HPP 