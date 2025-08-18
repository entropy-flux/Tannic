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
 
#ifndef TENSOR_HPP
#define TENSOR_HPP 

/**
 * @file Tensor.hpp
 * @author Eric Hermosis
 * @date 2025
 * @brief Core multidimensional tensor class for the Tannic Tensor Library. 
 * 
 * Defines `tannic::Tensor`, the primary data structure of the library, supporting:
 * 
 * - **Dynamic dtypes**: No need for templates, data types can be passed to tensors as arguments to simplify serialization and IO operations.
 * 
 * - **Memory management**: Host(CPU)/Device(eg. CUDA) allocation support.
 * 
 * - **Math operators and functions**: Math operators like +,*,- and matmul supported, for both CPU and CUDA tensors. 
 * 
 * - **Slicing/views**: Python-like indexing (`operator[]`, `range`, `transpose`) with negative index.
 * 
 * - **Eager evaluation**: Eager execution but allowing compile time optimizations like expression fusion using templates.
 * 
 * - **Strided storage**: Non-contiguous layouts via `Shape` and `Strides`.  
 *
 * @see Types.hpp(data types), Resources.hpp (Memory Managment), Functions.hpp (Supported functions)
 */

#include <iostream>  
#include <memory>
#include <cassert> 
#include <utility>  
 
#include "Types.hpp"
#include "Shape.hpp" 
#include "Strides.hpp" 
#include "Buffer.hpp"
#include "Slices.hpp" 
#include "Views.hpp"
#include "Operations.hpp" 
#include "Complex.hpp"
#include "Graph.hpp"

namespace tannic {  

/**
 * @class Tensor
 * @brief A multidimensional, strided tensor data structure.
 *
 * @details
 * The `Tensor` class is the primary data structure in the Tannic Tensor Library.
 * It represents a dynamic typed intention of a computation. It supports
 * slicing, transposition, and expression composition.
 *
 * #### Evaluation Mode:
 * 
 * - Tensors are currently **eager**: expressions assigned to them are immediately evaluated and stored.
 * - A lazy `constexpr` evaluation mode is planned for future versions, enabling compile-time computational graphs.
 *
 * #### Memory:
 *  
 * - Host and Device (e.g., CUDA) allocations are supported using allocators like `Host()` or `Device()`.
 * - Memory must be explicitly allocated via `initialize(Host())` or initialize(Device())  but maybe removed in the future.
 *
 * #### Example:
 * 
 * ```cpp
 * using namespace tannic; 
 *
 * Tensor X(float32, {2,2}); X.initialize(); // or X.initialize(Device()); for CUDA support   
 * X[0, range{0,-1}] = 1; 
 * X[1,0] = 3;
 * X[1,1] = 4;   
 *  
 * Tensor Y(float32, {1,2}); Y.initialize();  // or X.initialize(Device()); for CUDA support   
 * Y[0,0] = 4;
 * Y[0,1] = 6;    
 * 
 * std::cout << log(X) + Y * Y - exp(X) + matmul(X, Y.transpose());
 * ```
 *
 * This example demonstrates eager initialization, advanced indexing, broadcasting, and composite expression evaluation.
 * 
 * @warning: 
 * Explicit inialization required for now but maybe removed in the future.
 * If not properly initialized the tensors may segfault instead of throwing assertion error. 
 * This will be fixed when resources can be infered at the end of a templated expression.
 */
class Tensor {
public: 
    using rank_type = uint8_t;      ///< Type used for rank (number of dimensions).
    using size_type = std::size_t;  ///< Type used for size and shape dimensions.

    /**
     * @brief Constructs an uninitialized tensor with default (contiguous) strides.
     * @param dtype Data type of the tensor.
     * @param shape Shape (dimensions) of the tensor.
     */
    Tensor(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_) 
    ,   offset_(0)  {
        nbytes_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{}) * dsizeof(dtype_);
    }   

    /**
     * @brief Constructs an uninitialized tensor with custom strides and offset.
     * @param dtype Data type of the tensor.
     * @param shape Shape of the tensor.
     * @param strides Strides per dimension.
     * @param offset Byte offset from start of underlying memory buffer.
     */
    Tensor(type dtype, Shape shape, Strides strides, std::ptrdiff_t offset = 0)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides) 
    ,   offset_(offset)  {  
        if (rank() == 0) {
            nbytes_ = dsizeof(dtype_);
        }
        else {
            std::size_t nbytes = 0;
            std::size_t expected = 0;
            for (std::size_t dimension = 0; dimension < rank(); ++dimension) { 
                nbytes += strides_[dimension] * (shape_[dimension] - 1);
            } 
            nbytes_ = (nbytes + 1) * dsizeof(dtype_);
        }
    }  

    /**
     * @brief Constructs a tensor by forwarding an `Expression`-like object.
     * @tparam Expression Expression type satisfying the `Expression` concept.
     * @param expression An expression to evaluate and store as a tensor.
     */
    template <Expression Expression>
    Tensor(const Expression& expression) {
        *this = expression.forward(); 
    } 

    /**
     * @brief Assigns the result of an expression to the tensor.
     * @tparam Expression Expression type satisfying the `Expression` concept.
     * @param expression Expression to evaluate.
     * @return Reference to `*this`.
     */
    template <Expression Expression>
    Tensor& operator=(const Expression& expression) {
        *this = expression.forward(); 
        return *this;
    } 
 

public:  
    /// @name Metadata Access (May be constexpr in the future.)
    /// @{

    /// Returns the tensor's data type.
    type dtype() const { 
        return dtype_; 
    } 

    /// Returns the tensor's shape (dimension sizes per dimension).
    Shape const& shape() const {  
        return shape_; 
    }

    /// Returns the tensor's size at a given dimension.
    Shape::size_type size(int dimension) const {
        return shape_[dimension];
    }

    /// Returns the tensor's strides (step sizes per dimension).
    Strides const& strides() const { 
        return strides_; 
    } 
 
    /// Returns the offset of the tensor in the current buffer. 
    std::ptrdiff_t offset() const {
        return offset_;
    }   

    /// Returns the total number of bytes occupied by the tensor's elements.
    std::size_t nbytes() const { 
        return nbytes_;
    } 
    
    /// Returns the number of dimensions (rank) of the tensor.
    rank_type rank() const { 
        return shape_.rank(); 
    }          

    /// Returns a reference to this tensor (const-qualified).
    Tensor& forward() { 
        if (!is_initialized())
            initialize(); 
        return *this;
    } 

    /// Returns a reference to this tensor (non-const).
    Tensor const& forward() const {
        if (!is_initialized())
            initialize(); 
        return *this;
    } 
    /// @}
 
public:   
    /// @name Memory Management (Always runtime.)
    /// @{

    /**
     * @brief Allocates the memory buffer for the tensor.
     * @param allocator Memory allocator (defaults to `Host{}`).
     */
    void initialize(Allocator allocator = Host{}) const;
   
    /**
     * @brief Returns a pointer to the beginning of the tensor's data (accounting for offset).
     * @return Pointer to the tensor's data in bytes.
     */
    std::byte* bytes() const {
        return static_cast<std::byte*>(buffer_->address()) + offset_;
    }  

    /**
     * @brief Checks whether the tensor has been initialized with memory.
     * @return True if initialized, false otherwise.
     */
    bool is_initialized() const {
        return buffer_ ? true : false;
    } 

    /**
     * @brief Returns a reference to the allocator variant used to allocate this tensor's buffer.
     * @return Allocator reference.
     * @note Asserts if the tensor is not initialized.
     */
    Allocator const& allocator() const {
        if (!is_initialized())
            throw Exception("Cannot get resource of an initializer tensor."); 
        return buffer_->allocator();
    }   
    /// @}
   
public: 
    /// @name Indexing, Slicing and Views.
    /// @{

    /**
     * @brief Indexes the tensor with a single integral index.
     * @param index The index to access.
     * @return A slice expression representing the sub-tensor.
     * 
     * #### Example:
     * 
     * ```cpp
     * using namespace tannic; 
     *
     * Tensor X(float32, {2,2,2}); X.initialize(); 
     * X[0] = 1;    //arbitrary types assignment support.
     * X[1] = 2.f;
     * std::cout << X  << std::endl; // Tensor([[[1, 1], [1, 1]], 
     *                              //          [[2, 2], [2, 2]]] dtype=float32, shape=(2, 2, 2))
     * 
     * Tensor Y = X[1]; 
     * std::cout << Y << std::endl; // Tensor([[2, 2], 
     *                             //          [2, 2]] dtype=float32, shape=(2, 2))
     *  
     * std::cout << Y[0] << std::endl; // Tensor([2, 2] dtype=float32, shape=(2))
     * ```
     */
    template<Integral Index>
    auto operator[](Index index) const {   
        if (!is_initialized())
            initialize(); 
        return expression::Slice<Tensor, Index>(*this, std::make_tuple(index));
    }

    /**
     * @brief Indexes the tensor with a `Range` object.
     * @param range Range to apply to the first dimension.
     * @return A slice expression representing the sub-tensor.
     * 
     * #### Example:
     * 
     * ```cpp
     * using namespace tannic; 
     *
     * Tensor X(float32, {2,2,2}); X.initialize();  
     * X[{0,1}] = 5; // same as X[range{0,1}] = 5
     * ```
     */
    auto operator[](indexing::Range range) const { 
        if (!is_initialized())
            initialize(); 
        return expression::Slice<Tensor, indexing::Range>(*this, std::make_tuple(range));
    } 

    /**
     * @brief Indexes the tensor with multiple indices (e.g., integers or ranges).
     * @tparam Indexes Variadic index types.
     * @param indexes Indices to apply.
     * @return A slice expression representing the sub-tensor.
     * 
     * #### Example:
     * 
     * ```cpp
     * using namespace tannic; 
     *
     * Tensor X(float32, {2,2}); X.initialize(); // or X.initialize(Device()); for CUDA support   
     * X[0, range{0,-1}] = 1;  //  fills [0,0] and [0,1] with 1.
     * X[1,0] = 3;
     * X[1,1] = 4;       
     * ```
     */
    template<class ... Indexes>
    auto operator[](Indexes... indexes) const {
        if (!is_initialized())
            initialize(); 
        return expression::Slice<Tensor, Indexes...>(*this, std::make_tuple(indexes...));
    } 

    /**
     * @brief Returns a view of the tensor with two dimensions transposed.
     * @param first First dimension to swap (default: -1).
     * @param second Second dimension to swap (default: -2).
     * @return Transpose expression.
     * 
     */
    auto transpose(int first = -1, int second = -2) const {
        if (!is_initialized())
            initialize(); 
        return expression::Transpose<Tensor>(*this, std::make_pair<int, int>(std::move(first), std::move(second)));
    }  

    /// @}

public:
    Tensor(type dtype, Shape shape, Strides strides, std::ptrdiff_t offset, std::shared_ptr<Buffer> storage)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides)
    ,   offset_(offset)   
    ,   buffer_(std::move(storage)) 
    {
        node_ = std::make_shared<Node>(*this);
    }

protected:    
    template <Expression Source, class... Indexes> 
    friend class expression::Slice;

    template <Expression Source> 
    friend class expression::Transpose;

    template <Expression Source>
    friend class expression::Reshape;

    template <class Coordinates, Expression... Sources>
    friend class expression::Complexification;

    template <Expression Source>
    friend class expression::Realification;

    void assign(std::byte const*, std::ptrdiff_t); 
    bool compare(std::byte const*, std::ptrdiff_t) const; 
     

public:
    Node* node() const { 
        return node_.get();
    } 

private:
    // Note: I didn't decide yet if put all this inside node.
    // That will speed tensor copies but disallow make the tensors
    // only runtime since shared ptrs don't support constexpr.
    // For now kernel optimization is more important.
     
    type dtype_;
    Shape shape_; 
    Strides strides_; 
    std::size_t nbytes_ = 0; 
    std::ptrdiff_t offset_ = 0;    
    mutable std::shared_ptr<Buffer> buffer_ = nullptr;
    mutable std::shared_ptr<Node> node_ = nullptr;
};    

std::ostream& operator<<(std::ostream& ostream, Tensor const& tensor);  
 
template<Expression Source> 
inline std::ostream& operator<<(std::ostream& ostream, Source source) {
    Tensor tensor = source.forward();  
    ostream << tensor;
    return ostream;
} 

template<class Operation, Expression Operand>
Tensor operation::Unary<Operation, Operand>::forward() const { 
    Tensor result(dtype(), shape(), strides(), offset());  
    operation.forward(operand, result);
    return result;
}

template<Operator Operation, Expression Operand, Expression Cooperand>
Tensor operation::Binary<Operation, Operand, Cooperand>::forward() const {
    Tensor result(dtype(), shape(), strides());
    operation.forward(operand, cooperand, result);
    return result;
}  

template<Expression Source>
Tensor expression::Reshape<Source>::forward() const { 
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
}
 
template<Expression Source>
Tensor expression::Transpose<Source>::forward() const { 
    Tensor source = source_.forward(); 
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
}   

template<Expression Source, class... Indexes>
Tensor expression::Slice<Source, Indexes...>::forward() const {   
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
}         

template<Expression Source, class... Indexes>
void expression::Slice<Source, Indexes...>::assign(std::byte const* value, std::ptrdiff_t offset) { 
    if constexpr (Trait<Source>::is_assignable) {
        source_.assign(value, offset); 
    } 

    else {
        Tensor tensor = forward();
        tensor.assign(value, offset);
    }
}  

template<Expression Source, class... Indexes>
bool expression::Slice<Source, Indexes...>::compare(std::byte const* value, std::ptrdiff_t offset) const { 
    if constexpr (Trait<Source>::is_comparable) {
        return source_.compare(value, offset); 
    } 

    else {
        Tensor tensor = forward();
        return tensor.compare(value, offset);
    }
}  
 
template<class Coordinates, Expression Source>
Tensor expression::Complexification<Coordinates, Source>::forward() const {
    Tensor real = source.forward();
    Tensor complex(dtype(), shape(), strides(), offset(), real.buffer_); 
    return complex; 
} 

template<class Coordinates, Expression Real, Expression Imaginary>
Tensor expression::Complexification<Coordinates, Real, Imaginary>::forward() const {  
    Tensor result(dtype(), shape(), strides(), offset());
    Coordinates::forward(real.forward(), imaginary.forward(), result);
    return result;
} 

template<Expression Source>
Tensor expression::Realification<Source>::forward() const {
    Tensor complex = source.forward();
    Tensor real(dtype(), shape(), strides(), offset(), complex.buffer_); 
    return real; 
}

} // namespace tannic

#endif // TENSOR_HPP