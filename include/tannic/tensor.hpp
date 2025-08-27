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
#include <initializer_list> 
 
#include "types.hpp"
#include "shape.hpp" 
#include "strides.hpp" 
#include "buffer.hpp"
#include "slices.hpp" 
#include "views.hpp"
#include "operations.hpp" 
#include "complex.hpp"
#include "graph.hpp"

namespace tannic {  

/**
 * @class Tensor
 * @brief A multidimensional, strided tensor data structure.
 * 
 * The `Tensor` class is the primary data structure in the Tannic Tensor Library.
 * It represents a dynamic typed intention of a computation. It supports
 * slicing, transposition, and expression composition.
 *
 * #### Evaluation Mode:
 * 
 * - Tensors are currently eager : expressions assigned to them are immediately evaluated and stored.
 * - A lazy `constexpr` evaluation mode is planned for future versions, enabling compile-time computational graphs.
 *
 * #### Memory:
 *  
 * - Host and Device (e.g., CUDA) allocations are supported using environments like `Host()` or `Device()`.
 * - Memory can be explicitly allocated via `initialize(Host())` or initialize(Device()).
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
        std::size_t nelements = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{});
        nbytes_ = nbytesof(dtype, nelements);
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
            std::size_t nelements = 0; 
            std::size_t expected = 1;
            for (std::size_t dimension = 0; dimension < rank(); ++dimension) { 
                nelements += strides_[dimension] * (shape_[dimension] - 1); 
                if (strides_[dimension] != expected) {
                    is_contiguous_ = false;
                }
                expected *= shape_[dimension];
            } 
            nbytes_ = nbytesof(dtype, nelements + 1);
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

    /// Returns the number of dimensions (rank) of the tensor.
    rank_type rank() const { 
        return shape_.rank(); 
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

    /// Returns whether the tensor's elements are in contiguous layout or not.
    bool is_contiguous() const {
        return is_contiguous_;
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

    /**
     * @brief Constructs a 1D tensor from an initializer list.
     *
     * @tparam T Element type (deduced from the initializer list).
     * @param values A list of values to populate the tensor.
     *
     * @details
     * This constructor allows **direct construction of 1D tensors** using brace-enclosed lists:
     *
     * ```cpp
     * Tensor X = {1.0f, 2.0f, 3.0f, 4.0f};  // 1D tensor of shape {4}, dtype=float32
     * ```
     *
     * The tensor is immediately initialized on host, contiguous in memory, and its dtype is deduced from `T`.
     */
    template<typename T>
    Tensor(std::initializer_list<T> const& values)
    :   dtype_(dtypeof<T>())
    ,   shape_({values.size()})
    ,   strides_(shape_)
    ,   offset_(0) {
        if (dtype_ == boolean) {
            nbytes_ = (values.size() + 7) / 8;
            initialize();
            std::ptrdiff_t index = 0; 
            for (auto const& value : values) {
                assign((bool const*)(&value), index);
                ++index;
            }
        }

        else {
            nbytes_ = values.size() * dsizeof(dtype_);
            initialize();
            size_t index = 0;
            for (auto const& value : values) {
                assign((std::byte const*)(&value), index * dsizeof(dtype_));
                ++index;
            }
        } 
    }

    /**
     * @brief Constructs a 2D tensor from a nested initializer list.
     *
     * @tparam T Element type (deduced from the nested initializer list).
     * @param values A nested list of values (rows) to populate the tensor.
     *
     * @details
     * This constructor allows **direct construction of 2D tensors**:
     *
     * ```cpp
     * Tensor Y = {
     *     {1.0f, 2.0f, 3.0f},
     *     {4.0f, 5.0f, 6.0f}
     * };  // 2D tensor of shape {2,3}, dtype=float32
     * ```
     *
     * All inner lists (rows) must have the same length. The tensor is contiguous in memory.
     */
    template<typename T>
    Tensor(std::initializer_list<std::initializer_list<T>> const& values)
        : dtype_(dtypeof<T>())
        , shape_({values.size(), values.begin()->size()})
        , strides_(shape_)
        , offset_(0) 
    {

        if (dtype_ == boolean) {
            nbytes_ = (shape_[0] * shape_[1] + 7) / 8;
            initialize();
            std::ptrdiff_t index = 0;   
            for (auto const& row : values) {
                if (row.size() != shape_[1])
                    throw Exception("All rows must have the same number of columns");
                for (auto const& value : row) {
                    assign((bool const*)(&value), index);
                    ++index;
                }
            }
        } 

        else {
            nbytes_ = shape_[0] * shape_[1] * dsizeof(dtype_);
            initialize(); 
            size_t index = 0;   
            for (auto row : values) {
                if (row.size() != shape_[1])
                    throw Exception("All rows must have the same number of columns");
                for (auto const& value : row) {
                    assign((std::byte const*)(&value), index * dsizeof(dtype_));
                    ++index;
                }
            }
        } 
    }

    /**
     * @brief Constructs a 3D tensor from a triple-nested initializer list.
     *
     * @tparam T Element type (deduced from the nested lists).
     * @param values A triple-nested list of values (matrices) to populate the tensor.
     *
     * @details
     * This constructor allows **direct construction of 3D tensors**:
     *
     * ```cpp
     * Tensor Z = {
     *     {
     *         {1.0f, 2.0f},
     *         {3.0f, 4.0f}
     *     },
     *     {
     *         {5.0f, 6.0f},
     *         {7.0f, 8.0f}
     *     }
     * };  // 3D tensor of shape {2,2,2}, dtype=float32
     * ```
     *
     * All matrices must have the same number of rows, and all rows must have the same number of columns.
     * The tensor is contiguous in memory.
     */
    template<typename T>
    Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> const& values)
        : dtype_(dtypeof<T>())
        , shape_({values.size(), values.begin()->size(), values.begin()->begin()->size()})
        , strides_(shape_)
        , offset_(0)
    {
        if (dtype_ == boolean) {
            nbytes_ = (shape_[0] * shape_[1] * shape_[2] + 7) / 8;
            initialize();
            std::ptrdiff_t index = 0;   
            for (auto const& matrix : values) {
                if (matrix.size() != shape_[1])
                    throw Exception("All matrices must have the same number of rows");
                for (auto const& row : matrix) {
                    if (row.size() != shape_[2])
                        throw Exception("All rows must have the same number of columns");
                    for (auto const& value : row) {
                        assign((bool const*)(&value), index);
                        ++index;
                    }
                }
            }
        }

        else {
            nbytes_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{}) * dsizeof(dtype_);
            initialize();
            std::cout << shape_ << std::endl;
            std::cout << nbytes_ << std::endl;
            size_t index = 0;   
            for (auto const& matrix : values) {
                if (matrix.size() != shape_[1])
                    throw Exception("All matrices must have the same number of rows");
                for (auto const& row : matrix) {
                    if (row.size() != shape_[2])
                        throw Exception("All rows must have the same number of columns");
                    for (auto const& value : row) {
                        assign((std::byte const*)(&value), index * dsizeof(dtype_));
                        ++index;
                    }
                }
            }
        } 
    }   
 

    /**
     * @brief Constructs a 4D tensor from a quadruple-nested initializer list.
     *
     * @tparam T Element type (deduced from the nested lists).
     * @param values A quadruple-nested list of values (tensors) to populate the 4D tensor.
     *
     * @details
     * This constructor allows **direct construction of 4D tensors**:
     *
     * ```cpp
     * Tensor W = {
     *     {
     *         {
     *             {1.0f, 2.0f},
     *             {3.0f, 4.0f}
     *         },
     *         {
     *             {5.0f, 6.0f},
     *             {7.0f, 8.0f}
     *         }
     *     },
     *     {
     *         {
     *             {9.0f, 10.0f},
     *             {11.0f, 12.0f}
     *         },
     *         {
     *             {13.0f, 14.0f},
     *             {15.0f, 16.0f}
     *         }
     *     }
     * };  // 4D tensor of shape {2,2,2,2}, dtype=float32
     * ```
     *
     * All inner tensors must have consistent dimensions. The tensor is contiguous in memory.
     */
    template<typename T>
    Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> const& values)
        : dtype_(dtypeof<T>())
        , shape_({
            values.size(),
            values.begin()->size(),
            values.begin()->begin()->size(),
            values.begin()->begin()->begin()->size()
        })
        , strides_(shape_)
        , offset_(0)
    {
        
        if (dtype_ == boolean) {
            nbytes_ = (shape_[0] * shape_[1] * shape_[2] * shape_[3] + 7) / 8;
            initialize();
            std::ptrdiff_t index = 0;
            for (auto const& tensor3D : values) {
                if (tensor3D.size() != shape_[1])
                    throw Exception("All 3D tensors must have the same number of matrices");
                for (auto const& matrix : tensor3D) {
                    if (matrix.size() != shape_[2])
                        throw Exception("All matrices must have the same number of rows");
                    for (auto const& row : matrix) {
                        if (row.size() != shape_[3])
                            throw Exception("All rows must have the same number of columns");
                        for (auto const& value : row) {
                            assign((bool const*)(&value), index);
                            ++index;
                        }
                    }
                }
            }
        }

        else { 
            nbytes_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{}) * dsizeof(dtype_);
            initialize();

            size_t index = 0;
            for (auto const& tensor3D : values) {
                if (tensor3D.size() != shape_[1])
                    throw Exception("All 3D tensors must have the same number of matrices");
                for (auto const& matrix : tensor3D) {
                    if (matrix.size() != shape_[2])
                        throw Exception("All matrices must have the same number of rows");
                    for (auto const& row : matrix) {
                        if (row.size() != shape_[3])
                            throw Exception("All rows must have the same number of columns");
                        for (auto const& value : row) {
                            assign((std::byte const*)(&value), index * dsizeof(dtype_));
                            ++index;
                        }
                    }
                }
            } 
        } 
    }

public: 
    /**
     * @brief Assigns values to a 1D tensor from an initializer list.
     *
     * @tparam T Element type of the initializer list.
     * @param values A list of values to assign to the tensor.
     *
     * @details
     * This operator assigns values to an **existing 1D tensor**:
     *
     * ```cpp
     * Tensor X(float32, {3});
     * X = {1, 2, 3};  // values cast to float32
     * ```
     *
     * Requirements:
     * - The tensor must already be initialized.
     * - The tensor must be rank-1.
     * - The tensor must be contiguous.
     * - The size of `values` must match the tensor shape.
     *
     * Type conversion:
     * - Values are cast to the tensor’s existing dtype before being written.
     * - This ensures that the tensor’s dtype does not change after assignment.
     */ 
    template<typename T>
    Tensor& operator=(std::initializer_list<T> const& values) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");
    
        if (!is_initialized()) 
            initialize();

        if (rank() != 1 || shape_[0] != values.size())
            throw Exception("Shape mismatch in assignment from initializer_list");
 
        if (dtype_ == boolean) {
            std::ptrdiff_t index = 0;
            for (auto const& value : values) {
                assign((bool const*)(&value), index);
                ++index;
            }
        } 
        
        else { 
            auto fill = [this, &values](auto cast) { 
                using Cast = decltype(cast);
                size_t index = 0;
                for (auto value : values) {
                    Cast casted = value;
                    assign(expression::tobytes(casted), index * dsizeof(dtype_));
                    ++index;
                }
            }; 

            switch (dtype_) {
                case int8:    fill(int8_t{});   break;
                case int16:   fill(int16_t{});  break;
                case int32:   fill(int32_t{});  break;
                case int64:   fill(int64_t{});  break;
                case float32: fill(float{});    break;
                case float64: fill(double{});   break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        }
        
        return *this;
  
    }

    /**
     * @brief Assigns values to a 2D tensor from a nested initializer list.
     *
     * @tparam T Element type of the nested initializer list.
     * @param values A nested list of values to assign to the tensor (rows).
     *
     * @details
     * This operator assigns values to an **existing 2D tensor**:
     *
     * ```cpp
     * Tensor Y(float32, {2, 3});
     * Y = {
     *     {1, 2, 3},
     *     {4, 5, 6}
     * };  // values cast to float32
     * ```
     *
     * Requirements:
     * - The tensor must already be initialized.
     * - The tensor must be rank-2.
     * - The tensor must be contiguous.
     * - All rows must have the same length, matching the tensor’s second dimension.
     *
     * Type conversion:
     * - Values are cast to the tensor’s existing dtype before being written.
     */
    template<typename T>
    Tensor& operator=(std::initializer_list<std::initializer_list<T>> const& values) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");
            
        if (!is_initialized()) 
            initialize();

        if (rank() != 2 || shape_[0] != values.size() || shape_[1] != values.begin()->size())
            throw Exception("Shape mismatch in assignment from nested initializer_list"); 

        if (dtype_ == boolean) {
            std::ptrdiff_t index = 0;
            for (auto const& row : values) {
                if (row.size() != shape_[1])
                    throw Exception("Row length mismatch in assignment from initializer_list");
                for (auto const& value : row) {
                    assign((bool const*)(&value), index);
                    ++index;
                }
            } 
        }

        else { 
            auto fill = [this, &values](auto cast) { 
                using Cast = decltype(cast);
                size_t index = 0;
                for (auto const& row : values) {
                    if (row.size() != shape_[1])
                        throw Exception("Row length mismatch in assignment from initializer_list");

                    for (auto value : row) {
                        Cast casted = value;
                        assign(expression::tobytes(casted), index * dsizeof(dtype_));
                        ++index;
                    }
                }
            };

            switch (dtype_) {
                case int8:    fill(int8_t{});   break;
                case int16:   fill(int16_t{});  break;
                case int32:   fill(int32_t{});  break;
                case int64:   fill(int64_t{});  break;
                case float32: fill(float{});    break;
                case float64: fill(double{});   break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        } 

        return *this;
    }

    /**
     * @brief Assigns values to a 4D tensor from a quadruple-nested initializer list.
     *
     * @tparam T Element type of the nested initializer list.
     * @param values A quadruple-nested list of values to assign to the tensor.
     *
     * @details
     * This operator assigns values to an **existing 4D tensor**:
     *
     * ```cpp
     * Tensor W(float32, {2, 2, 2, 2});  
     * W = {
     *     {
     *         {
     *             {1,  2}, {3,  4}
     *         },
     *         {
     *             {5,  6}, {7,  8}
     *         }
     *     },
     *     {
     *         {
     *             {9, 10}, {11, 12}
     *         },
     *         {
     *             {13, 14}, {15, 16}
     *         }
     *     }
     * };  // values cast to float32
     * ```
     *
     * Requirements:
     * - The tensor must already be initialized.
     * - The tensor must be rank-4.
     * - The tensor must be contiguous.
     * - All nested dimensions must match the tensor’s shape.
     *
     * Type conversion:
     * - Values are cast to the tensor’s existing dtype before being written.
     */
    template<typename T>
    Tensor& operator=(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> const& values) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");

        if (!is_initialized()) 
            initialize();

        if (rank() != 4 
            || shape_[0] != values.size() 
            || shape_[1] != values.begin()->size() 
            || shape_[2] != values.begin()->begin()->size() 
            || shape_[3] != values.begin()->begin()->begin()->size())
            throw Exception("Shape mismatch in assignment from quadruple-nested initializer_list");


        if (dtype_ == boolean) {
            std::ptrdiff_t index = 0;
            for (auto const& tensor3D : values) {
                if (tensor3D.size() != shape_[1])
                    throw Exception("3D tensor count mismatch");

                for (auto const& matrix : tensor3D) {
                    if (matrix.size() != shape_[2])
                        throw Exception("Matrix row count mismatch");

                    for (auto const& row : matrix) {
                        if (row.size() != shape_[3])
                            throw Exception("Row length mismatch");

                        for (auto const& value : row) {
                            assign((bool const*)(&value), index);
                            ++index;
                        }
                    }
                }
            }
            return *this;
        }

        else { 
            auto fill = [this, &values](auto cast) { 
                using Cast = decltype(cast);
                size_t index = 0;
                for (auto const& tensor3D : values) {
                    if (tensor3D.size() != shape_[1])
                        throw Exception("3D tensor count mismatch");

                    for (auto const& matrix : tensor3D) {
                        if (matrix.size() != shape_[2])
                            throw Exception("Matrix row count mismatch");

                        for (auto const& row : matrix) {
                            if (row.size() != shape_[3])
                                throw Exception("Row length mismatch");

                            for (auto value : row) {
                                Cast casted = value;
                                assign(expression::tobytes(casted), index * dsizeof(dtype_));
                                ++index;
                            }
                        }
                    }
                }
            };

            switch (dtype_) {
                case int8:    fill(int8_t{});   break;
                case int16:   fill(int16_t{});  break;
                case int32:   fill(int32_t{});  break;
                case int64:   fill(int64_t{});  break;
                case float32: fill(float{});    break;
                case float64: fill(double{});   break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        } 
        return *this;
    }



public:   
    /// @name Memory Management (Always runtime.)
    /// @{

    /**
     * @brief Allocates the memory buffer for the tensor.
     * @param environment Memory environment (defaults to `Host{}`).
     */
    void initialize(Environment environment = Host{}) const;
   
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
     * @brief Returns a reference to the environment variant used to allocate this tensor's buffer.
     * @return Environment reference.
     * @note Asserts if the tensor is not initialized.
     */
    Environment const& environment() const {
        if (!is_initialized())
            throw Exception("Cannot get resource of an initializer tensor."); 
        return buffer_->environment();
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
 
    /**
     * @brief Returns a view of the tensor with dimensions permuted.
     * @param indexes Indices to permute
     * @return Permutation expression.
     * 
     */
    template<Integral ... Indexes>
    auto permute(Indexes... indexes) const {
        if (!is_initialized())
            initialize(); 
        return expression::Permutation<Tensor, Indexes...>(*this, std::make_tuple(indexes...));
    } 

     /**
     * @brief Returns a view of the tensor with given sizes.
     * @param sizes sizes of the new shape of the tensor
     * @return View expression.
     * 
     */
    template<Integral ... Sizes>
    auto view(Sizes... sizes) const {
        if (!is_initialized())
            initialize(); 
        return expression::View<Tensor>(*this, sizes...);
    } 

    /**
     * @brief Returns a view of the tensor with all dimensions of size 1 removed.
     * @return Squeeze expression.
     *
     * #### Example:
     * ```cpp
     * using namespace tannic;
     *
     * Tensor X(float32, {1, 3, 1, 5}); X.initialize();
     * Tensor Y = X.squeeze();     // shape = (3, 5)
     * ```
     */
    auto squeeze() const {
        if (!is_initialized())
            initialize();
        return expression::Squeeze<Tensor>(*this);
    }
 

    /**
     * @brief Returns a view of the tensor with a dimension of size 1 inserted at the given axis.
     * @param axis Dimension index where the new axis is inserted (supports negative indexing).
     * @return Unsqueeze expression.
     *
     * #### Example:
     * ```cpp
     * using namespace tannic;
     *
     * Tensor X(float32, {3, 5}); X.initialize();
     * Tensor Y = X.unsqueeze(0);   // shape = (1, 3, 5)
     * Tensor Z = X.unsqueeze(2);   // shape = (3, 1, 5)
     * ```
     */
    template<Integral... Axes>
    auto unsqueeze(Axes... axes) {
        return expression::Unsqueeze<Tensor>(*this, axes...);
    }
    /// @}

public:

    Tensor(type dtype, Shape shape, std::ptrdiff_t offset, std::shared_ptr<Buffer> storage)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape)
    ,   offset_(offset)   
    ,   buffer_(std::move(storage)) 
    {
        std::size_t nelements = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{});
        nbytes_ =  nbytesof(dtype_, nelements);
        node_ = std::make_shared<Node>(*this);
    }

    Tensor(type dtype, Shape shape, Strides strides, std::ptrdiff_t offset, std::shared_ptr<Buffer> storage)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides)
    ,   offset_(offset)   
    ,   buffer_(std::move(storage)) 
    {
        if (rank() == 0) {
            nbytes_ = nbytesof(dtype_, 1);
        }
        else {
            std::size_t nelements = 0; 
            std::size_t expected = 1;
            for (std::size_t dimension = 0; dimension < rank(); ++dimension) { 
                nelements += strides_[dimension] * (shape_[dimension] - 1); 
                if (strides_[dimension] != expected) {
                    is_contiguous_ = false;
                }
                expected *= shape_[dimension];
            } 
            nbytes_ = nbytesof(dtype, nelements + 1);
        }
        node_ = std::make_shared<Node>(*this);
    }

protected:    
    template <Expression Source, class... Indexes> 
    friend class expression::Slice;

    template <Expression Source> 
    friend class expression::Transpose;

    template <Expression Source>
    friend class expression::View; 

    template <Expression Source>
    friend class expression::Squeeze; 

    template <Expression Source>
    friend class expression::Unsqueeze;

    template <Expression Source, Integral... Indexes> 
    friend class expression::Permutation; 

    template <Expression Source>
    friend class expression::Expansion;
 
    template <Expression Source>
    friend class expression::Flatten;

    template <class Coordinates, Expression... Sources>
    friend class expression::Complexification;

    template <Expression Source>
    friend class expression::Realification;

    void assign(std::byte const*, std::ptrdiff_t); 
    void assign(bool const*, std::ptrdiff_t); 

    bool compare(std::byte const*, std::ptrdiff_t) const; 
     

public:
    Node* node() const { 
        return node_.get();
    } 

private:
    // Note: I didn't decide yet if put all this inside Node.
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
    bool is_contiguous_ = true;
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
Tensor expression::View<Source>::forward() const { 
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
}
 
template<Expression Source>
Tensor expression::Squeeze<Source>::forward() const { 
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
}
 
template<Expression Source>
Tensor expression::Unsqueeze<Source>::forward() const { 
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
}
 
template<Expression Source>
Tensor expression::Flatten<Source>::forward() const { 
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
} 

template<Expression Source, Integral... Indexes>
Tensor expression::Permutation<Source, Indexes...>::forward() const {   
    Tensor source = source_.forward();
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
}         
 
template<Expression Source>
Tensor expression::Transpose<Source>::forward() const { 
    Tensor source = source_.forward(); 
    return Tensor(dtype(), shape(), strides(), offset(), source.buffer_);
}   

template<Expression Source>
Tensor expression::Expansion<Source>::forward() const { 
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
void expression::Slice<Source, Indexes...>::assign(bool const* value, std::ptrdiff_t offset) { 
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