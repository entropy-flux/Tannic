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
#include "context.hpp"

namespace tannic {  
 
class Tensor {
public: 
    using rank_type = uint8_t;     
    using size_type = std::size_t;   


    Tensor() = default;
 
    Tensor(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_) 
    ,   offset_(0)  {
        nelements_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{}); 
    }
 
    Tensor(type dtype, Shape shape, Strides strides, std::ptrdiff_t offset = 0)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides) 
    ,   offset_(offset)  
    {    
        std::size_t expected = 1;
        for (std::size_t dimension = 0; dimension < rank(); ++dimension) { 
            std::size_t index = rank() - 1 - dimension;   
            if (strides_[index] != expected) {
                is_contiguous_ = false;
            }
            nelements_ *= shape_[index]; 
            expected *= shape_[index];
        }   
    }
 
    template <Composable Expression>
    Tensor(const Expression& expression) {
        Context context{};
        *this = expression.forward(context); 
    } 
 
    template <Composable Expression>
    Tensor& operator=(const Expression& expression) {
        Context context{};
        *this = expression.forward(context); 
        return *this;
    }  
 
public:   
    type dtype() const { 
        return dtype_; 
    } 
 
    rank_type rank() const { 
        return shape_.rank(); 
    }           

    Shape const& shape() const {  
        return shape_; 
    }
 
    Shape::size_type size(int dimension) const {
        return shape_[dimension];
    }
 
    Strides const& strides() const { 
        return strides_; 
    } 
  
    std::ptrdiff_t offset() const {
        return offset_;
    } 
     
    std::size_t nelements() const {  
        return nelements_;
    }  

    std::size_t nbytes() const {  
        return nbytesof(dtype_, nelements_);
    } 
 
    bool is_contiguous() const {
        return is_contiguous_;
    }

    bool is_singleton() const {
        return (nelements_ == 1);
    }
       
    Tensor& forward(Context const& context) { 
        if (!is_initialized())
            initialize(); 
        return *this;
    } 
 
    Tensor const& forward(Context const& context) const {
        if (!is_initialized())
            initialize(); 
        return *this;
    }   
 
public:    
    void initialize(Environment environment = Host{}) const;
    
    std::byte* bytes() const {
        return static_cast<std::byte*>(buffer_->address()) + offset_;
    }  
    
    bool is_initialized() const {
        return buffer_ ? true : false;
    } 

    Environment const& environment() const {
        if (!is_initialized())
            throw Exception("Cannot get resource of an initializer tensor."); 
        return buffer_->environment();
    }   
    

public: 

    template<Arithmetic T>
    Tensor(std::initializer_list<T> const& values)
    :   dtype_(dtypeof<T>())
    ,   shape_({values.size()})
    ,   strides_(shape_)
    ,   offset_(0)
    ,   nelements_(shape_[0]) {
        if (dtype_ == boolean) { 
            initialize();
            std::ptrdiff_t index = 0; 
            for (auto const& value : values) {
                assign((bool const*)(&value), index);
                ++index;
            }
        }

        else { 
            initialize();
            size_t index = 0;
            for (auto const& value : values) {
                assign((std::byte const*)(&value), index * dsizeof(dtype_));
                ++index;
            }
        } 
    }

    template<Arithmetic T>
    Tensor(std::initializer_list<std::initializer_list<T>> const& values)
    :   dtype_(dtypeof<T>())
    ,   shape_({values.size(), values.begin()->size()})
    ,   strides_(shape_)
    ,   offset_(0) 
    ,   nelements_(shape_[0] * shape_[1])
    {

        if (dtype_ == boolean) { 
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

    template<Arithmetic T>
    Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> const& values)
    :   dtype_(dtypeof<T>())
    ,   shape_({values.size(), values.begin()->size(), values.begin()->begin()->size()})
    ,   strides_(shape_)
    ,   offset_(0)
    ,   nelements_(shape_[0] * shape_[1] * shape_[2])
        
    {
        if (dtype_ == boolean) { 
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
            initialize(); 
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
 

    template<Arithmetic T>
    Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> const& values)
    :    dtype_(dtypeof<T>())
    ,    shape_({
            values.size(),
            values.begin()->size(),
            values.begin()->begin()->size(),
            values.begin()->begin()->begin()->size()
        })
    ,   strides_(shape_)
    ,   offset_(0)
    ,   nelements_(shape_[0] * shape_[1] * shape_[2] * shape_[3])
    {
        
        if (dtype_ == boolean) { 
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

    template<Arithmetic T>
    void initialize(std::initializer_list<T> values, Environment environment = Host{}) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");
      
        if (rank() != 1 || shape_[0] != values.size())
            throw Exception("Shape mismatch in assignment from initializer_list");
 
        if (!is_initialized()) 
            initialize(environment);

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
                    Cast casted(value);
                    assign(expression::tobytes(casted), index * dsizeof(dtype_));
                    ++index;
                }
            }; 

            switch (dtype_) {
                case int8:    fill(int8_t{});    break;
                case int16:   fill(int16_t{});   break;
                case int32:   fill(int32_t{});   break;
                case int64:   fill(int64_t{});   break;
                case float16:  fill(float16_t{}) ; break;   
                case bfloat16: fill(bfloat16_t{}); break;   
                case float32: fill(float{});     break;
                case float64: fill(double{});    break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        } 
    }
 
    
    template<Arithmetic T>
    void initialize(std::initializer_list<std::initializer_list<T>> const & values, Environment environment = Host{}) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors"); 
 
        if (rank() != 2 || shape_[0] != values.size() || shape_[1] != values.begin()->size())
            throw Exception("Shape mismatch in assignment from nested initializer_list"); 

        if (!is_initialized()) 
            initialize(environment);

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
                        Cast casted(value);
                        assign(expression::tobytes(casted), index * dsizeof(dtype_));
                        ++index;
                    }
                }
            };

            switch (dtype_) {
                case int8:    fill(int8_t{});    break;
                case int16:   fill(int16_t{});   break;
                case int32:   fill(int32_t{});   break;
                case int64:   fill(int64_t{});   break;
                case float16:  fill(float16_t{}) ; break;   
                case bfloat16: fill(bfloat16_t{}); break;   
                case float32: fill(float{});     break;
                case float64: fill(double{});    break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        }  
    }
 
    template<Arithmetic T>
    void initialize(std::initializer_list<std::initializer_list<std::initializer_list<T>>> const& values, Environment environment = Host{}) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");

        if (rank() != 3 
            || shape_[0] != values.size() 
            || shape_[1] != values.begin()->size() 
            || shape_[2] != values.begin()->begin()->size())
            throw Exception("Shape mismatch in assignment from triple-nested initializer_list");

        if (!is_initialized()) 
            initialize(environment);

        if (dtype_ == boolean) {
            std::ptrdiff_t index = 0;
            for (auto const& matrix : values) {
                if (matrix.size() != shape_[1])
                    throw Exception("Matrix row count mismatch");
                for (auto const& row : matrix) {
                    if (row.size() != shape_[2])
                        throw Exception("Row length mismatch");
                    for (auto const& value : row) {
                        assign((bool const*)(&value), index);
                        ++index;
                    }
                }
            }
        }

        else { 
            auto fill = [this, &values](auto cast) { 
                using Cast = decltype(cast);
                size_t index = 0;
                for (auto const& matrix : values) {
                    if (matrix.size() != shape_[1])
                        throw Exception("Matrix row count mismatch");
                    for (auto const& row : matrix) {
                        if (row.size() != shape_[2])
                            throw Exception("Row length mismatch");
                        for (auto value : row) { 
                            Cast casted(value);
                            assign(expression::tobytes(casted), index * dsizeof(dtype_));
                            ++index;
                        }
                    }
                }
            };

            switch (dtype_) {
                case int8:    fill(int8_t{});    break;
                case int16:   fill(int16_t{});   break;
                case int32:   fill(int32_t{});   break;
                case int64:   fill(int64_t{});   break;
                case float16:  fill(float16_t{}) ; break;   
                case bfloat16: fill(bfloat16_t{}); break;   
                case float32: fill(float{});     break;
                case float64: fill(double{});    break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        } 
} 

    template<Arithmetic T>
    void initialize(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> values, Environment environment = Host{}) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");

        if (rank() != 4 
            || shape_[0] != values.size() 
            || shape_[1] != values.begin()->size() 
            || shape_[2] != values.begin()->begin()->size() 
            || shape_[3] != values.begin()->begin()->begin()->size())
            throw Exception("Shape mismatch in assignment from quadruple-nested initializer_list");

        if (!is_initialized()) 
            initialize(environment);

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
                                Cast casted(value);
                                assign(expression::tobytes(casted), index * dsizeof(dtype_));
                                ++index;
                            }
                        }
                    }
                }
            };

            switch (dtype_) {
                case int8:    fill(int8_t{});    break;
                case int16:   fill(int16_t{});   break;
                case int32:   fill(int32_t{});   break;
                case int64:   fill(int64_t{});   break;
                case float16:  fill(float16_t{}) ; break;   
                case bfloat16: fill(bfloat16_t{}); break;   
                case float32: fill(float{});     break;
                case float64: fill(double{});    break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        }  
    } 


    template<Arithmetic T>
    void initialize(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>> values, Environment environment = Host{}) {
        if (!is_contiguous())
            throw Exception("Assign to initializer list supported only for contiguous tensors");

        if (rank() != 5 
            || shape_[0] != values.size() 
            || shape_[1] != values.begin()->size() 
            || shape_[2] != values.begin()->begin()->size() 
            || shape_[3] != values.begin()->begin()->begin()->size()
            || shape_[4] != values.begin()->begin()->begin()->begin()->size())
            throw Exception("Shape mismatch in assignment from quintuple-nested initializer_list");

        if (!is_initialized()) 
            initialize(environment);

        if (dtype_ == boolean) {
            std::ptrdiff_t index = 0;
            for (auto const& tensor4D : values) {
                if (tensor4D.size() != shape_[1])
                    throw Exception("4D tensor count mismatch");

                for (auto const& tensor3D : tensor4D) {
                    if (tensor3D.size() != shape_[2])
                        throw Exception("3D tensor count mismatch");

                    for (auto const& matrix : tensor3D) {
                        if (matrix.size() != shape_[3])
                            throw Exception("Matrix row count mismatch");

                        for (auto const& row : matrix) {
                            if (row.size() != shape_[4])
                                throw Exception("Row length mismatch");

                            for (auto const& value : row) {
                                assign((bool const*)(&value), index);
                                ++index;
                            }
                        }
                    }
                }
            } 
        }  
        else { 
            auto fill = [this, &values](auto cast) { 
                using Cast = decltype(cast);
                size_t index = 0;
                for (auto const& tensor4D : values) {
                    if (tensor4D.size() != shape_[1])
                        throw Exception("4D tensor count mismatch");

                    for (auto const& tensor3D : tensor4D) {
                        if (tensor3D.size() != shape_[2])
                            throw Exception("3D tensor count mismatch");

                        for (auto const& matrix : tensor3D) {
                            if (matrix.size() != shape_[3]) 
                                throw Exception("Matrix row count mismatch");

                            for (auto const& row : matrix) {
                                if (row.size() != shape_[4])
                                    throw Exception("Row length mismatch");

                                for (auto value : row) {
                                    Cast casted(value);
                                    assign(expression::tobytes(casted), index * dsizeof(dtype_));
                                    ++index;
                                }
                            }
                        }
                    }
                }
            };

            switch (dtype_) {
                case int8:    fill(int8_t{});    break;
                case int16:   fill(int16_t{});   break;
                case int32:   fill(int32_t{});   break;
                case int64:   fill(int64_t{});   break;
                case float16:  fill(float16_t{}); break;   
                case bfloat16: fill(bfloat16_t{}); break;   
                case float32: fill(float{});     break;
                case float64: fill(double{});    break;
                default: throw Exception("Unsupported dtype in assignment");
            } 
        }  
    }

    
public: 

    template<Integral Index>
    auto operator[](Index index) const {   
        if (!is_initialized())
            initialize(); 
        return expression::Slice<Tensor, Index>(*this, std::make_tuple(index));
    }

    auto operator[](indexing::Range range) const { 
        if (!is_initialized())
            initialize(); 
        return expression::Slice<Tensor, indexing::Range>(*this, std::make_tuple(range));
    } 

    template<class ... Indexes>
    auto operator[](Indexes... indexes) const {
        if (!is_initialized())
            initialize(); 
        return expression::Slice<Tensor, Indexes...>(*this, std::make_tuple(indexes...));
    } 

    auto transpose(int first = -1, int second = -2) const {
        if (!is_initialized())
            initialize(); 
        return expression::Transpose<Tensor>(*this, std::make_pair<int, int>(std::move(first), std::move(second)));
    }  
 
    template<Integral ... Indexes>
    auto permute(Indexes... indexes) const {
        if (!is_initialized())
            initialize(); 
        return expression::Permutation<Tensor>(*this, indexes...);
    } 

    template<Integral ... Sizes>
    auto view(Sizes... sizes) const {
        if (!is_initialized())
            initialize(); 
        return expression::Reshape<Tensor>(*this, sizes...);
    } 

    auto squeeze() const {
        if (!is_initialized())
            initialize();
        return expression::Squeeze<Tensor>(*this);
    }
 
    template<Integral... Axes>
    auto unsqueeze(Axes... axes) {
        return expression::Unsqueeze<Tensor>(*this, axes...);
    }
    
public:

    Tensor(type dtype, Shape shape, std::ptrdiff_t offset, std::shared_ptr<Buffer> storage)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape)
    ,   offset_(offset)   
    ,   buffer_(std::move(storage)) 
    {
        nelements_ = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<>{}); 
        node_ = std::make_shared<Node>(*this);
    }

    Tensor(type dtype, Shape shape, Strides strides, std::ptrdiff_t offset, std::shared_ptr<Buffer> storage)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides)
    ,   offset_(offset)   
    ,   buffer_(std::move(storage)) 
    {  
        std::size_t expected = 1;
        for (std::size_t dimension = 0; dimension < rank(); ++dimension) { 
            std::size_t index = rank() - 1 - dimension;   // walk backward logically
            if (strides_[index] != expected) {
                is_contiguous_ = false;
            }
            nelements_ *= shape_[index];
            expected *= shape_[index];
        }   
        node_ = std::make_shared<Node>(*this);
    }

protected:    
    template <Composable Source, class... Indexes> 
    friend class expression::Slice;

    template <Composable Source> 
    friend class expression::Transpose;

    template <Composable Source>
    friend class expression::Reshape; 

    template <Composable Source>
    friend class expression::Squeeze; 

    template <Composable Source>
    friend class expression::Unsqueeze;

    template <Composable Source> 
    friend class expression::Permutation; 

    template <Composable Source>
    friend class expression::Expansion;
 
    template <Composable Source>
    friend class expression::Flatten;

    template <class Coordinates, Composable... Sources>
    friend class expression::Complexification;

    template <Composable Source>
    friend class expression::Realification;

    void assign(std::byte const*, std::ptrdiff_t); 
    void assign(bool const*, std::ptrdiff_t); 

    bool compare(std::byte const*, std::ptrdiff_t) const; 
    void copy(Tensor const& other);
    
public:
    uintptr_t id() const {
        return node_->id;
    } 

private: 
    type dtype_ = any;
    Shape shape_{}; 
    Strides strides_{}; 
    std::size_t nelements_ = 1;
    std::ptrdiff_t offset_ = 0;    
    mutable std::shared_ptr<Buffer> buffer_ = nullptr;
    mutable std::shared_ptr<Node> node_ = nullptr;
    bool is_contiguous_ = true;
};    

enum class IOStyle {
    Tannic,
    PyTorch
}; 

static IOStyle iostyle = IOStyle::PyTorch;
inline void setiostyle(IOStyle style) {
    iostyle = style;
}

std::ostream& operator<<(std::ostream& ostream, Tensor const& tensor);  

template<Composable Source> 
inline std::ostream& operator<<(std::ostream& ostream, Source source) {
    Context context{};
    Tensor tensor = source.forward(context);  
    ostream << tensor;
    return ostream;
} 

template<class Operation, Composable Operand>
Tensor expression::Unary<Operation, Operand>::forward(Context const& context) const { 
    Tensor result(dtype(), shape(), strides(), offset());  
    this->operation.forward(std::get<0>(this->operands), result);
    return result;
}

template<Operator Operation, Composable Operand, Composable Cooperand>
Tensor expression::Binary<Operation, Operand, Cooperand>::forward(Context const& context) const {
    Tensor result(dtype(), shape(), strides());
    this->operation.forward(std::get<0>(this->operands), std::get<1>(this->operands), result);
    return result;
}  


template<Operator Operation, Composable Operand>
Tensor expression::Binary<Operation, Operand, Scalar>::forward(Context const& context) const {
    Tensor result(dtype(), shape(), strides());
    this->operation.forward(std::get<0>(this->operands), std::get<1>(this->operands), result);
    return result;
}  

template<Composable Source>
Tensor expression::Reshape<Source>::forward(Context const& context) const { 
    Tensor source = source_.forward(context);
    return Tensor(this->dtype(), shape(), strides(), offset(), source.buffer_);
}
 
template<Composable Source>
Tensor expression::Squeeze<Source>::forward(Context const& context) const { 
    Tensor source = source_.forward(context);
    return Tensor(this->dtype(), shape(), strides(), offset(), source.buffer_);
}
 
template<Composable Source>
Tensor expression::Unsqueeze<Source>::forward(Context const& context) const { 
    Tensor source = source_.forward(context);
    return Tensor(this->dtype(), shape(), strides(), offset(), source.buffer_);
}
 
template<Composable Source>
Tensor expression::Flatten<Source>::forward(Context const& context) const { 
    Tensor source = source_.forward(context);
    return Tensor(this->dtype(), shape(), strides(), offset(), source.buffer_);
} 

template<Composable Source>
Tensor expression::Permutation<Source>::forward(Context const& context) const {   
    Tensor source = source_.forward(context);
    return Tensor(this->dtype(), shape(), strides(), offset(), source.buffer_);
}         
 
template<Composable Source>
Tensor expression::Transpose<Source>::forward(Context const& context) const { 
    Tensor source = source_.forward(context); 
    return Tensor(this->dtype(), shape(), strides(), offset(), source.buffer_);
}   

template<Composable Source>
Tensor expression::Expansion<Source>::forward(Context const& context) const { 
    Tensor source = source_.forward(context);
    return Tensor(this->dtype(), shape(), strides(), offset(), source.buffer_);
}
 
template<Composable Source, class... Indexes>
Tensor expression::Slice<Source, Indexes...>::forward(Context const& context) const {   
    Tensor source = source_.forward(context);
    return Tensor(this->dtype(), shape(), strides(), offset(), source.buffer_);
}         

template<Composable Source, class... Indexes>
void expression::Slice<Source, Indexes...>::assign(std::byte const* value, std::ptrdiff_t offset) { 
    if constexpr (Trait<Source>::is_assignable) {
        source_.assign(value, offset); 
    } 

    else {
        Context context{};
        Tensor tensor = this->forward(context);
        tensor.assign(value, offset);
    }
}  

template<Composable Source, class... Indexes>
void expression::Slice<Source, Indexes...>::assign(bool const* value, std::ptrdiff_t offset) { 
    if constexpr (Trait<Source>::is_assignable) {
        source_.assign(value, offset); 
    } 

    else {
        Context context{};
        Tensor tensor = this->forward(context);
        tensor.assign(value, offset);
    }
}   

template <Composable Source, class... Indexes>
template <Composable Expression>
void expression::Slice<Source, Indexes...>::operator=(Expression expression) { 
    if(this->shape() != expression.shape()) {
        throw Exception("Cannot copy tensors with different shapes");
    }
    Context context{};
    Tensor target = this->forward(context); 
    target.copy(expression.forward(context));
} 

template<Composable Source, class... Indexes>
bool expression::Slice<Source, Indexes...>::compare(std::byte const* value, std::ptrdiff_t offset) const { 
    if constexpr (Trait<Source>::is_comparable) {
        return source_.compare(value, offset); 
    } 

    else {
        Context constext{};
        Tensor tensor = forward(constext);
        return tensor.compare(value, offset);
    }
}  
 
template<class Coordinates, Composable Source>
Tensor expression::Complexification<Coordinates, Source>::forward(Context const& context) const {
    Tensor real = source.forward(context);
    Tensor complex(dtype(), shape(), strides(), offset(), real.buffer_); 
    return complex; 
} 

template<class Coordinates, Composable Real, Composable Imaginary>
Tensor expression::Complexification<Coordinates, Real, Imaginary>::forward(Context const& context) const {  
    Tensor result(dtype(), shape(), strides(), offset());
    Coordinates::forward(real.forward(context), imaginary.forward(context), result);
    return result;
} 

template<Composable Source>
Tensor expression::Realification<Source>::forward(Context const& context) const {
    Tensor complex = source.forward(context);
    Tensor real(dtype(), shape(), strides(), offset(), complex.buffer_); 
    return real; 
}

} // namespace tannic

#endif // TENSOR_HPP