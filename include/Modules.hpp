// Copyright 2025 Eric Cardozo
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

#ifndef MODULE_HPP
#define MODULE_HPP

#include <iostream>
#include <utility> 
#include <list>
#include <functional>

template<class Derived>
struct Module {
public:
    template<typename... Operands>
    auto operator()(Operands&&... operands) const -> decltype(auto) {
        return static_cast<const Derived*>(this)->forward(std::forward<Operands>(operands)...);
    }
};
 
template <class Module>
class List { 
public:
    List() = default;

    List(std::initializer_list<Module> list)
    :   modules_(list) {}

    template<typename Iterator>
    List(Iterator begin, Iterator end) 
    :   modules_(begin, end) {}
 
    void add(Module const& module) {
        modules_.push_back(module);
    }
 
private:
    std::list<Module> modules_{};
};


class Embedding : public Module<Embedding> {
    public: 
    template<typename Lenght, typename Dimension, class Allocator = Host>
    Embedding(type itype, type dtype, Lenght lenght, Dimension dimension, Allocator allocator = Allocator{})
    :   itype_(itype)
    ,   dtype_(dtype)
    ,   shape_(lenght, dimension)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
    {}

    template<class... Indexes>
    Tensor forward(Indexes... indexes) const {   
        Tensor lookup(itype_, Shape(sizeof...(indexes)), storage_.allocator());
        Tensor result(dtype_, Shape(sizeof...(indexes), shape_.back()), storage_.allocator()); 
        forward(result, lookup);
        return result;
    }

    type dtype() const { return dtype_; }
    type itype() const { return itype_; }
    std::byte* address() { return static_cast<std::byte*>(storage_.address()); } 
    std::byte const* address() const { return static_cast<std::byte const*>(storage_.address()); } 

    protected:
    void forward(Tensor& result, Tensor const& lookup) const;

    private:
    type dtype_;
    type itype_;
    Shape shape_;
    Storage storage_;
};

#endif  // MODULE_HPP