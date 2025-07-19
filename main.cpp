#include <iostream>
#include <utility> 
#include <list>
#include <functional> 

#include <tannic.hpp> 
#include <tannic/Serialization.hpp>

using namespace tannic;

struct Module {
    using size_type = std::size_t;
    template<typename Self, typename... Operands>
    auto operator()(this Self&& self, Operands&&... operands) -> decltype(auto) {
        return std::forward<Self>(self).forward(std::forward<Operands>(operands)...);
    }
};

class Parameter {
public: 
    using rank_type = uint8_t;
    using size_type = std::size_t;  

    Parameter(type dtype, Shape shape)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_)   
    {}

    Parameter(type dtype, Shape shape, Strides strides)
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(strides)   
    {}
 
    type dtype() const { 
        return dtype_; 
    }

    Shape const& shape() const { 
        return shape_; 
    }

    Strides const& strides() const { 
        return strides_; 
    }

    std::ptrdiff_t offset() const {
        return 0;
    }

    auto nbytes() const { 
        return shape_.size() * dsizeof(dtype_); 
    }
    
    auto rank() const { 
        return shape_.rank(); 
    }           

    void initialize(Allocator allocator) const {
        storage_ = std::make_shared<Storage>(nbytes(), allocator);
    }

    Tensor forward() const {
        return Tensor(dtype_, shape_, strides_, 0, storage_);
    }
 
private:
    type dtype_;
    Shape shape_; 
    Strides strides_;  
    mutable std::shared_ptr<Storage> storage_ = nullptr;
};  

struct Linear : Module {
    Parameter weight;
    Parameter bias;

    Linear(type dtype, size_type input_features, size_t output_features)
    :   weight(dtype, Shape(input_features, output_features))
    ,   bias(dtype, Shape(output_features)) {}

    void initialize(Allocator allocator = Host{}) {
        weight.initialize(allocator);
        bias.initialize(allocator);
    }

    Tensor forward(Tensor input) {
        return matmul(input, transpose(weight, -1, -2)) + bias;
    }
};

int main() {
    Linear linear(float32, 5, 5); linear.initialize(); 
    Tensor X(float32, {5,5}); X.initialize();
    Tensor Y = linear(X);
    std::cout << Y;
}