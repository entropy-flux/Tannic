#include <tannic.hpp>
#include <string>
#include <string_view>

using namespace tannic;

class Parameter {
public:
    constexpr Parameter(type dtype, Shape shape)
    :   dtype_(dtype),
        shape_(shape),
        strides_(shape) 
    {}

    void initialize(Allocator allocator = Host{}) const;

    std::size_t key() const {
        return key_;
    }

    constexpr auto dtype() const {
        return dtype_;
    }

    constexpr Shape const& shape() const {
        return shape_;
    }

    constexpr Strides const& strides() const {
        return strides_;
    }

    constexpr std::ptrdiff_t offset() const {
        return 0;
    }

    constexpr auto nbytes() const {
        return shape_.size() * dsizeof(dtype_);
    }

    Tensor forward() const;

private:
    type dtype_;
    Shape shape_;
    Strides strides_;
    mutable std::size_t key_ = 0;
    mutable std::ptrdiff_t offset_ = 0;
};

class Parameters {
public:
    static Parameters& instance() {
        static Parameters instance;
        return instance;
    }

    std::size_t push(std::shared_ptr<Buffer> storage) {

        parameters_.push_back(storage);
        return parameters_.size() - 1;
    }

    std::shared_ptr<Buffer> get(std::size_t key) const {
        return parameters_[key];
    }

private:
    Parameters() = default;
    ~Parameters() = default;
    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete; 
    std::vector<std::shared_ptr<Buffer>> parameters_;
};  

void Parameter::initialize(Allocator allocator) const { 
    Parameters& parameters = Parameters::instance();
    key_ = parameters.push(std::make_shared<Buffer>(nbytes(), allocator));
}

Tensor Parameter::forward() const {
    Parameters& parameters = Parameters::instance();
    return Tensor(dtype_, shape_, strides_, 0, parameters.get(key_));
}


/* 
MLP(
  (input_layer): Linear(in_features=784, out_features=256, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
  (activation): ReLU()
  (output_layer): Linear(in_features=256, out_features=10, bias=True)
) 

struct Linear {
    Parameter weight;
    Parameter bias;
};

int main() {
    constexpr Linear input_layer {
        .weight = Parameter(float32, {784, 256}),
        .bias = Parameter(float32, {784})
    };

    constexpr Linear output_layer { 
        .weight = Parameter(float32, {256, 10}),
        .bias = Parameter(float32, {10})
    }
}
*/
 