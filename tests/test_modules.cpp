#include <gtest/gtest.h>
#include "IO/Serialization.hpp"
#include "IO/Persistence.hpp"
#include "Tensor.hpp" 
#include "Modules.hpp"
#include "Algebra/Transformations.hpp"

struct Linear : public Module<Linear>{
    Tensor weight;

    Linear(int input_features, int output_features, type dtype)
    :   weight({input_features, output_features}, dtype) {}

    Linear(Tensor weight) : weight(weight) {}

    auto forward(Tensor input) const {
        return linear(input, weight);
    }
};

TEST(Test, Constructors) {
    Tensor weight{{2,3}, float32};
    Linear module{weight};
    Tensor input({1,2,3}, float32);
    auto expression = module(input);

    List<Linear> modules = {
        Linear(784, 512, float32),
        Linear(512, 256, float32)
    };
    modules.add(Linear(256, 10, float32)); 
}


TEST(Test, Embedding) {
    Embedding emmbeding(4, 4, float32, integer16); 
    Tensor sequence = emmbeding(1, 3);
    ASSERT_EQ(sequence.shape(), Shape(2, 4)); 
}