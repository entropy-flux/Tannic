#include <gtest/gtest.h>
#include "Serialization.hpp"
#include "Persistence.hpp"
#include "Tensor.hpp" 
#include "Modules.hpp"
#include "Transformations.hpp"

struct Linear : public Module<Linear>{
    Tensor weight;

    Linear(int input_features, int output_features, type dtype)
    :   weight(dtype, {input_features, output_features}) {}

    Linear(Tensor weight) : weight(weight) {}

    auto forward(Tensor input) const {
        return linear(input, weight);
    }
};

TEST(Test, Constructors) {
    Tensor weight{float32, {2,3}};
    Linear module{weight};
    Tensor input(float32, {1,2,3});
    auto expression = module(input);

    List<Linear> modules = {
        Linear(784, 512, float32),
        Linear(512, 256, float32)
    };
    modules.add(Linear(256, 10, float32)); 
}


TEST(Test, Embedding) {
    Embedding emmbeding(integer16, float32, 4, 4); 
    Tensor sequence = emmbeding(1, 3);
    ASSERT_EQ(sequence.shape(), Shape(2, 4)); 
}