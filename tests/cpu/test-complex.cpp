#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>

#include "tensor.hpp"
#include "complex.hpp" 
#include "comparisons.hpp"

using namespace tannic;

class ComplexTests : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(ComplexTests, ComplexificationAndRealification) {
    Tensor X(float32, {2,2}); X.initialize();  
    X[0][0] = 1;
    X[0][1] = 0;
    X[1][0] = 2;
    X[1][1] = 3;  
    Tensor Z = complexify(X);

    float* data = reinterpret_cast<float*>(Z.bytes());
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 0);
    ASSERT_EQ(data[2], 2);
    ASSERT_EQ(data[3], 3);
    
    Tensor Y = realify(Z);


    ASSERT_EQ(X[0][0], 1);
    ASSERT_EQ(X[0][1], 0); 
    ASSERT_EQ(X[1][0], 2); 
    ASSERT_EQ(X[1][1], 3); 
}  

TEST_F(ComplexTests, HostPolarConversionManual) {  
    Tensor X(float32, {2, 2});
    X.initialize(); 

    X[0][0] = 1.0f;  X[0][1] = 6.0f;
    X[1][0] = 2.0f;  X[1][1] = 3.0f;
 
    Tensor Y(float32, {2, 2});
    Y.initialize();

    Y[0][0] = 2.0f;  Y[0][1] = 1.0f;
    Y[1][0] = 1.5f;  Y[1][1] = 3.14f;
 
    Tensor Z = polar(X, Y);  
    Tensor W = realify(Z);  

    Tensor expected(float32, {2, 2, 2}); expected.initialize({
        {{-0.416147, 0.909297}, {3.24181, 5.04883}},
        {{0.141474, 1.99499}, {-3.0, 0.00477764}}
    });
 
    EXPECT_TRUE(allclose(W, expected));
}

Tensor freq_cis(
    type dtype,
    size_t model_dimension,
    size_t sequence_length_limit,
    double theta = 10000.0
) {
        auto scale = std::log(theta) / static_cast<double>(model_dimension);
        Tensor rho = ones(dtype, {sequence_length_limit, model_dimension / 2});
        Tensor phi(dtype, {sequence_length_limit, model_dimension / 2}); 
        for(auto position = 0; position < sequence_length_limit; position++) {
            for(auto dimension = 0; dimension < model_dimension / 2; dimension++) { 
                phi[position, dimension] = position * std::exp(-2 * dimension * scale); 
            }
        } 
        return polar(rho, phi);
}

Tensor embed_freqs(Tensor sequence, Tensor frequencies) {
    size_t batch_size = sequence.size(0);
    size_t number_of_heads = sequence.size(1);
    size_t sequence_length = sequence.size(2);
    size_t heads_dimension = sequence.size(3);
    sequence = sequence.view(batch_size, number_of_heads, sequence_length, heads_dimension / 2, 2);
    sequence = complexify(sequence);
    sequence = sequence * frequencies;
    sequence = realify(sequence);
    return sequence.view(batch_size, number_of_heads, sequence_length, heads_dimension);
}

TEST_F(ComplexTests, LlamaFreqCis) { 
    Tensor X_expected(float64, {8, 3, 2}); X_expected.initialize({
        { {1.0000f, 0.0000f}, {1.0000f, 0.0000f}, {1.0000f, 0.0000f} },
        { {0.5403f, 0.8415f}, {0.9989f, 0.0464f}, {1.0000f, 0.0022f} },
        { {-0.4161f, 0.9093f}, {0.9957f, 0.0927f}, {1.0000f, 0.0043f} },
        { {-0.9900f, 0.1411f}, {0.9903f, 0.1388f}, {1.0000f, 0.0065f} },
        { {-0.6536f, -0.7568f}, {0.9828f, 0.1846f}, {1.0000f, 0.0086f} },
        { {0.2837f, -0.9589f}, {0.9732f, 0.2300f}, {0.9999f, 0.0108f} },
        { {0.9602f, -0.2794f}, {0.9615f, 0.2749f}, {0.9999f, 0.0129f} },
        { {0.7539f, 0.6570f}, {0.9477f, 0.3192f}, {0.9999f, 0.0151f} }
    }); 


    Tensor X = freq_cis(float64, 6, 8);  
    EXPECT_TRUE(allclose(X_expected, realify(X), 0.1, 0.1));
}

TEST_F(ComplexTests, ComplexifyTest) {  
    Tensor freqs(float32, {8, 3, 2}); 
    freqs.initialize({
        { {1.0000f, 0.0000f}, {1.0000f, 0.0000f}, {1.0000f, 0.0000f} },
        { {0.5403f, 0.8415f}, {0.9989f, 0.0464f}, {1.0000f, 0.0022f} },
        { {-0.4161f, 0.9093f}, {0.9957f, 0.0927f}, {1.0000f, 0.0043f} },
        { {-0.9900f, 0.1411f}, {0.9903f, 0.1388f}, {1.0000f, 0.0065f} },
        { {-0.6536f, -0.7568f}, {0.9828f, 0.1846f}, {1.0000f, 0.0086f} },
        { {0.2837f, -0.9589f}, {0.9732f, 0.2300f}, {0.9999f, 0.0108f} },
        { {0.9602f, -0.2794f}, {0.9615f, 0.2749f}, {0.9999f, 0.0129f} },
        { {0.7539f, 0.6570f}, {0.9477f, 0.3192f}, {0.9999f, 0.0151f} }
    }); 
    freqs = complexify(freqs);
    freqs = unsqueeze(transpose(freqs, 0, 1), 0, 1);

    Tensor sequence(float32, {1, 1, 3, 16});
    sequence.initialize({
        {  
            { 
                {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                {17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0},
                {33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0}
            }
        }
    });

    Tensor output_expected(float32, {1, 1, 3, 16});
    output_expected.initialize({
        {   // batch=1
            {   // head=1
                {1.0, 2.0, -1.7451, 4.6857, -7.5363, 2.0499, -8.0588, -6.9323,
                 1.6856, -13.3472, 14.6275, -7.1435, 16.3942, 9.8106, 0.7965, 21.9174},
                {17.0, 18.0, 18.0511, 20.8596, 18.8703, 23.8521, 19.4457, 26.9596,
                 19.7704, 30.1678, 19.8364, 33.4596, 19.6365, 36.8171, 19.1643, 40.2216},
                {33.0, 34.0, 34.9208, 36.0770, 36.8366, 38.1591, 38.7400, 40.2535,
                 40.6388, 42.3526, 42.5205, 44.4600, 44.4021, 46.5759, 46.2705, 48.7049}
            }
        }
    });

    Tensor output = embed_freqs(sequence, freqs);
    EXPECT_TRUE(allclose(output, output_expected));
}

/*
Equivalent torch code:
import torch
from torch import Tensor
from torch import view_as_complex, view_as_real

def apply_frequencies(sequence: Tensor, frequencies: Tensor) -> Tensor:
    batch_size, number_of_heads, sequence_length, heads_dimension = sequence.shape 
    sequence = sequence.view(batch_size, number_of_heads, sequence_length, heads_dimension // 2, 2)
    sequence = view_as_complex(sequence)   
    sequence = sequence * frequencies   
    sequence = view_as_real(sequence)   
    return sequence.view(batch_size, number_of_heads, sequence_length, heads_dimension)


# Explicit sequence (batch=1, heads=1, seq_len=3, head_dim=16)
sequence = torch.tensor(
    [
        [
            [
                [1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
                [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0],
                [33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0,
                 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0]
            ]
        ]
    ]
)  # shape: (1, 1, 3, 16)


# Frequencies (your full 8x3 set, as real/imag pairs)
freq_data = [
    [[1.0000, 0.0000], [1.0000, 0.0000], [1.0000, 0.0000]],
    [[0.5403, 0.8415], [0.9989, 0.0464], [1.0000, 0.0022]],
    [[-0.4161, 0.9093], [0.9957, 0.0927], [1.0000, 0.0043]],
    [[-0.9900, 0.1411], [0.9903, 0.1388], [1.0000, 0.0065]],
    [[-0.6536, -0.7568], [0.9828, 0.1846], [1.0000, 0.0086]],
    [[0.2837, -0.9589], [0.9732, 0.2300], [0.9999, 0.0108]],
    [[0.9602, -0.2794], [0.9615, 0.2749], [0.9999, 0.0129]],
    [[0.7539, 0.6570], [0.9477, 0.3192], [0.9999, 0.0151]],
]

# Convert into complex tensor
frequencies = torch.tensor(freq_data, dtype=torch.float32)
frequencies = torch.view_as_complex(frequencies)   # shape (8, 3) 

frequencies = frequencies.permute(1, 0).unsqueeze(0).unsqueeze(0)  # (1,1,3,8)
 
output = apply_frequencies(sequence, frequencies)

print("Input shape:", sequence.shape)
print("Frequencies shape:", frequencies.shape)
print("Output shape:", output.shape)
print("Output tensor:\n", output)
*/