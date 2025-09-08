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
    X.initialize(Host());

    X[0][0] = 1.0f;  X[0][1] = 6.0f;
    X[1][0] = 2.0f;  X[1][1] = 3.0f;
 
    Tensor Y(float32, {2, 2});
    Y.initialize(Host());

    Y[0][0] = 2.0f;  Y[0][1] = 1.0f;
    Y[1][0] = 1.5f;  Y[1][1] = 3.14f;
 
    Tensor Z = polar(X, Y);
 
    float* data = reinterpret_cast<float*>(Z.bytes());
 
    const float expected_real[4] = {-0.4161f, 3.2418f, 0.1415f, -3.0000f};
    const float expected_imag[4] = {0.9093f, 5.0488f, 1.9950f, 0.0047f};

    // Check each element (real and imaginary)
    for (int idx = 0; idx < 4; ++idx) {
        float real_val = data[2 * idx];      // real part at even indices
        float imag_val = data[2 * idx + 1];  // imag part at odd indices

        ASSERT_NEAR(real_val, expected_real[idx], 1e-3f) << "Real part mismatch at index " << idx;
        ASSERT_NEAR(imag_val, expected_imag[idx], 1e-3f) << "Imag part mismatch at index " << idx;
    }
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
    Tensor freqs(float64, {8, 3, 2}); 
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
    Tensor cplx = complexify(freqs);
    Tensor real = realify(cplx);
    EXPECT_TRUE(allclose(freqs, real));
}