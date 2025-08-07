#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <cmath>

#include "Tensor.hpp"
#include "Complex.hpp" 

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
    Tensor Z = complex(X);

    float* data = reinterpret_cast<float*>(Z.bytes());
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 0);
    ASSERT_EQ(data[2], 2);
    ASSERT_EQ(data[3], 3);
    
    Tensor Y = real(Z);


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