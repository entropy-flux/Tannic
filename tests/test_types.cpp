#include <gtest/gtest.h>
#include <cstddef>
#include "Types.hpp" 

class Test : public ::testing::Test {
protected:
    void SetUp() override {
        buffer = ::operator new(512);
    }

    void TearDown() override {
        ::operator delete(buffer);
    }

    void* buffer;
};


TEST_F(Test, TestTypes) { 
    const float f32_val = 3.14f;
    const double f64_val = 2.71828;
    const int8_t i8_val = -42;
    const int16_t i16_val = 1000;
    const int32_t i32_val = 123456;
  

    std::byte* offset = static_cast<std::byte*>(buffer);
     
    traits[float32].assign(offset, f32_val); offset += traits[float32].size;
    traits[float64].assign(offset, f64_val); offset += traits[float64].size;
    traits[integer8].assign(offset, i8_val); offset += traits[integer8].size;
    traits[integer16].assign(offset, i16_val); offset += traits[integer16].size;
    traits[integer32].assign(offset, i32_val); offset = static_cast<std::byte*>(buffer);
      
    auto retrieved_f32 = traits[float32].retrieve(offset); offset += traits[float32].size; 
    auto retrieved_f64 = traits[float64].retrieve(offset); offset += traits[float64].size; 
    auto retrieved_i8 = traits[integer8].retrieve(offset); offset += traits[integer8].size;  
    auto retrieved_i16 = traits[integer16].retrieve(offset); offset += traits[integer16].size; 
    auto retrieved_i32 = traits[integer32].retrieve(offset);  
    
    ASSERT_FLOAT_EQ(cast<float>(retrieved_f32), f32_val);
    ASSERT_DOUBLE_EQ(cast<double>(retrieved_f64), f64_val);
    ASSERT_EQ(cast<int8_t>(retrieved_i8), i8_val);
    ASSERT_EQ(cast<int16_t>(retrieved_i16), i16_val);
    ASSERT_EQ(cast<int32_t>(retrieved_i32), i32_val);

    
    offset = static_cast<std::byte*>(buffer);

    ASSERT_TRUE(traits[float32].compare(offset, f32_val)); offset += traits[float32].size;
    ASSERT_TRUE(traits[float64].compare(offset, f64_val)); offset += traits[float64].size;
    ASSERT_TRUE(traits[integer8].compare(offset, i8_val)); offset += traits[integer8].size;
    ASSERT_TRUE(traits[integer16].compare(offset, i16_val)); offset += traits[integer16].size;
    ASSERT_TRUE(traits[integer32].compare(offset, i32_val));
    
    offset = static_cast<std::byte*>(buffer);
 
    ASSERT_FALSE(traits[float32].compare(offset, 0.0f)); offset += traits[float32].size;
    ASSERT_FALSE(traits[float64].compare(offset, 0.0)); offset += traits[float64].size;
    ASSERT_FALSE(traits[integer8].compare(offset, static_cast<int8_t>(0))); offset += traits[integer8].size;
    ASSERT_FALSE(traits[integer16].compare(offset, static_cast<int16_t>(0))); offset += traits[integer16].size;
    ASSERT_FALSE(traits[integer32].compare(offset, static_cast<int32_t>(0)));
}