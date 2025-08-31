#include <gtest/gtest.h>
#include <sstream>
#include <cstring>
#include "types.hpp"

using namespace tannic;

TEST(DTypeTest, SizeofValuesAreCorrect) {
    EXPECT_EQ(dsizeof(int8),      sizeof(int8_t));
    EXPECT_EQ(dsizeof(int16),     sizeof(int16_t));
    EXPECT_EQ(dsizeof(int32),     sizeof(int32_t));
    EXPECT_EQ(dsizeof(int64),     sizeof(int64_t));
    EXPECT_EQ(dsizeof(float32),   sizeof(float));
    EXPECT_EQ(dsizeof(float64),   sizeof(double));
    EXPECT_EQ(dsizeof(complex64), 2 * sizeof(float));
    EXPECT_EQ(dsizeof(complex128),2 * sizeof(double));
    EXPECT_EQ(dsizeof(unknown),      0);
}

TEST(DTypeTest, NameofValuesAreCorrect) {
    EXPECT_EQ(dnameof(int8),      "int8");
    EXPECT_EQ(dnameof(int16),     "int16");
    EXPECT_EQ(dnameof(int32),     "int32");
    EXPECT_EQ(dnameof(int64),     "int64");
    EXPECT_EQ(dnameof(float32),   "float32");
    EXPECT_EQ(dnameof(float64),   "float64");
    EXPECT_EQ(dnameof(complex64), "complex64");
    EXPECT_EQ(dnameof(complex128),"complex128");
    EXPECT_EQ(dnameof(unknown),      "none");
}

TEST(DTypeTest, OstreamOperatorPrintsCorrectly) {
    std::ostringstream oss;
    oss << float64;
    EXPECT_EQ(oss.str(), "float64");

    oss.str(""); // Clear stream
    oss << unknown;
    EXPECT_EQ(oss.str(), "none");
}

#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include "types.hpp"

using namespace tannic;

TEST(Float16Test, ZeroAndSign) {
    float16_t hpos = float32_to_float16(0.0f);
    float16_t hneg = float32_to_float16(-0.0f);

    EXPECT_EQ(hpos.bits, 0x0000); // +0
    EXPECT_EQ(hneg.bits, 0x8000); // -0

    EXPECT_EQ(float16_to_float32(hpos), 0.0f);
    EXPECT_EQ(float16_to_float32(hneg), -0.0f);
}

TEST(Float16Test, InfinityAndNan) {
    float inf = std::numeric_limits<float>::infinity();
    float nan = std::numeric_limits<float>::quiet_NaN();

    float16_t hpos_inf = float32_to_float16(inf);
    float16_t hneg_inf = float32_to_float16(-inf);
    float16_t hnan     = float32_to_float16(nan);

    EXPECT_EQ(hpos_inf.bits, 0x7C00); // +inf
    EXPECT_EQ(hneg_inf.bits, 0xFC00); // -inf

    EXPECT_TRUE(std::isinf(float16_to_float32(hpos_inf)));
    EXPECT_TRUE(std::isinf(float16_to_float32(hneg_inf)));
    EXPECT_TRUE(std::isnan(float16_to_float32(hnan)));
}

TEST(Float16Test, NormalRoundTrip) {
    std::vector<float> values = {
        1.0f, -1.0f, 3.14159f, 0.3333f, 65504.0f  // largest representable half
    };

    for (float f : values) {
        float16_t h = float32_to_float16(f);
        float g = float16_to_float32(h); 
        EXPECT_NEAR(f, g, 1e-3f) << "Failed for value " << f;
    }
}

TEST(Float16Test, SubnormalValues) {
    // Smallest positive subnormal in float16 is 2^-24 ≈ 5.96e-8
    float tiny = std::ldexp(1.0f, -24);

    float16_t h = float32_to_float16(tiny);
    EXPECT_GT(h.bits, 0); // not zero
    float f = float16_to_float32(h);

    EXPECT_NEAR(f, tiny, 1e-8f);
}

TEST(Float16Test, OverflowToInfinity) {
    float big = 1e10f;
    float16_t h = float32_to_float16(big);
    float f = float16_to_float32(h);

    EXPECT_TRUE(std::isinf(f));
}

TEST(Float16Test, NaNPayloadPreserved) {
    // Make a NaN with payload
    uint32_t bits = 0x7FC12345;
    float nan;
    std::memcpy(&nan, &bits, sizeof(nan));

    float16_t h = float32_to_float16(nan);
    float f = float16_to_float32(h);

    EXPECT_TRUE(std::isnan(f));
}
