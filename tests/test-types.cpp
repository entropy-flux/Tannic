#include <gtest/gtest.h>
#include <sstream>
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