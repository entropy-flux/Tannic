#include <gtest/gtest.h> 
#include "Operations.hpp"
#include "Transformations.hpp"

using namespace symbol;

TEST(Test, BroadcastSingleShapeReturnsSameShape) {
    Shape shape{2, 3, 4};
    auto result = Negation{}.broadcast(shape);
    ASSERT_EQ(result, shape);
}

TEST(Test, BroadcastSameShapeReturnsSameShape) {
    Shape a{2, 3};
    Shape b{2, 3};
    auto result = Addition{}.broadcast(a, b);
    ASSERT_EQ(result, Shape(2, 3));
}

TEST(Test, BroadcastWithOnesExpandsCorrectly) {
    Shape a{1, 3};
    Shape b{2, 3};
    auto result = Addition{}.broadcast(a, b);
    ASSERT_EQ(result, Shape(2, 3));
}

TEST(Test, BroadcastScalarAndHigherDim) {
    Shape a{}; // scalar
    Shape b{4, 5};
    auto result = Addition{}.broadcast(a, b);
    ASSERT_EQ(result, Shape(4, 5));
}

TEST(Test, BroadcastIncompatibleShapesThrows) {
    Shape a{3, 2};
    Shape b{2, 3};
    EXPECT_THROW(Addition{}.broadcast(a, b), std::invalid_argument);
}

// ----- linear Tests ----- 

TEST(Test, VectorMatrixProduct) {
    Shape a{3};
    Shape b{2, 3};
    auto result = Linear{}.broadcast(a, b);
    ASSERT_EQ(result, Shape(2));
}

TEST(Test, Batchedlinear) {
    Shape a{5, 2, 3};
    Shape b{5, 4, 3};
    auto result = Linear{}.broadcast(a, b);
    ASSERT_EQ(result, Shape(5, 2, 4));
}

TEST(Test, BroadcastedBatcheslinear) {
    Shape a{1, 2, 3};
    Shape b{4, 5, 3};
    auto result = Linear{}.broadcast(a, b);
    ASSERT_EQ(result, Shape(4, 2, 5));
}

TEST(Test, linearIncompatibleInnerDimensionsThrows) {
    Shape a{2, 3};
    Shape b{2, 2};
    EXPECT_THROW(Linear{}.broadcast(a, b), std::invalid_argument);
}

TEST(Test, linearIncompatibleBatchesThrows) {
    Shape a{2, 2, 3};
    Shape b{3, 3, 4};
    EXPECT_THROW(Linear{}.broadcast(a, b), std::invalid_argument);
}