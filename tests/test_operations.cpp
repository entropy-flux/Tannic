#include <gtest/gtest.h> 
#include "Operations.hpp"
 

TEST(Test, BroadcastSingleShapeReturnsSameShape) {
    Shape shape{2, 3, 4};
    auto result = Operation::broadcast(shape);
    ASSERT_EQ(result, shape);
}

TEST(Test, BroadcastSameShapeReturnsSameShape) {
    Shape a{2, 3};
    Shape b{2, 3};
    auto result = Operation::broadcast(a, b);
    ASSERT_EQ(result, Shape(2, 3));
}

TEST(Test, BroadcastWithOnesExpandsCorrectly) {
    Shape a{1, 3};
    Shape b{2, 3};
    auto result = Operation::broadcast(a, b);
    ASSERT_EQ(result, Shape(2, 3));
}

TEST(Test, BroadcastScalarAndHigherDim) {
    Shape a{}; // scalar
    Shape b{4, 5};
    auto result = Operation::broadcast(a, b);
    ASSERT_EQ(result, Shape(4, 5));
}

TEST(Test, BroadcastIncompatibleShapesThrows) {
    Shape a{3, 2};
    Shape b{2, 3};
    EXPECT_THROW(Operation::broadcast(a, b), const char*);
}

// ----- Matmul Tests -----

TEST(Test, VectorDotProduct) {
    Shape a{4};
    Shape b{4};
    auto result = Matmul::broadcast(a, b);
    ASSERT_EQ(result, Shape(4));
}

TEST(Test, VectorMatrixProduct) {
    Shape a{3};
    Shape b{3, 2};
    auto result = Matmul::broadcast(a, b);
    ASSERT_EQ(result, Shape(2));
}

TEST(Test, MatrixVectorProduct) {
    Shape a{2, 3};
    Shape b{3};
    auto result = Matmul::broadcast(a, b);
    ASSERT_EQ(result, Shape(2));
}

TEST(Test, BatchedMatmul) {
    Shape a{5, 2, 3};
    Shape b{5, 3, 4};
    auto result = Matmul::broadcast(a, b);
    ASSERT_EQ(result, Shape(5, 2, 4));
}

TEST(Test, BroadcastedBatchesMatmul) {
    Shape a{1, 2, 3};
    Shape b{4, 3, 5};
    auto result = Matmul::broadcast(a, b);
    ASSERT_EQ(result, Shape(4, 2, 5));
}

TEST(Test, MatmulIncompatibleInnerDimensionsThrows) {
    Shape a{2, 3};
    Shape b{2, 2};
    EXPECT_THROW(Matmul::broadcast(a, b), std::invalid_argument);
}

TEST(Test, MatmulIncompatibleBatchesThrows) {
    Shape a{2, 2, 3};
    Shape b{3, 3, 4};
    EXPECT_THROW(Matmul::broadcast(a, b), const char*);
}