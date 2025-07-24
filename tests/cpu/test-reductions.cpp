#include <gtest/gtest.h>
#include <vector>
#include <numeric>
 
#include "Tensor.hpp"   
#include "Reductions.hpp"

using namespace tannic;

TEST(TestReductions, TestArgmax) {
    Tensor X(float32, {7}); X.initialize();
    X[0] = 3; X[1] = 5; X[2] = 4; X[3] = 1; X[4] = 5; X[5] = 9; X[6] = 2;
    
    Tensor Z = argmax(X);
    
}

/*
import torch

def test_argmax():
    print("=== Testing argmax ===")

    # Test 1: Basic 1D tensor
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2])
    print("\nTest 1 - 1D tensor:")
    print("Input:", x)
    print("torch.argmax():", torch.argmax(x))  # Should be 5 (value 9)

    # Test 2: 2D tensor with dim=0
    x = torch.tensor([[1, 5, 3],
                      [4, 2, 6]])
    print("\nTest 2 - 2D tensor dim=0:")
    print("Input:\n", x)
    print("torch.argmax(dim=0):\n", torch.argmax(x, dim=0))  # Should be [1, 0, 1]

    # Test 3: 2D tensor with dim=1 and keepdim
    print("\nTest 3 - 2D tensor dim=1 keepdim=True:")
    print("Input:\n", x)
    print("torch.argmax(dim=1, keepdim=True):\n",
          torch.argmax(x, dim=1, keepdim=True))  # Should be [[1], [2]]

    # Test 4: 3D tensor with negative dim
    x = torch.rand(2, 3, 4)  # Random 3D tensor
    print("\nTest 4 - 3D tensor dim=-1:")
    print("Input shape:", x.shape)
    print("torch.argmax(dim=-1).shape:", torch.argmax(x, dim=-1).shape)  # Should be (2, 3)

def test_argmin():
    print("\n=== Testing argmin ===")

    # Test 1: Basic 1D tensor
    x = torch.tensor([3, 1, 4, 1, 5, 9, 2])
    print("\nTest 1 - 1D tensor:")
    print("Input:", x)
    print("torch.argmin():", torch.argmin(x))  # Should be 1 or 3 (value 1)

    # Test 2: 2D tensor with dim=1
    x = torch.tensor([[1, 5, 3],
                      [4, 2, 6]])
    print("\nTest 2 - 2D tensor dim=1:")
    print("Input:\n", x)
    print("torch.argmin(dim=1):\n", torch.argmin(x, dim=1))  # Should be [0, 1]

    # Test 3: Edge case - all equal values
    x = torch.ones(3, 3)
    print("\nTest 3 - All equal values:")
    print("Input:\n", x)
    print("torch.argmin(dim=0):", torch.argmin(x, dim=0))  # Should be [0, 0, 0] (first occurrence)

def test_edge_cases():
    print("\n=== Testing Edge Cases ===")

    # Test 1: Empty tensor
    x = torch.tensor([])
    print("\nTest 1 - Empty tensor:")
    try:
        print(torch.argmax(x))
    except Exception as e:
        print("Expected error:", e)

    # Test 2: Scalar tensor
    x = torch.tensor(5)
    print("\nTest 2 - Scalar tensor:")
    print("torch.argmax():", torch.argmax(x))  # Should be 0

    # Test 3: Large tensor
    x = torch.rand(1000, 1000)
    print("\nTest 3 - Large tensor:")
    print("torch.argmax().shape:", torch.argmax(x).shape)  # Should be ()

if __name__ == "__main__":
    test_argmax()
    test_argmin()
    test_edge_cases()
*/