# Tannic 
 
## Introduction

While exploring the most recent models, I began noticing some weird patterns, CUDA kernels hardcoded as strings, pointers, and constexpr hacks embedded in Python sublanguages. I’m not saying this approach is inherently bad, but I couldn’t shake the feeling that it would be far cleaner and more maintainable to rewrite everything directly in C++ using those features directly.

On the other hand, many existing C++ frameworks, while fully native, are low-level and hard to use or extend. They often force developers to manage complex memory layouts or backend-specific details using macros, which makes adding new operations or integrating new hardware backends cumbersome.

This insight led me to create Tannic, a lightweight, fully C++ tensor library designed from the ground up for clarity, composability, and extensibility. It maintains a Python-like API feel, so developers can enjoy familiar, intuitive syntax while working entirely in C++. The library is designed to be easy to adopt, easy to extend, and consistent in its behavior—even as new operations, data types, or backends are added.

## What is Tannic?

**Tannic** is an extensible C++ tensor library built around a host–device execution model.
Unlike monolithic frameworks, it provides only a minimal set of built‑in operators, focusing on a flexible architecture where new operations, data types, and backends can be added easily.
This approach keeps the library lightweight while enabling adaptation to a wide range of computational needs.

This library is designed to serve as the foundational core for a neural network inference framework, but is equally suited to other domains such as classical ML or physics simulations—all without requiring Python. 

Below is a minimal example demonstrating tensor creation, initialization, basic indexing, and arithmetic operations with Tannic:

```cpp
#include <iostream>
#include <tannic.hpp>

using namespace tannic;

int main() { 
    Tensor X(float32, {2,2}); // and X.initialize(Device()) for CUDA support
    
    X[0, range{0,-1}] = 1;  
    X[1,0] = 3;             
    X[1,1] = 4;           
    
    Tensor Y(float32, {1,2}); 
    Y[0,0] = 4;                            
    Y[0,1] = 6;    
    
    Y = log(X) + Y * Y - exp(X) + matmul(X, Y.transpose()); // assign expressions dynamically like in python
    std::cout << Y; 
}
```

It will output: 

```
Tensor([[23.2817, 43.2817], 
        [33.0131, 18.7881]] dtype=float32, shape=(2, 2))
```

Equivalent PyTorch code for comparison:

```python
import torch
 
X = torch.zeros((2, 2), dtype=torch.float32)
 
X[0, 0:] = 1       
X[1, 0] = 3
X[1, 1] = 4
 
Y = torch.zeros((1, 2), dtype=torch.float32)
 
Y[0, 0] = 4     
Y[0, 1] = 6       
Y = torch.log(X) + Y * Y - torch.exp(X) + torch.matmul(X, Y.t())
print(Y) 
```  

Giving:

```
tensor([[23.2817, 43.2817],
        [33.0131, 18.7881]])
```

## Status

Note: Tannic is currently in an early development stage. It is functional but not fully optimized, and some features may still have bugs. The C backend API—used to extend the library—is under active development and may change significantly. The public API described in the documentation is mostly stable, with only minor breaking changes expected as the library evolves.

While the library is currently written in C++23, the arrival of C++26, is shaping up to be a monumental- too significant to ignore. At some point, it may be requirement for Tannic. 


## Features

- Dynamic typing: Flexible tensor data types that support runtime type specification, enabling features like easy tensor serialization and deserialization, but that also support compile time specifications thanks to constexpr. 

- Constexpr templated expressions: This allows custom kernel fusion strategies using SFINAE and compile time assertions and shape calculations.

- Broadcasting: NumPy‑style automatic shape expansion in arithmetic operations, enabling intuitive and efficient tensor computations across dimensions.

- Advanced indexing and slicing: Intuitive multi-dimensional tensor access and manipulation.

- Host–Device execution model: Unified support for CPU and CUDA-enabled GPU computation within the same codebase. While the device backend is currently developed in CUDA, the design is not tied to it and can support other backends in the future.

- Minimal core operators: Only essential built-in operations to keep the library lightweight and extensible. 


## What is comming...
 

- **cuBlas and cuTensor optional support**: This may be added soon to accelerate tensor computations.

- **Autograd**: Autograd is not necessary for inference, so it will be added to the library later when the runtime api is optimized and mature.

- **Graph mode**: A constexpr graph mode will be added to the library, possibly with the arrival of C++26.

- **Quantization support**: The library will support necessary dtypes to create quantized neural networks like bitnet.

- **Additional backends**: Expansion beyond CUDA to support other device backends is planned. Host-Device computational model can be used as well with other hardware vendors.

- **Multi GPU support**. Unfortunately I don't have either the expertise or the resources to add multigpu support, but the whole library was build taking this in mind so it won't be a breaking change when added.   