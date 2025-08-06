# Tannic 
 
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
    Tensor X(float32, {2,2}); X.initialize(); // or X.initialize(Device()) for CUDA support
    
    X[0, range{0,-1}] = 1;  
    X[1,0] = 3;             
    X[1,1] = 4;           
    
    Tensor Y(float32, {1,2}); Y.initialize();  // Explicit initialization required for now
    Y[0,0] = 4;                                // but may be removed in the future.
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


## Status

Note: Tannic is currently in an early development stage. It is functional but not fully optimized, and some features may still have bugs. The C backend API—used to extend the library—is under active development and may change significantly. The public API described in the documentation is mostly stable, with only minor breaking changes expected as the library evolves.

While the library is currently written in C++23, the arrival of C++26, is shaping up to be a monumental- too significant to ignore. At some point, it may be a hard requirement for Tannic. 


## Features

- Dynamic typing: Flexible tensor data types that support runtime type specification, enabling features like easy tensor serialization and deserialization.

- Broadcasting: NumPy‑style automatic shape expansion in arithmetic operations, enabling intuitive and efficient tensor computations across dimensions.

- Advanced indexing and slicing: Intuitive multi-dimensional tensor access and manipulation.

- Host–Device execution model: Unified support for CPU and CUDA-enabled GPU computation within the same codebase. While the device backend is currently developed in CUDA, the design is not tied to it and can support other backends in the future.

- Minimal core operators: Only essential built-in operations to keep the library lightweight and extensible. 


## What is comming...

- Autograd: Autograd is not necessary for inference, so it will be added to the library later when the runtime api is optimized and mature.

- Graph mode: A constexpr graph mode will be added to the library, possibly with the arrival of C++26.

- Quantization support: The library will support necessary dtypes to create quantized neural networks like bitnet.

- Additional backends: Expansion beyond CUDA to support other device backends is planned. Host-Device computational model can be used as well with other hardware vendors.

---

## Requirements

- C++23 compiler: A compiler with C++23 support is required to build and run Tannic. 

- OpenBLAS (optional): If installed on your system, OpenBLAS will accelerate matrix multiplication.

- CUDA Toolkit (optional): CUDA 12+ is required only if you want GPU support. If not installed, Tannic will build and run with CPU-only support.
 

## Installation

Clone the repository and include `tannic.hpp` and build it as follows:

### Debug build:
Use this for development — includes extra checks, assertions, 
and debug symbols for easier troubleshooting.

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
ctest --output-on-failure
``` 


### Release build
Use this for deployment or benchmarking — builds with full 
compiler optimizations and without debug checks.

```bash 
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) 
```

### Run the example
You can run the example provided in the main.cpp from the build folder:
```bash
./main
```
 
### Include Tannic in your project
```cpp
#include <tannic.hpp>
``` 

### CUDA support
CUDA support is enabled by default if a compatible CUDA toolkit (12+) is detected during configuration.
If no CUDA installation is found, Tannic will automatically fall back to a CPU‑only build.
You can explicitly disable CUDA with:

```
cmake .. -DTANNIC_ENABLE_CUDA=OFF
```
 

## Contributing

Contributions to Tannic are welcome! If you'd like to report bugs, request features, or submit pull requests, please follow these guidelines:

- Fork the repository and create a new branch for your feature or bugfix.

- Include tests and documentation for new features or bug fixes.

- Open a pull request describing your changes and their purpose. 

Tannic is licensed under the Apache License 2.0, a permissive open-source license that allows you to use, modify, and distribute the code freely—even in commercial projects.

The only thing I ask in return is proper credit to the project and its contributors. Recognition helps the project grow and motivates continued development.

By contributing, you agree that your contributions will also be licensed under Apache 2.0 and that proper attribution is appreciated.
 

## License

Tannic is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
 