# Tannic 
 
## Introduction

While exploring the most recent models, I began noticing some weird patterns, CUDA kernels hardcoded as strings, pointers, and constexpr hacks embedded in Python sublanguages. I’m not saying this approach is inherently bad, but I couldn’t shake the feeling that it would be far more sane to rewrite everything directly in C++ using those features directly.

On the other hand, many existing C++ frameworks, while fully native, are low-level and hard to use or extend. They often force developers to manage complex memory layouts or backend-specific details using macros, which makes adding new operations or integrating new hardware backends cumbersome.

This insight led me to create Tannic, a lightweight, fully C++ tensor library designed from the ground up for clarity, composability, and extensibility. It maintains a Python-like API feel, so developers can enjoy familiar, intuitive syntax while working entirely in C++. The library is designed to be easy to adopt, easy to extend, and consistent in its behavior—even as new operations, data types, or backends are added.

## What is Tannic?

**Tannic** is an extensible C++ tensor library built around a host–device execution model.
Unlike monolithic frameworks, it provides only a minimal set of built‑in operators, focusing on a flexible architecture where new operations, data types, and backends can be added easily.
This approach keeps the library lightweight while enabling adaptation to a wide range of computational needs.

This library is designed to serve as the foundational core for a neural network inference framework, but is equally suited to other domains such as classical ML or physics simulations—all without requiring Python. 

You can find full examples of neural networks inference with the Tannic framework here: 
- [CNN Server Example](https://github.com/entropy-flux/cnn-server-example) – demonstrates serving convolutional neural network models.  
- [ViT Server Example](https://github.com/entropy-flux/vit-server-example) – demonstrates serving Vision Transformer models.  


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

## Installation

This guide is currently in a “works on my machine” state. If you encounter any issues while building Tannic, your feedback is greatly appreciated, please open an issue or submit a pull request. Contributions to improve this guide are also very welcome!

### Requirements

- A C++23 compatible compiler.

- CMake 3.28+   

- (Optional) OpenBLAS: accelerates matrix multiplication  

- (Optional) CUDA Toolkit 12+: only required for GPU support 

Other optional requirements may be added in the future. Also the arrival of C++26, is shaping up to be a huge and too significant to ignore. At some point, it may be requirement for Tannic. 

### Clone the repository:

```bash
git clone https://github.com/entropy-flux/Tannic.git
cd Tannic 
```

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
cd build
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

These defaults provide a fast setup for development with the current state of the library. As Tannic evolves, CUDA configuration options and behavior may change.

## Usage

To use Tannic, simply include it in your project and interact with it similarly to a Python framework:

```cpp
#include <iostream>
#include <tannic.hpp>

using namespace tannic;

int main() { 
    Tensor X(float32, {2,2}); // and X.initialize(Device()) for CUDA support
    
    X[0, range{0,-1}] = 1;  
    X[1,0] = 3;             
    X[1,1] = 4;           
    
    Tensor Y(float64, {1,2}); 
    Y[0,0] = 4;                            
    Y[0,1] = 6;    
    
    Y = log(X) + Y * Y - exp(X) + matmul(X, Y.transpose()); // assign expressions dynamically like in python
    std::cout << Y;
    
    Tensor Z = {
        {1,2,3,4},
        {2,3,4,5}
    };                      // int tensor

    Tensor W = {
        {1.0f, 2.0f, 3.0f},
        {1.0f, 2.0f, 3.0f}
    };                      // float tensor

    std::cout << argmax(Z) << std::endl>>
}
``` 

Functions in Tannic do not immediately compute results. Instead, they build Expression objects, described in detail in the  [concepts](https://entropy-flux.github.io/Tannic/concepts.html) documentation. Basically an `Expression` is any class that follows the pattern:

```cpp
template<typename T>
concept Expression = requires(const T expression) {
    { expression.dtype()   } -> std::same_as<type>;
    { expression.shape()   } -> std::same_as<Shape const&>;
    { expression.strides() } -> std::same_as<Strides const&>;
      expression.offset();  // convertible to ptrdifft_t
      expression.forward(); // same as Tensor or Tensor const&
}; 
```

Any class that follows that pattern is a valid Tannic expression and can interact with other components of the library. All available expressions are detailed under the [class list](https://entropy-flux.github.io/Tannic/annotated.html) section. You can scroll to the members of each expression and find information about how dtypes are promoted, or how shapes are calculated. 

The library is built around the Host-Device computational model, so in order to use CUDA you just have to initialize kernels tensors on the Device you want to use, for example:

```cpp
int main() { 
    Tensor X(float32, {2,2}); X.initialize(Device());
    X[0, range{0,-1}] = 1;   // assignment just works the same on device
    X[1,0] = 3;             
    X[1,1] = 4;           
    

    Tensor Y(float32, {1,2}); Y.initialize(Device());
    ...

    Y = log(X) + Y * Y - exp(X) + matmul(X, Y.transpose()); // assign expressions dynamically like in python
    // Y is now calculated using CUDA.
    ...
}
``` 

The library currently lacks of some easily implementable CUDA features like copying a tensor from Host to Device and viceversa or printing CUDA tensors, I will add them soon.  
 
Data types are dynamic to make it easier to serialize and deserialize tensors at runtime, and deliver machine learning models that can work with arbitrary data types. They are represented using a C enum to be compatible with the C api runtime on wich the library relies on. 

```cpp
enum type { 
    none,
    int8,
    int16,
    int32,
    int64,
    float32,
    float64,
    complex64,   
    complex128,  
    TYPES
};
```

This design also paves the way for future features such as tensor quantization. 

## Contributing

Contributions to Tannic are welcome!
Whether you’re reporting bugs, proposing features, improving documentation, or optimizing kernels, your help is greatly appreciated.

## Ways to Contribute 

- **Write new test cases to cover code that is not yet tested**. I’m using a test-driven approach to build the library, but some edge cases may still be missing.

- **Refactor existing tests to use built-in features**. Many current tests manipulate tensor pointers directly to verify values because they were written before proper indexing and slicing functionality was implemented. This approach is tedious and can be simplified using tensor accessors. Take as example:

```cpp 
Tensor x(float32, {2,2}); x.initialize()
float* data = reinterpret_cast<float*>(x.bytes());
data[3] = 3;
ASSERT_EQ(data[3], 3);
```

can be refactored into: 

```cpp 
Tensor x(float32, {2,2}); x.initialize() 
x[1][1] = 3;
ASSERT_EQ(x[1][1], 3); // GTest support this but not  ASSERT_EQ(x[1,1], 3)
```

This is especially important in CUDA tests, where manually copying device memory sync to the host hurts test performance.

- **Improve installation documentation**. One of the main challenges with C++ adoption is the complexity of building and linking libraries. I plan to create a comprehensive guide on installing and integrating the library into other projects.

- **Optimize builds**. Currently there is a single CMakeLists.txt inside the cmake folder that compiles all the project. Decoupled builds for cpu and cuda backends will be a nice to have.

- **Optimize kernels**. Kernels are currently unoptimized since I'm still focusing on builing necessary features. The kernels can be found on .cpp and .cu files inside src/cpu and src/cuda files.

- **New features**. If you propose new features for the library, please ensure they align with the scope of a tensor library. For example, operations like tensor contractions would be a great addition, but machine learning components —such as neural network activation functions or attention mechanisms— are outside the scope.
(Don’t worry—I'm building a separate neural networks library on top of this one!) That said, I’m always open to fresh ideas, so don’t hesitate to share your suggestions.

## How to Contribute

Fork the repository and create a new branch for your feature or bug fix.
Example:

```bash
git checkout -b metal/metal-backend
```

Open a pull request describing:
- The purpose of your changes.
- Any relevant issues they address.
- Implementation details if needed.

Target branch: PRs for now should just target main till the library matures.


## Project structure

The project is organized into the following main components:

- Frontend (C++23) – Implemented in C++23 and distributed across multiple .hpp header files in include/tannic/. 

    * Implementations of non-constexpr and non templated functions are located in the src/ directory.

    * Some functions currently implemented in headers but not marked constexpr (e.g., certain member functions of the `Tensor` class) may become constexpr in the future.


- Backends – Contain platform-specific execution code:

    * src/cpu/ for the CPU backend. 

    * src/cuda/ for the CUDA backend.


- C Runtime Utilities – C utilities located in include/tannic/runtime/, used for building the C API required to extend the library, writting the backend and binding it to the C++ frontend.

    * All kernels must be implemented in terms of the C interfaces.

    * Vendor-specific constructs (e.g., streams, events, handles) must not be exposed in the C API. Instead, they should be abstracted using IDs or type erasure.
        
        * Example: `cudaStream_t` represents a computation queue, which is not specific to CUDA. In the C API, it is stored in a type-erased form:

        ```c
        struct stream_t { 
            uintptr_t address;
        };  
        ```

        Then if it was created as a cuda stream will be recovered as:

        ```cpp
        cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
        ```

    * C utilities should not be exposed in the C++ frontend, except for data types (dtypes) which are included for compatibility and convenience.

Currently what is in the src root folder is a mess, lot of code repetition and nasty helper functions with bunch of templates. I promise I will take the time to refactor this but after I find a way to dynamically add and remove nodes from a graph dynamically based on reference counting.  This refactor won't change anything on the public API. 


## Creating new features.

The C++ frontend is based on templated expressions, this means that when you write an expression, for example:

```cpp
auto expr = matmul(X, Y) + Z;
```

The result is not computed inmediatly, instead a templated expression is created with type:

```
Binary<Addition, Transformation<Composition, Tensor, Tensor>>, Tensor> // (matmul is actually a composition of tensors :)
```

This expression holds the the following methods:

- constexpr type dtype(): The resulting dtype of the expression, calculated using promotion tables or custom logic.

- constexpr Shape shape(): The resulting broadcasted shape of the expression. 

- constexpr Strides strides(): The resulting strides of the expression, calculated from the shape.

- ptrdiff_t offset(): The possition in bytes where the tensor starts in the current buffer. In this case 0 since a new tensor is created.

- Tensor forward(): A method that actually performs the computation using the already calculated metadata.

This allows you to:

- Create new symbols: All expressions that follows the concept (don't worry this is just like a python protocol):

    ```cpp
    template<typename T>
    concept Expression = requires(const T expression) {
        { expression.dtype()   } -> std::same_as<type>;
        { expression.shape()   } -> std::same_as<Shape const&>;
        { expression.strides() } -> std::same_as<Strides const&>;
        expression.offset();  // convertible to ptrdifft_t
        expression.forward(); // same as Tensor or Tensor const&
    }; 
    ```

    Will work with current `Tensor` class and other templated expressions in this library.


- Create new data structures: Again if your data structure follows the prior concept it can be used as well with the library, for example you can create `Scalar`, `Parameter`, `Sequence` classes and plug them into operations like if they were tensors, the resulting expression will be something like this:

```
Binary<Addition, Transformation<Composition, Parameter, Sequence>>, Scalar>
```

Finally the computation will be done when calling forward or when the expression is assigned to a non const tensor:

```cpp
Tensor W = matmul(X, Y) + Z;
```

This allows a python alike behavior since you can chain operations on the same variable like this:

```cpp
Tensor W = matmul(X, Y) + Z;
W = W * log(Z);
W = W * exp(Y) + X[1]; 
std::cout << W[1] + Z[0]; // IO supported for expressions.
``` 


### License & Attribution

Tannic is licensed under the Apache License 2.0, a permissive open-source license that allows you to use, modify, and distribute the code freely—even in commercial projects.

By contributing, you agree that your contributions will also be licensed under Apache 2.0 and that proper attribution is appreciated.

The only thing I ask in return is proper credit to the project and its contributors. Recognition helps the project grow and motivates continued development.