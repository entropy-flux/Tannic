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