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