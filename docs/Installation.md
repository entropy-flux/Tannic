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