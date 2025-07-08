#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace cuda {

inline void checkError(cudaError_t status, const char* message, const char* file, int line) {
    if (status != cudaSuccess) {
        std::ostringstream error;
        error << "CUDA Error at " << file << ":" << line << "\n"
              << "  Code: " << static_cast<int>(status) << " (" << cudaGetErrorName(status) << ")\n"
              << "  Message: " << cudaGetErrorString(status);
        if (message && *message)
            error << "\n  Context: " << message;
        throw std::runtime_error(error.str());
    }
}

} // namespace cuda

#define CUDA_CHECK(call) cuda::checkError((call), #call, __FILE__, __LINE__)
 