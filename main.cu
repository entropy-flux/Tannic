#include <iostream>
#include <mutex>
#include <cuda_runtime.h>

class Devices {
private:
    int count;
 
    Devices() {
        cudaError_t cudaStatus = cudaGetDeviceCount(&count);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "CUDA Error: " << cudaGetErrorString(cudaStatus) << std::endl;
            count = 0;
        }
    }

public: 
    Devices(const Devices&) = delete;
    Devices& operator=(const Devices&) = delete;
    Devices(Devices&&) = delete;
    Devices& operator=(Devices&&) = delete; 

    static Devices& getInstance() {
        static Devices instance;
        return instance;
    }

    static int getDeviceCount() {
        Devices& instance = getInstance();
        return instance.count;
    }
    ~Devices() = default;
};
  

int main() {
    std::cout << Devices::getDeviceCount();
    return 0;
}