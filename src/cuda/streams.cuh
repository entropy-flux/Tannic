#pragma once
#include <cuda_runtime.h> 
#include <vector>
#include <forward_list> 

namespace cuda {  
 
class Streams {
private:
    Streams();
    ::std::vector<::std::forward_list<cudaStream_t>> streams_;

public: 
    ~Streams(); 
    Streams(const Streams&) = delete;
    Streams& operator=(const Streams&) = delete;
    Streams(Streams&&) = delete;
    Streams& operator=(Streams&&) = delete;
    cudaStream_t pop(int device);
    void put(int device, cudaStream_t stream);

    static Streams& instance() {
        static Streams instance;
        return instance;
    }
}; 

}