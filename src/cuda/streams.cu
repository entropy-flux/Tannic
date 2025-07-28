#include "cuda/streams.cuh" 
#include <vector>
#include <forward_list>
#include <cuda_runtime.h> 
#include "cuda/exc.cuh"

namespace cuda {   
    
Streams::Streams() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);  
    streams_.resize(count);
}

Streams::~Streams() {
    for (auto& device : streams_) {
        for (cudaStream_t stream : device) {
            cudaStreamDestroy(stream); 
        }
    }
}

cudaStream_t Streams::pop(int device) {
    auto& streams = streams_[device];
    if (streams.empty()) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        return stream;
    } else {
        cudaStream_t stream = streams.front();
        streams.pop_front();
        return stream;
    }
}

void Streams::put(int device, cudaStream_t stream) {
    streams_[device].push_front(stream);
}

}