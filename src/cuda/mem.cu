#include "cuda/exc.cuh"
#include "cuda/mem.cuh"
#include "cuda/streams.cuh"

namespace cuda { 
  
int getDeviceCount() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count); CUDA_CHECK(err);
    return count;
}

void setDevice(int id) {
    CUDA_CHECK(cudaSetDevice(id));
}

void* allocate(const device_t* resource, size_t nbytes) {
    setDevice(resource->id); 
    void* ptr = nullptr;
    if (resource->traits & SYNC) { 
        CUDA_CHECK(cudaMalloc(&ptr, nbytes));
    } else {
        Streams& streams = Streams::instance();
        cudaStream_t stream = streams.pop(resource->id);
        CUDA_CHECK(cudaMallocAsync(&ptr, nbytes, stream));
        streams.put(resource->id, stream);
    }
    return ptr;
} 

void* deallocate(const device_t* resource, void* ptr) {
    setDevice(resource->id);
    if (resource->traits & SYNC) {
        CUDA_CHECK(cudaFree(ptr));
    } else {
        Streams& streams = Streams::instance();
        cudaStream_t stream = streams.pop(resource->id);
        CUDA_CHECK(cudaFreeAsync(ptr, stream));
        streams.put(resource->id, stream);
    }
    return nullptr;
}

void copyFromHost(const device_t* resource, const void* src , void* dst, size_t nbytes) {
    setDevice(resource->id);
    if (resource->traits & SYNC) {
        cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice);
    } 
    else {
        Streams& streams = Streams::instance();
        cudaStream_t stream = streams.pop(resource->id);
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream);
        streams.put(resource->id, stream); 
    }
} 

bool compareFromHost(const device_t* resource, const void* hst_ptr, const void* dvc_ptr, size_t nbytes) {  
    void* buffer = malloc(nbytes); 
    CUDA_CHECK(cudaMemcpy(buffer, dvc_ptr, nbytes, cudaMemcpyDeviceToHost));
    bool result = (memcmp(hst_ptr, buffer, nbytes) == 0);
    free(buffer);   
    return result;
}

} // namespace cuda  