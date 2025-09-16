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
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(resource->id);
    CUDA_CHECK(cudaMallocAsync(&ptr, nbytes, stream));
    streams.put(resource->id, stream); 
    return ptr;
} 

void* deallocate(const device_t* resource, void* ptr) {
    setDevice(resource->id);  
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(resource->id);
    CUDA_CHECK(cudaFreeAsync(ptr, stream));
    streams.put(resource->id, stream); 
    return nullptr;
}

void copyFromHost(const device_t* resource, const void* src , void* dst, size_t nbytes) {
    setDevice(resource->id);  
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(resource->id);
    cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream);
    streams.put(resource->id, stream);  
} 

void copyDeviceToHost(const device_t* resource, const void* src, void* dst, size_t nbytes) {
    setDevice(resource->id);
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(resource->id);
    CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, stream));
    streams.put(resource->id, stream);
} 

bool compareFromHost(const device_t* resource, const void* hst_ptr, const void* dvc_ptr, size_t nbytes) {  
    void* buffer = malloc(nbytes);  
    Streams& streams = Streams::instance();
    cudaStream_t stream = streams.pop(resource->id);     
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpyAsync(buffer, dvc_ptr, nbytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    bool result = (memcmp(hst_ptr, buffer, nbytes) == 0);
    streams.put(resource->id, stream);
    free(buffer);   
    return result;
}

} // namespace cuda  