#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>   
#include <vector>
#include <forward_list>

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

enum environment {
    HOST,
    DEVICE
};

enum host {
    PAGEABLE = 1 << 0, 
    PINNED   = 1 << 1, 
    MAPPED   = 1 << 2      
};
 
 
struct host_t {
    enum host traits;
};

struct device_t {
    int id; 
}; 

struct allocator_t { 
    enum environment environment;
    union {
        struct host_t host;
        struct device_t device;
    } resource;
}; 

struct tensor_t {
    uint8_t rank;
    void* address;
    const uint32_t* shape;
    const int64_t* strides;  
    enum type dtype; 
};  
 
struct stream_t { 
    uintptr_t address;
};  
 

class Streams {
private:
    Streams() {
        int count;
        cudaError_t err = cudaGetDeviceCount(&count);  
        streams_.resize(count);
    }

    std::vector<std::forward_list<cudaStream_t>> streams_;

public: 
    ~Streams() {
        for (auto& device : streams_) {
            for (cudaStream_t stream : device) {
                cudaStreamDestroy(stream); 
            }
        }
    }

    Streams(const Streams&) = delete;
    Streams& operator=(const Streams&) = delete;
    Streams(Streams&&) = delete;
    Streams& operator=(Streams&&) = delete;

    cudaStream_t pop(int device) {
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

    void put(int device, cudaStream_t stream) {
        streams_[device].push_front(stream);
    }

    static Streams& instance() {
        static Streams instance;
        return instance;
    }
};       

stream_t pop_stream(const device_t* device) {
    stream_t s;
    cudaStream_t stream = Streams::instance().pop(device->id);
    s.address = reinterpret_cast<uintptr_t>(stream);
    return s;
}

void put_stream(const device_t* device, stream_t s) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(s.address);
    Streams::instance().put(device->id, stream);
}

#include <unordered_map>

class Cache {
    struct Chunk {
        void* address;
        size_t nbytes;
    };

private:
    std::unordered_map<uintptr_t, Chunk> chunks_; 
    Cache() = default;

public:
    Cache(const Cache&) = delete;
    Cache& operator=(const Cache&) = delete;
    Cache(Cache&&) = delete;
    Cache& operator=(Cache&&) = delete;

    ~Cache() {
        for (auto& [key, chunk] : chunks_) { 
            cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(key);
            cudaFreeAsync(chunk.address, cudaStream);
        }
        chunks_.clear();

        int devices = 0;
        cudaGetDeviceCount(&devices);
        for (int dvc = 0; dvc < devices; ++dvc) {
            cudaSetDevice(dvc);            
            cudaDeviceSynchronize();    
        }
    }

    static Cache& instance() {
        static Cache instance;
        return instance;
    }

    void* get(const stream_t* stream, size_t nbytes) {
        auto iterator = chunks_.find(stream->address);
        if (iterator == chunks_.end()) {
            cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream->address); 
            void* address = nullptr; 
            cudaMallocAsync(&address, nbytes, cudaStream);
            chunks_.emplace(stream->address, Chunk{address, nbytes});
            return address;
        } 

        else {
            Chunk& chunk = iterator->second;
            if (nbytes > chunk.nbytes) {
                cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(iterator->first);
                cudaFreeAsync(chunk.address, cudaStream);
                cudaMallocAsync(&chunk.address, nbytes, cudaStream);
            }
            return chunk.address;
        }
    }
}; 

size_t lsizeof(uint8_t rank) {
    return rank * (sizeof(uint32_t) + sizeof(int64_t));
}



int main() { 
    device_t device{0};
    stream_t stream = pop_stream(&device);


    put_stream(&device, stream);
}