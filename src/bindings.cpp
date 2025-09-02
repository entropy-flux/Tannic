#include "bindings.hpp"
#include "tensor.hpp"
#include "runtime/tensor.h"
#include "runtime/resources.h"
#include "runtime/streams.h"

#ifdef CUDA
#include "cuda/streams.cuh"
#include "cuda/mem.cuh"
#else 
namespace cuda {
struct Streams {
    static Streams& instance() {static Streams s; return s; }
    void* pop(int) { throw std::runtime_error("CUDA streams not available"); }
    void  put(int, void*) { throw std::runtime_error("CUDA streams not available"); }
};
}
#endif

namespace tannic {

stream_t pop_stream(const device_t* dvc) {
#ifdef CUDA
    stream_t stream;
    cuda::Streams& streams = cuda::Streams::instance();
    cudaStream_t cudaStream = streams.pop(dvc->id);
    stream.address = reinterpret_cast<uintptr_t>(cudaStream);
    return stream;
#else
    throw std::runtime_error("pop_stream: CUDA not available");
#endif
}

void put_stream(const device_t* dvc, stream_t stream) {
#ifdef CUDA
    cuda::Streams& streams = cuda::Streams::instance();
    cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream.address);
    streams.put(dvc->id, cudaStream);
#else
    throw std::runtime_error("put_stream: CUDA not available");
#endif
}

host_t structure(Host const& resource) {
    // OLD FUNCTION TO BE DELETED.
    return host_t{
        .traits = resource.pinned() ? PINNED : PAGEABLE
    };
}

device_t structure(Device const& resource) {
    // OLD FUNCTION TO BE DELETED.
    return device_t{
        .id = resource.id(),
        .traits = resource.blocking() ? SYNC : ASYNC
    };
}

environment_t structure(Environment const& environment) {
    // OLD FUNCTION TO BE DELETED.
    if (std::holds_alternative<Host>(environment)) {
        Host const& resource = std::get<Host>(environment);
        return environment_t{
            .environment = HOST,
            .resource = {.host = structure(resource)},
        };
    } else {
        Device const& resource = std::get<Device>(environment);
        return environment_t{
            .environment = DEVICE,
            .resource = {.device = structure(resource)},
        };
    }
}

tensor_t structure(Tensor const& tensor) {
    // OLD FUNCTION TO BE DELETED.
    const Environment& alloc = tensor.environment();
    shape_t shape{};
    strides_t strides{};
    for (int dimension = 0; dimension < tensor.rank(); ++dimension) {
        shape.sizes[dimension] = tensor.shape()[dimension];
        strides.sizes[dimension] = tensor.strides()[dimension];
    }

    if (std::holds_alternative<Host>(alloc)) {
        Host const& resource = std::get<Host>(alloc);
        return tensor_t{
            .address = (void*)(tensor.bytes()),
            .rank = tensor.rank(),
            .shape = shape,
            .strides = strides,
            .dtype = tensor.dtype(),
            .environment = {
                .environment = HOST,
                .resource = {.host = structure(resource)},
            },
            .size = tensor.nelements(),
            .layout = tensor.is_singleton() ? SINGLETON : tensor.is_contiguous() ? CONTIGUOUS : STRIDED
        };
    } else {
        Device const& resource = std::get<Device>(alloc);
        return tensor_t{
            .address = (void*)(tensor.bytes()),
            .rank = tensor.rank(),
            .shape = shape,
            .strides = strides,
            .dtype = tensor.dtype(),
            .environment = {
                .environment = DEVICE,
                .resource = {.device = structure(resource)},
            },
            .size = tensor.nelements(),
            .layout = tensor.is_singleton() ? SINGLETON : tensor.is_contiguous() ? CONTIGUOUS : STRIDED
        };
    }
}

status resolve_two_environment(const environment_t* a, const environment_t* b, environment_t* result_out) {
    if (!a || !b || !result_out) {
        return NULL_ALLOCATOR;
    }

    result_out->environment = HOST;
    result_out->resource.host.traits = PAGEABLE;

    if (a->environment == DEVICE && b->environment == DEVICE) {
        if (a->resource.device.id != b->resource.device.id) {
            return INCOMPATIBLE_DEVICES;
        }

        result_out->environment = DEVICE;
        result_out->resource.device.id = a->resource.device.id;
        result_out->resource.device.traits =
            (a->resource.device.traits == ASYNC || b->resource.device.traits == ASYNC)
            ? ASYNC : SYNC;
    }
    else if (a->environment == DEVICE) {
        return INCOMPATIBLE_DEVICES;
    }
    else if (b->environment == DEVICE) {
        return INCOMPATIBLE_DEVICES;
    }
    else {
        if ((a->resource.host.traits & MAPPED) || (b->resource.host.traits & MAPPED)) {
            result_out->resource.host.traits = MAPPED;
        }
        else if ((a->resource.host.traits & PINNED) || (b->resource.host.traits & PINNED)) {
            result_out->resource.host.traits = PINNED;
        }
    }
    return SUCCESS;
}

status resolve_three_environment(
    const environment_t* a,
    const environment_t* b,
    const environment_t* c,
    environment_t* result_out
) {
    if (!a || !b || !c || !result_out) {
        return NULL_ALLOCATOR;
    }
 
    result_out->environment = HOST;
    result_out->resource.host.traits = PAGEABLE;

    bool any_device = false;
    int device_id = -1;
    enum device device_traits = SYNC;

    const environment_t* envs[3] = {a, b, c};

    for (int i = 0; i < 3; ++i) {
        const auto& env = envs[i];
        if (env->environment == DEVICE) {
            if (!any_device) { 
                any_device = true;
                device_id = env->resource.device.id;
                device_traits = env->resource.device.traits;
            } else { 
                if (env->resource.device.id != device_id) {
                    return INCOMPATIBLE_DEVICES;
                }   
                if (env->resource.device.traits == ASYNC) device_traits = ASYNC;
            }
        } else { 
            if (env->resource.host.traits & MAPPED) result_out->resource.host.traits = MAPPED;
            else if (env->resource.host.traits & PINNED) result_out->resource.host.traits = PINNED;
        }
    }

    if (any_device) {
        result_out->environment = DEVICE;
        result_out->resource.device.id = device_id;
        result_out->resource.device.traits = device_traits;
    }

    return SUCCESS;
}
 
}