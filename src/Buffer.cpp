#include <variant>
#include <cstddef>
#include <utility>

#include "Bindings.hpp"
#include "Buffer.hpp" 

namespace tannic {

Buffer::Buffer(std::size_t nbytes, Allocator allocator)
:   nbytes_(nbytes)
,   allocator_(allocator) {
    address_ = std::visit([&](auto& variant) -> void* {
        return variant.allocate(nbytes_);
    }, allocator_);
} 

Buffer::Buffer(Buffer&& other) noexcept 
:   nbytes_(std::exchange(other.nbytes_, 0))
,   allocator_(std::move(other.allocator_))
,   address_(std::exchange(other.address_, nullptr)) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) { 
        std::visit([&](auto& variant) {
            variant.deallocate(address_, nbytes_);
        }, allocator_);
        nbytes_ = std::exchange(other.nbytes_, 0);
        allocator_ = std::move(other.allocator_);
        address_ = std::exchange(other.address_, nullptr);
    }
    return *this;
}

Buffer::~Buffer() {  
    std::visit([&](auto& alloc) {
        alloc.deallocate(address_, nbytes_);
    }, allocator_); 
    address_ = nullptr;
    nbytes_ = 0;
}

void* Buffer::address() { 
    return address_; 
}

const void* Buffer::address() const { 
    return address_; 
}

std::size_t Buffer::nbytes() const { 
    return nbytes_; 
}

Allocator const& Buffer::allocator() const { 
    return allocator_; 
} 
 

} // namespace TANNIC 