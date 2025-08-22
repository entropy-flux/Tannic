#include <variant>
#include <cstddef>
#include <utility>

#include "bindings.hpp"
#include "buffer.hpp" 

namespace tannic {

Buffer::Buffer(std::size_t nbytes, Environment environment)
:   nbytes_(nbytes)
,   environment_(environment) {
    address_ = std::visit([&](auto& variant) -> void* {
        return variant.allocate(nbytes_);
    }, environment_);
} 

Buffer::Buffer(Buffer&& other) noexcept 
:   nbytes_(std::exchange(other.nbytes_, 0))
,   environment_(std::move(other.environment_))
,   address_(std::exchange(other.address_, nullptr)) {}


Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) { 
        std::visit([&](auto& variant) {
            variant.deallocate(address_, nbytes_);
        }, environment_);
        nbytes_ = std::exchange(other.nbytes_, 0);
        environment_ = std::move(other.environment_);
        address_ = std::exchange(other.address_, nullptr);
    }
    return *this;
}

Buffer::~Buffer() {  
    std::visit([&](auto& alloc) {
        alloc.deallocate(address_, nbytes_);
    }, environment_); 
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

Environment const& Buffer::environment() const { 
    return environment_; 
}  

} // namespace TANNIC 