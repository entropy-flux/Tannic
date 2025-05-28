#ifndef STORAGE_HPP
#define STORAGE_HPP

#include <variant>
#include <atomic>
#include <cstddef>
#include <vector>

#include "Types.hpp"
#include "Memory/Allocators.hpp"
#include <utility>

class Storage {
public: 

    Storage(std::size_t size, uint8_t dsize, Allocator allocator = Host{})
        : memory_(size * dsize) 
        , allocator_(allocator) {
            references_ = new std::atomic<std::size_t>(1); 
            address_ = std::visit([&](auto& allocator) -> void* {
                return allocator.allocate(memory_);
            }, allocator_);
        }
 
    Storage(const Storage& other)
        : memory_(other.memory_) 
        , allocator_(other.allocator_)
        , address_(other.address_)
        , references_(other.references_) {
            ++(*references_);
        }
 
    Storage(Storage&& other) noexcept
        : memory_(other.memory_) 
        , allocator_(std::move(other.allocator_))
        , address_(std::exchange(other.address_, nullptr))
        , references_(std::exchange(other.references_, nullptr)) {}

 
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            release();
            memory_ = other.memory_; 
            allocator_ = other.allocator_;
            address_ = other.address_;
            references_ = other.references_;
            ++(*references_);
        }
        return *this;
    }
    
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            release();
            memory_ = other.memory_; 
            allocator_ = std::move(other.allocator_);
            address_ = std::exchange(other.address_, nullptr);
            references_ = std::exchange(other.references_, nullptr); 
        }
        return *this;
    }

    ~Storage() {
        release();
    }

    std::size_t references() const {
        return references_ ? references_->load() : 0;
    }

    void* address() const {
        return address_;
    }

private:
    void release() {
        if (references_) {
            if (--(*references_) == 0) {
                if (address_) {
                    std::visit([&](auto& allocator) {
                        allocator.deallocate(address_, memory_);
                    }, allocator_);
                }
                delete references_;
            }
            references_ = nullptr;
            address_ = nullptr;
        }
    }

    std::size_t memory_; 
    Allocator allocator_;
    void* address_ = nullptr;
    std::atomic<std::size_t>* references_ = nullptr;
};
  
#endif // STORAGE_HPP