#ifndef ALLOCATORS_HPP
#define ALLOCATORS_HPP

#include <variant>
#include <atomic>
#include <cstddef>

struct View {
    void* buffer = nullptr;
    
    void* allocate(std::size_t) {
        return buffer;
    }

    void deallocate(void*, std::size_t) { 
        buffer = nullptr;
    }

};

struct Host {
    public:
    void* allocate(std::size_t memory) {
        void* address = ::operator new(memory);
        return address;
    }

    void deallocate(void* address, std::size_t) { 
        ::operator delete(address);
    }

};

using Allocator = std::variant<Host, View>;

#endif // ALLOCATORS_HPP