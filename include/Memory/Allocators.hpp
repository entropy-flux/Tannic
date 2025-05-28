#include <variant>
#include <atomic>
#include <cstddef>

class Host {
    public:
    void* allocate(std::size_t memory) {
        void* address = ::operator new(memory);
        return address;
    }

    void deallocate(void* address, std::size_t memory) {
        ::operator delete(address);
    }

};

using Allocator = std::variant<Host>;