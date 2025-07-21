#include <iostream>   
#include <cstddef> 
#include <memory>
#include <cstdint> 

#include "Types.hpp" 
#include "Resources.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "Tensor.hpp"

namespace tannic { 

class Blob { 
public:
    Blob(std::size_t nbytes, Allocator allocator = Host{});
    Blob(std::shared_ptr<Storage> storage);
    std::byte* buffer() const noexcept;
    std::size_t nbytes() const noexcept;
    std::ptrdiff_t offset() const;
    std::shared_ptr<Storage> storage() const;

private: 
    std::ptrdiff_t offset_ = 0;
    std::shared_ptr<Storage> storage_ = nullptr;
}; 

std::ostream& operator<<(std::ostream& os, const Blob& blob); 
Blob serialize(Tensor const& tensor);
Tensor deserialize(Blob const& blob, Allocator allocator = Host{}); 
void write(const Blob& blob, const std::string& path, uint32_t alignment = 0 );
Blob read(const std::string& path, Allocator allocator = Host{});

} // namespace tannic