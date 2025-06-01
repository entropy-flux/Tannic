#include <variant>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <ranges>
#include <iomanip>
#include <cstddef>   
#include "include/Types.hpp"
#include "include/Tensor.hpp"


#pragma pack(push, 1)
struct Header {
    char magic[4] = {'M', 'L', 'B', 'C'};
    uint8_t body = 0;
    uint64_t tail = 0; 

    Header() = default;

    Header(uint8_t offset, uint64_t size)
    :   body(offset)
    ,   tail(offset + size) {}

};

struct Metadata { 
    type dtype = any;
    uint8_t rank = 0;
    uint64_t shape[8] = {};
    size_t strides[8] = {};

    Metadata() = default;

    Metadata(type dtype, const Shape& shape, const Strides& strides)
    :   dtype(dtype)
    ,   rank(static_cast<uint8_t>(shape.rank())) { 
        std::copy(shape.begin(), shape.end(), this-> shape);
        std::copy(strides.begin(), strides.end(), this-> strides);
    }
};

#pragma pack(pop)
 

inline uint32_t align(uint32_t offset, uint32_t alignment) {
    if (alignment == 0) return offset;
    return (offset + alignment - 1) / alignment * alignment;
} 
 

class Blob { 
    public:
    Blob(std::byte* bytes, std::size_t size)
    :   bytes_(bytes)
    ,   size_(size)
    ,   allocator_(View{bytes}) {}

    ~Blob() {
        if (bytes_) {
            std::visit([this](auto& alloc) {
                alloc.deallocate(bytes_, size_);
            }, allocator_);
        }
    }
 
    Blob(const Blob&) = delete;
    Blob& operator=(const Blob&) = delete;
 
    Blob(Blob&& other) noexcept
        : bytes_(std::exchange(other.bytes_, nullptr))
        , size_(std::exchange(other.size_, 0))
        , allocator_(std::move(other.allocator_)) {
    }

    Blob& operator=(Blob&& other) noexcept {
        if (this != &other) {
            if (bytes_) {
                std::visit([this](auto& alloc) {
                    alloc.deallocate(bytes_, size_);
                }, allocator_);
            }
            bytes_ = std::exchange(other.bytes_, nullptr);
            size_ = std::exchange(other.size_, 0);
            allocator_ = std::move(other.allocator_);
        }
        return *this;
    }
    
    std::byte* bytes() noexcept { return bytes_; }
    const std::byte* bytes() const noexcept { return bytes_; }
    std::size_t size() const noexcept { return size_; }

    private: 
    std::byte* bytes_ = nullptr;
    std::size_t size_ = 0;
    Allocator allocator_;    
    
    Blob(Header const& header, Tensor const& tensor, Metadata const& metadata, Allocator allocator) 
    :   size_(header.tail + sizeof(Metadata)) 
    ,   allocator_(allocator) { 
        bytes_ = std::visit([size = size_](auto& variant) {
            return static_cast<std::byte*>(variant.allocate(size));
        }, allocator);

        if (!bytes_) 
            throw std::bad_alloc();
            
        std::memcpy(bytes_, &header, header.body); 
        std::memcpy(bytes_ + header.body, tensor.address(), header.tail - header.body);
        std::memcpy(bytes_ + header.tail, &metadata, sizeof(Metadata));
    }
    
    friend Blob serialize(const Tensor& tensor, uint32_t alignment, Allocator allocator);
}; 

Blob serialize(Tensor const& tensor, uint32_t alignment = 0, Allocator allocator = Host{}) {
    Header header(align(sizeof(Header), alignment), tensor.storage().memory());  
    Metadata metadata(tensor.dtype(), tensor.shape(), tensor.strides());
    return Blob(header, tensor, metadata, allocator);
} 

Tensor deserialize(Blob const& blob) {
    Header header; 
    std::memcpy(&header, blob.bytes(), sizeof(Header));
    assert(std::string(header.magic, 4) == "MLBC" && "Invalid magic number");

    Metadata metadata; 
    std::memcpy(&metadata, blob.bytes() + header.tail, sizeof(Metadata));
    void* address = const_cast<void*>(static_cast<void const*>(blob.bytes() + header.body));
    Shape shape(std::span<Shape::size_type>(metadata.shape, metadata.rank));
    Strides strides(std::span<Strides::step_type>(metadata.strides, metadata.rank));
    Storage storage(shape.size(), dsizeof(metadata.dtype), View{address});
    return Tensor(storage, std::move(shape), std::move(strides), metadata.dtype, 0);
}