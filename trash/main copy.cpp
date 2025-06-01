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
    char name[32] = {};
    type dtype;
    uint8_t rank = 0;
    uint64_t shape[8] = {};
    size_t strides[8] = {};

    Metadata() = default;

    Metadata(const std::string& name, type dtype, const Shape& shape, const Strides& strides)
    :   dtype(dtype)
    ,   rank(static_cast<uint8_t>(shape.rank())) {
        std::ranges::copy(name | std::views::take(31), this-> name);
        this-> name[31] = '\0';
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
    
    friend Blob serialize(const std::string& name, const Tensor& tensor, uint32_t alignment, Allocator allocator);
}; 


Blob serialize(const std::string& name, const Tensor& tensor, uint32_t alignment = 0, Allocator allocator = Host{}) {
    if (name.size() >= 32)
        throw std::runtime_error("Tensor name too long (max 31 characters)");
 
    Header header(align(sizeof(Header), alignment), tensor.storage().memory());  
    Metadata metadata(name, tensor.dtype(), tensor.shape(), tensor.strides());
    return Blob(header, tensor, metadata, allocator);
} 
 

class Deserializer {
    const std::byte* buffer_;
    size_t size_;
    Header header_;
    Metadata meta_;

public:
    Deserializer(const std::byte* data, size_t len)
    :   buffer_(data)
    ,   size_(len)  {
        if (len < sizeof(Header))
            throw std::runtime_error("Buffer too small for header");

        std::memcpy(&header_, buffer_, sizeof(Header));
        if (std::string(header_.magic, 4) != "MLBC")
            throw std::runtime_error("Invalid magic number");

        if (header_.tail + sizeof(Metadata) > size_)
            throw std::runtime_error("Invalid metadata offset");

        std::memcpy(&meta_, buffer_ + header_.tail, sizeof(Metadata));
    }

    Tensor read_tensor() const {
        size_t count = 1;
        for (uint8_t i = 0; i < meta_.rank; ++i)
            count *= meta_.shape[i];

        size_t data_size = count * dsizeof(meta_.dtype);

        if (header_.body + data_size > size_)
            throw std::runtime_error("Tensor data out of bounds");

        Shape shape(meta_.shape, meta_.shape + meta_.rank);
        Strides strides(meta_.strides, meta_.strides + meta_.rank);

        // buffer_ is const std::byte*, we cast to void* for Storage.
        void* data_ptr = const_cast<void*>(static_cast<const void*>(buffer_ + header_.body));
        Storage storage(data_size / dsizeof(meta_.dtype), dsizeof(meta_.dtype), View{data_ptr});

        return Tensor(storage, std::move(shape), std::move(strides), meta_.dtype, 0);
    }

    const Metadata& metadata() const noexcept { return meta_; }
};


int main() {  

    Tensor tensor({2, 3}, float32);
    float* data_ptr = static_cast<float*>(tensor.address());
    for (int i = 0; i < 6; ++i) data_ptr[i] = float(i + 1);
    std::cout << tensor;

    auto serialized = serialize("matrix", tensor, 64);

    std::cout << "Blob Bytes:\n";
    for (size_t i = 0; i < serialized.size(); ++i) {
        // convert std::byte to int for printing
        std::cout << std::hex << std::setw(2) << std::setfill('0') << std::to_integer<int>(serialized.bytes()[i]) << ' ';
        if ((i + 1) % 16 == 0) std::cout << '\n';
    }
    std::cout << std::dec << std::endl;
 

    Deserializer deserializer(serialized.bytes(), serialized.size());
    Tensor t = deserializer.read_tensor();

    std::cout << "Deserialized Tensor:\n";
    std::cout << "  DType: " << t.dtype() << "\n";
    std::cout << "  Shape: " << t.shape() << "\n";
    std::cout << "  Strides: " << t.strides() << "\n";
    std::cout << "  Data: " << t << "\n";   
    return 0;
}
