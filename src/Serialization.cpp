#include "Serialization.hpp"

#include <iomanip>    
#include <fstream>
#include <cstring>

// TODO:
// - Support non contiguous tensors
// - Cuda support 
// - Non copy versions

namespace tannic {
 
Blob::Blob(std::size_t nbytes, Allocator allocator) {
    storage_ = std::make_shared<Storage>(nbytes, allocator);
}

Blob::Blob(std::shared_ptr<Storage> storage) {
    storage_ = storage;
}

std::byte* Blob::buffer() const noexcept {
    return static_cast<std::byte*>(storage_->address()) + offset_;
} 

std::size_t Blob::nbytes() const noexcept {
    return storage_->nbytes();
}

std::ptrdiff_t Blob::offset() const {
    return offset_;
}

std::shared_ptr<Storage> Blob::storage() const {
    return storage_;
}
 
std::ostream& operator<<(std::ostream& os, const Blob& blob) {
    os << std::hex << std::setfill('0');
    for (size_t index = 0; index < blob.nbytes(); ++index) {
        os << std::setw(2) << static_cast<int>(blob.buffer()[index]) << " ";
        if ((index + 1) % 16 == 0) os << "\n";
    }
    os << std::dec << "\n";
    return os;
}  

struct Header {
    char magic[6] = {'T', 'A', 'N', 'N', 'I', 'C'};
    size_t tail = 0;

    Header(Blob const& source) {
        std::byte const* buffer = source.buffer();
        const char* claim = reinterpret_cast<const char*>(buffer);
        if(std::memcmp(magic, claim, sizeof(magic)) != 0) {
            throw std::runtime_error("Invalid magic");
        }  
        std::memcpy(&tail, buffer + sizeof(magic), sizeof(size_t));
    }

    Header(Tensor const& source) {
        tail = source.nbytes() + sizeof(magic) + sizeof(size_t);
    }

    size_t nbytes() const {
        return sizeof(magic) + sizeof(size_t);
    }
};

struct Metadata {
    uint8_t rank = 0;
    size_t const* shape = nullptr;
    size_t const* strides = nullptr;
    type dtype = none;

    Metadata(Tensor const& source, Header const& header) {
        rank = source.rank();
        shape = source.shape().address();
        strides = source.strides().address();
        dtype = source.dtype();
    }

    Metadata(Blob const& source, Header const& header) {
        std::byte const* buffer = source.buffer() + header.tail;
        std::memcpy(&rank, buffer, sizeof(uint8_t));
        shape = reinterpret_cast<size_t const*>(buffer + sizeof(uint8_t));
        strides = reinterpret_cast<size_t const*>(buffer + sizeof(uint8_t) + rank * sizeof(size_t));
        dtype = *reinterpret_cast<type const*>(buffer + sizeof(uint8_t) + 2*rank * sizeof(size_t));
    }


    size_t nbytes() const {
        return sizeof(uint8_t) + 2 * sizeof(size_t) * rank + sizeof(type);
    }
};

Blob serialize(Tensor const& tensor) { 
    Header header(tensor);
    Metadata metadata(tensor, header);

    std::size_t nbytes = tensor.nbytes();
    std::size_t htail = nbytes + sizeof(size_t) + sizeof(header.magic);
    
    Blob blob(header.nbytes() + tensor.nbytes() + metadata.nbytes());
    std::memcpy(blob.buffer(), header.magic, sizeof(header.magic));
    std::memcpy(blob.buffer() + sizeof(header.magic), &htail, sizeof(size_t));
    std::memcpy(blob.buffer() + sizeof(header.magic) + sizeof(size_t), tensor.buffer(), nbytes); 
    std::memcpy(blob.buffer() + header.tail, &metadata.rank, sizeof(uint8_t)); 
    std::memcpy(blob.buffer() + header.tail + sizeof(uint8_t), metadata.shape, sizeof(size_t) * metadata.rank);
    std::memcpy(blob.buffer() + header.tail + sizeof(uint8_t) + metadata.rank * sizeof(size_t), metadata.strides, sizeof(size_t) * metadata.rank);
    std::memcpy(blob.buffer() + header.tail + sizeof(uint8_t) + 2 * metadata.rank * sizeof(size_t), &metadata.dtype, sizeof(type));
    return blob;
}

Tensor deserialize(Blob const& blob, Allocator allocator) {
    Header header(blob);
    Metadata metadata(blob, header); 
    Shape shape(metadata.shape, metadata.shape + metadata.rank);
    Strides strides(metadata.strides, metadata.strides + metadata.rank);
    Tensor deserialized(metadata.dtype, shape, strides); deserialized.initialize(allocator);
    std::memcpy(deserialized.buffer(), blob.buffer() + sizeof(header.magic) + sizeof(size_t), header.tail - sizeof(header.magic) - sizeof(size_t));
    return deserialized;
}

void write(const Blob& blob, const std::string& path, uint32_t alignment) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) 
        throw std::runtime_error("Failed to open file for writing: " + path);
    
        stream.write(reinterpret_cast<const char*>(blob.buffer()), blob.nbytes());
}

Blob read(const std::string& path, Allocator allocator) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) 
        throw std::runtime_error("Failed to open file for reading: " + path);
    
    std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    Blob blob(size, allocator); 
    stream.read(reinterpret_cast<char*>(blob.buffer()), size);
    if (!stream) 
        throw std::runtime_error("Failed to read from file: " + path);

    return blob;
}

} // namespace tannic