#include <type_traits>
#include <initializer_list>
#include <variant>
#include <optional>
#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

#include "include/Types.hpp"
#include "include/Tensor.hpp"

#pragma pack(push, 1)
struct Header {
    char magic[4] = {'M', 'L', 'O', 'T'};
    uint32_t offset = 0;
    uint32_t metadata_offset = 0;
    uint32_t reserved = 0;
};

struct Metadata {
    char name[32] = {};
    type dtype = float32;
    size_t size = 0;
    uint8_t rank = 0;
    uint64_t dimensions[8] = {};
    uint32_t alignment = 0; 
    uint32_t offset = 0;
};
#pragma pack(pop)

uint32_t align(uint32_t offset, uint32_t alignment) {
    if (alignment == 0) return offset;
    return (offset + alignment - 1) / alignment * alignment;
}

class Serializer {
    std::vector<uint8_t> buffer;

public:
    void serialize(const std::string& name, const Tensor& tensor, uint32_t alignment = 0) {
        if (name.size() >= sizeof(Metadata::name))
            throw std::runtime_error("Tensor name too long (max 31 characters)");

        uint32_t header_size = sizeof(Header);
        uint32_t aligned_data_offset = align(header_size, alignment);

        // Resize for data
        buffer.resize(aligned_data_offset);
        size_t tensor_bytes = tensor.shape().size() * dsizeof(tensor.dtype());
        const uint8_t* tensor_data = static_cast<const uint8_t*>(tensor.address());
        buffer.insert(buffer.end(), tensor_data, tensor_data + tensor_bytes);

        // Build metadata
        Metadata meta{};
        std::copy(name.begin(), name.end(), meta.name);
        meta.dtype = tensor.dtype();
        meta.size = tensor.shape().size();
        meta.rank = tensor.rank();
        std::copy(tensor.shape().begin(), tensor.shape().end(), meta.dimensions);
        meta.alignment = alignment;
        meta.offset = aligned_data_offset;

        uint32_t metadata_offset = static_cast<uint32_t>(buffer.size());
        const uint8_t* meta_ptr = reinterpret_cast<const uint8_t*>(&meta);
        buffer.insert(buffer.end(), meta_ptr, meta_ptr + sizeof(Metadata));

        Header header;
        std::memcpy(header.magic, "MLOT", 4);
        header.offset = aligned_data_offset;
        header.metadata_offset = metadata_offset;

        std::memcpy(buffer.data(), &header, sizeof(Header));
    }

    const uint8_t* Data() const { return buffer.data(); }
    size_t Size() const { return buffer.size(); }
};

class Deserializer {
    const uint8_t* buffer;
    size_t size;
    Header header;
    Metadata meta;

public:
    Deserializer(const uint8_t* data, size_t len) : buffer(data), size(len) {
        if (len < sizeof(Header))
            throw std::runtime_error("Buffer too small for header");

        std::memcpy(&header, buffer, sizeof(Header));
        if (std::string(header.magic, 4) != "MLOT")
            throw std::runtime_error("Invalid magic number");

        if (header.metadata_offset + sizeof(Metadata) > size)
            throw std::runtime_error("Invalid metadata offset");

        std::memcpy(&meta, buffer + header.metadata_offset, sizeof(Metadata));
    }

    Tensor read_tensor() const {
        size_t data_size = meta.size * dsizeof(meta.dtype);
        if (meta.offset + data_size > size)
            throw std::runtime_error("Tensor data out of bounds");

        Shape shape(meta.dimensions, meta.dimensions + meta.rank);
        void* data_ptr = const_cast<void*>(static_cast<const void*>(buffer + meta.offset));
        Storage storage(meta.size, dsizeof(meta.dtype), View{data_ptr});

        return Tensor(storage, std::move(shape), Strides(shape), meta.dtype, 0);
    }

    const Metadata& metadata() const { return meta; }
};

void print_hex(const uint8_t* data, size_t size) {
    std::cout << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; ++i) {
        std::cout << std::setw(2) << static_cast<int>(data[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << "\n";
    }
    std::cout << std::dec << "\n";
}

int main() {
    Serializer s;

    Tensor tensor({2, 3}, float32);
    for (int i = 0; i < 6; ++i)
        static_cast<float*>(tensor.address())[i] = float(i + 1);

    s.serialize("matrix", tensor, 64);

    std::cout << "Serialized Bytes:\n";
    print_hex(s.Data(), s.Size());

    Deserializer d(s.Data(), s.Size());
    Tensor t = d.read_tensor();

    std::cout << "Deserialized Tensor:\n";
    std::cout << "  DType: " << traits[t.dtype()].name << "\n";
    std::cout << "  Shape: " << t.shape() << "\n";
    std::cout << "  Data: " << t << "\n";
}
