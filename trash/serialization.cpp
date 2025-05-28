#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <type_traits>

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <type_traits>

enum class DType : uint8_t {
    F32 = 0,
    I32 = 1,
};

#pragma pack(push, 1)
struct MainHeader {
    char magic[4] = {'C', 'T', 'M', 'L'};  // "CTML"
    uint32_t num_arrays = 0;
    uint32_t directory_offset = 0;
    uint32_t reserved = 0;
};

struct DirectoryEntry {
    char name[32] = {};
    DType dtype = DType::F32;
    uint32_t size = 0;
    uint32_t ndim = 0;
    uint32_t dims[8] = {};
    uint32_t alignment = 64;
    uint32_t data_offset = 0;
};
#pragma pack(pop)

template <typename T>
constexpr DType dtype_of();

template <>
constexpr DType dtype_of<float>() { return DType::F32; }

template <>
constexpr DType dtype_of<int32_t>() { return DType::I32; }

uint32_t align_offset(uint32_t offset, uint32_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

class ArraySerializer {
    std::vector<uint8_t> buffer;
    std::vector<DirectoryEntry> entries;

public:
    ArraySerializer() {
        // Reserve space for MainHeader at the start
        buffer.resize(sizeof(MainHeader), 0);
        std::memcpy(buffer.data(), "CTML", 4);
    }

    template <typename T>
    void write_tensor(const std::string& name, const T* data, size_t size,
                      const uint32_t* shape, uint32_t ndim,
                      uint32_t alignment = 64) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, int32_t>,
                      "Only float/int32_t supported");

        if (name.size() >= sizeof(DirectoryEntry::name))
            throw std::runtime_error("Array name too long (max 31 chars)");
        if (ndim > 8) throw std::runtime_error("Tensor rank > 8 not supported");

        uint32_t current_offset = static_cast<uint32_t>(buffer.size());
        uint32_t aligned_offset = align_offset(current_offset, alignment);

        if (aligned_offset > current_offset) {
            buffer.resize(aligned_offset, 0);
        }

        // Append tensor data
        const uint8_t* raw_data = reinterpret_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), raw_data, raw_data + size * sizeof(T));

        // Create directory entry
        DirectoryEntry entry{};
        std::fill(std::begin(entry.name), std::end(entry.name), 0);
        std::copy(name.begin(), name.end(), entry.name);
        entry.dtype = dtype_of<T>();
        entry.size = static_cast<uint32_t>(size);
        entry.ndim = ndim;
        std::copy(shape, shape + ndim, entry.dims);
        entry.alignment = alignment;
        entry.data_offset = aligned_offset;

        entries.push_back(entry);
    }

    // Finalize directory and write header at front
    void finalize() {
        uint32_t directory_offset = static_cast<uint32_t>(buffer.size());

        for (const auto& entry : entries) {
            const uint8_t* entry_ptr = reinterpret_cast<const uint8_t*>(&entry);
            buffer.insert(buffer.end(), entry_ptr, entry_ptr + sizeof(DirectoryEntry));
        }

        // Write main header (overwrite beginning)
        MainHeader header{};
        std::memcpy(header.magic, "CTML", 4);
        header.num_arrays = static_cast<uint32_t>(entries.size());
        header.directory_offset = directory_offset;

        std::memcpy(buffer.data(), &header, sizeof(MainHeader));
    }

    // Get pointer to serialized data buffer
    const uint8_t* data() const { return buffer.data(); }
    size_t size() const { return buffer.size(); }
};


class ArrayDeserializer {
    const uint8_t* buffer = nullptr;
    size_t buffer_size = 0;

    MainHeader header{};
    std::vector<DirectoryEntry> entries;

public:
    ArrayDeserializer(const uint8_t* data, size_t size)
        : buffer(data), buffer_size(size) {
        if (size < sizeof(MainHeader))
            throw std::runtime_error("Buffer too small for header");

        std::memcpy(&header, buffer, sizeof(MainHeader));
        if (std::string(header.magic, 4) != "CTML")
            throw std::runtime_error("Invalid magic header");

        size_t dir_size = header.num_arrays * sizeof(DirectoryEntry);
        if (header.directory_offset + dir_size > buffer_size)
            throw std::runtime_error("Directory offset/size out of range");

        entries.resize(header.num_arrays);
        std::memcpy(entries.data(), buffer + header.directory_offset, dir_size);
    }

    const DirectoryEntry* find_entry(const std::string& name) const {
        for (const auto& e : entries) {
            std::string entry_name(e.name, strnlen(e.name, sizeof(e.name)));
            if (entry_name == name) return &e;
        }
        return nullptr;
    }

    template <typename T>
    std::vector<T> read_tensor(const std::string& name) const {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, int32_t>,
                      "Only float/int32_t supported");

        const DirectoryEntry* entry = find_entry(name);
        if (!entry)
            throw std::runtime_error("Tensor not found: " + name);

        if constexpr (std::is_same_v<T, float>) {
            if (entry->dtype != DType::F32)
                throw std::runtime_error("Dtype mismatch (expected float)");
        } else {
            if (entry->dtype != DType::I32)
                throw std::runtime_error("Dtype mismatch (expected int32)");
        }

        size_t data_start = entry->data_offset;
        size_t data_bytes = entry->size * sizeof(T);
        if (data_start + data_bytes > buffer_size)
            throw std::runtime_error("Tensor data out of buffer range");

        const T* data_ptr = reinterpret_cast<const T*>(buffer + data_start);
        return std::vector<T>(data_ptr, data_ptr + entry->size);
    }

    void print_all_tensors() const {
        for (const auto& entry : entries) {
            std::string name(entry.name, strnlen(entry.name, sizeof(entry.name)));
            std::cout << "Tensor: " << name << "\n";
            std::cout << "  DType: " << (entry.dtype == DType::F32 ? "float" : "int32") << "\n";
            std::cout << "  Shape: [";
            for (uint32_t i = 0; i < entry.ndim; ++i) {
                std::cout << entry.dims[i];
                if (i + 1 < entry.ndim) std::cout << ", ";
            }
            std::cout << "]\n";

            // Print tensor data
            if (entry.dtype == DType::F32) {
                auto data = read_tensor<float>(name);
                std::cout << "  Data: ";
                for (auto v : data) std::cout << v << " ";
                std::cout << "\n";
            } else if (entry.dtype == DType::I32) {
                auto data = read_tensor<int32_t>(name);
                std::cout << "  Data: ";
                for (auto v : data) std::cout << v << " ";
                std::cout << "\n";
            }
        }
    }
};

#include <iomanip>

void print_hex(const uint8_t* data, size_t size) {
    std::cout << std::hex << std::setfill('0');
    for (size_t i = 0; i < size; i++) {
        std::cout << std::setw(2) << static_cast<int>(data[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << "\n";  // new line every 16 bytes
    }
    std::cout << std::dec << "\n";  // reset to decimal
}

int main() {
    // Serialize
    ArraySerializer serializer;

    float matrix[6] = {1, 2, 3, 4, 5, 6};
    uint32_t shape_matrix[2] = {2, 3};
    serializer.write_tensor("matrix", matrix, 6, shape_matrix, 2);

    int32_t vector[4] = {10, 20, 30, 40};
    uint32_t shape_vector[1] = {4};
    serializer.write_tensor("vector", vector, 4, shape_vector, 1);

    serializer.finalize();

    // Print serialized data in hex
    std::cout << "Serialized data:\n";
    print_hex(serializer.data(), serializer.size());

    // Deserialize
    try {
        ArrayDeserializer deserializer(serializer.data(), serializer.size());
        std::cout << "\nDeserialized tensors:\n";
        deserializer.print_all_tensors();
    } catch (const std::exception& e) {
        std::cerr << "Deserialization error: " << e.what() << "\n";
    }

    return 0;
}
