#include <cstdint>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <type_traits>
#include <algorithm>

// --- DType Enum ---
enum class DType : uint8_t {
    F32 = 0,
    I32 = 1,
};

// --- Main Header ---
#pragma pack(push, 1)
struct MainHeader {
    char magic[4] = {'C', 'T', 'M', 'L'};  // "CTML"
    uint32_t num_arrays = 0;
    uint32_t directory_offset = 0;
    uint32_t reserved = 0;
};
#pragma pack(pop)

// --- Directory Entry ---
#pragma pack(push, 1)
struct DirectoryEntry {
    char name[32] = {};
    DType dtype = DType::F32;
    uint32_t size = 0;           // total elements
    uint32_t ndim = 0;           // number of dims
    uint32_t dims[8] = {};       // dimensions (max rank 8)
    uint32_t alignment = 64;
    uint32_t data_offset = 0;
};
#pragma pack(pop)

// --- Helpers ---
template <typename T>
constexpr DType dtype_of();

template <>
constexpr DType dtype_of<float>() { return DType::F32; }

template <>
constexpr DType dtype_of<int32_t>() { return DType::I32; }

uint32_t align_offset(uint32_t offset, uint32_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

// --- Writer ---
class MultiArrayWriter {
    std::ofstream file;
    std::vector<DirectoryEntry> entries;

public:
    MultiArrayWriter(const std::string& filename) {
        file.open(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open file for writing");

        // Reserve space for MainHeader
        MainHeader header;
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));
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

        uint32_t current_offset = static_cast<uint32_t>(file.tellp());
        uint32_t aligned_offset = align_offset(current_offset, alignment);

        if (aligned_offset > current_offset) {
            std::vector<char> padding(aligned_offset - current_offset, 0);
            file.write(padding.data(), padding.size());
        }

        file.write(reinterpret_cast<const char*>(data), size * sizeof(T));

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

    void finalize() {
        uint32_t directory_offset = static_cast<uint32_t>(file.tellp());

        for (const auto& entry : entries) {
            file.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
        }

        MainHeader header{};
        std::copy(std::begin(header.magic), std::end(header.magic), std::begin(header.magic)); // "CTML"
        header.num_arrays = static_cast<uint32_t>(entries.size());
        header.directory_offset = directory_offset;

        file.seekp(0);
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        file.close();
    }
};

// --- Reader ---
struct TensorInfo {
    DType dtype;
    uint32_t size;
    uint32_t ndim;
    uint32_t dims[8];
};

class MultiArrayReader {
    std::ifstream file;
    MainHeader header;
    std::vector<DirectoryEntry> entries;

public:
    MultiArrayReader(const std::string& filename) {
        file.open(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open file");

        file.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (std::string(header.magic, 4) != "CTML")
            throw std::runtime_error("Invalid file magic");

        file.seekg(header.directory_offset);
        entries.resize(header.num_arrays);
        file.read(reinterpret_cast<char*>(entries.data()), header.num_arrays * sizeof(DirectoryEntry));
    }

    const DirectoryEntry* find_entry(const std::string& name) const {
        auto it = std::find_if(entries.begin(), entries.end(), [&](const DirectoryEntry& e) {
            return std::string(e.name, strnlen(e.name, sizeof(e.name))) == name;
        });
        if (it == entries.end()) return nullptr;
        return &(*it);
    }

    template <typename T>
    std::vector<T> read_tensor(const std::string& name, TensorInfo* out_info = nullptr) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, int32_t>,
                      "Only float/int32_t supported");

        const DirectoryEntry* entry = find_entry(name);
        if (!entry) throw std::runtime_error("Array not found: " + name);

        if constexpr (std::is_same_v<T, float>) {
            if (entry->dtype != DType::F32) throw std::runtime_error("Dtype mismatch (expected float)");
        } else if constexpr (std::is_same_v<T, int32_t>) {
            if (entry->dtype != DType::I32) throw std::runtime_error("Dtype mismatch (expected int32)");
        }

        if (out_info) {
            out_info->dtype = entry->dtype;
            out_info->size = entry->size;
            out_info->ndim = entry->ndim;
            std::copy(std::begin(entry->dims), std::begin(entry->dims) + entry->ndim, out_info->dims);
        }

        file.seekg(entry->data_offset);
        std::vector<T> data(entry->size);
        file.read(reinterpret_cast<char*>(data.data()), entry->size * sizeof(T));
        return data;
    }

    std::vector<std::string> list_array_names() const {
        std::vector<std::string> names;
        for (const auto& e : entries) {
            names.emplace_back(e.name, strnlen(e.name, sizeof(e.name)));
        }
        return names;
    }
};

// --- Example usage ---
int main() {
    try {
        MultiArrayWriter writer("tensor_model.ctml");

        // Write a 2x3 float tensor
        float matrix[6] = {1, 2, 3, 4, 5, 6};
        uint32_t shape_matrix[2] = {2, 3};
        writer.write_tensor("matrix", matrix, 6, shape_matrix, 2);

        // Write a 4-element int32 vector
        int32_t vector[4] = {10, 20, 30, 40};
        uint32_t shape_vector[1] = {4};
        writer.write_tensor("vector", vector, 4, shape_vector, 1);

        writer.finalize();

        MultiArrayReader reader("tensor_model.ctml");

        std::cout << "Arrays in file:\n";
        for (const auto& name : reader.list_array_names()) {
            std::cout << " - " << name << "\n";
        }

        TensorInfo info{};
        auto loaded_matrix = reader.read_tensor<float>("matrix", &info);

        std::cout << "Loaded tensor 'matrix' shape: [";
        for (uint32_t i = 0; i < info.ndim; i++)
            std::cout << info.dims[i] << (i + 1 < info.ndim ? ", " : "");
        std::cout << "] data: ";
        for (auto v : loaded_matrix) std::cout << v << " ";
        std::cout << "\n";

        auto loaded_vector = reader.read_tensor<int32_t>("vector", &info);
        std::cout << "Loaded tensor 'vector' shape: [";
        for (uint32_t i = 0; i < info.ndim; i++)
            std::cout << info.dims[i] << (i + 1 < info.ndim ? ", " : "");
        std::cout << "] data: ";
        for (auto v : loaded_vector) std::cout << v << " ";
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
