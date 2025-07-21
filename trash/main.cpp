#include <tannic.hpp>
#include <string> 

using namespace tannic; 


#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

struct TensorMetadata {
    std::string name;
    uint8_t dtype_code;
    uint8_t rank;
    std::vector<uint64_t> shape;
    std::vector<uint64_t> strides;
    uint64_t offset;
    uint64_t nbytes;
};
 

std::vector<TensorMetadata> read_tnnc_metadata(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open metadata file.");

    // Read header
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "TANNIC")
        throw std::runtime_error("Invalid file header.");
    
    // Skip padding (26 bytes)
    file.ignore(32 - 6);

    std::vector<TensorMetadata> tensors;

    while (file.peek() != EOF) {
        TensorMetadata meta;

        // Name length and name
        uint8_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), 1);

        meta.name.resize(name_len);
        file.read(meta.name.data(), name_len);

        // Dtype code and rank
        file.read(reinterpret_cast<char*>(&meta.dtype_code), 1);
        file.read(reinterpret_cast<char*>(&meta.rank), 1);

        // Shape and strides
        meta.shape.resize(meta.rank);
        meta.strides.resize(meta.rank);
        for (int i = 0; i < meta.rank; ++i) {
            file.read(reinterpret_cast<char*>(&meta.shape[i]), sizeof(uint64_t));
            file.read(reinterpret_cast<char*>(&meta.strides[i]), sizeof(uint64_t));
        }

        // Offset and nbytes
        file.read(reinterpret_cast<char*>(&meta.offset), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&meta.nbytes), sizeof(uint64_t));

        tensors.push_back(std::move(meta));
    }

    return tensors;
}

/*
struct TensorMetadata {
    std::string name;
    uint64_t offset;
    uint64_t nbytes;
    uint8_t dtype_code;
    uint8_t rank;
    std::vector<uint64_t> shape;
    std::vector<uint64_t> strides; 
};



    auto& instance = Parameters::instance();  // access singleton

    while (metadata.peek() != EOF) {
        uint8_t name_length = 0;
        metadata.read(reinterpret_cast<char*>(&name_length), 1);

        std::string name(name_length, '\0');
        metadata.read(name.data(), name_length);

        uint64_t offset = 0;
        uint64_t nbytes = 0;
        metadata.read(reinterpret_cast<char*>(&offset), sizeof(uint64_t));
        metadata.read(reinterpret_cast<char*>(&nbytes), sizeof(uint64_t));

        // Store in map (cast offset to ptrdiff_t)
        instance.offsets_.emplace(std::move(name), static_cast<std::ptrdiff_t>(offset));


*/
 


#include <unordered_map>

class Parameters {
public:
    static Parameters& instance() {
        static Parameters instance;
        return instance;
    }

    static void initialize(std::string const& filename) {       
        std::ifstream metadata(filename + ".metadata.tnnc", std::ios::binary);
        if (!metadata)
            throw std::runtime_error("Could not open metadata file.");

        char magic[6]; metadata.read(magic, 6);
        if (std::string(magic, 6) != "TANNIC")
            throw std::runtime_error("Invalid magic header");    
        
        metadata.ignore(32 - 6);
        auto& instance = Parameters::instance(); 
        while (metadata.peek() != EOF) {
            uint8_t namelength; metadata.read(reinterpret_cast<char*>(&namelength), 1); 
            std::string name(namelength, '\0'); metadata.read(name.data(), namelength);    
            std::size_t offset = 0; metadata.read(reinterpret_cast<char*>(&offset), sizeof(size_t));
            std::size_t nbytes = 0; metadata.read(reinterpret_cast<char*>(&nbytes), sizeof(size_t));  
            instance.offsets_[std::move(name)] = static_cast<std::ptrdiff_t>(offset);            

            uint8_t type = 0; metadata.read(reinterpret_cast<char*>(&type), 1);
            uint8_t rank = 0; metadata.read(reinterpret_cast<char*>(&rank), 1); 
            metadata.ignore(static_cast<std::streamsize>(rank) * 2 * sizeof(size_t));
        }
    } 

private:
    Parameters() = default;
    ~Parameters() = default;
    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete; 
    std::shared_ptr<Storage> storage_ = nullptr;
    std::unordered_map<std::string, std::ptrdiff_t> offsets_;
};   

int main() {
    try {
        auto metadata = read_tnnc_metadata("./data/mlp.metadata.tnnc");
        for (const auto& tensor : metadata) {
            std::cout << "Tensor: " << tensor.name << "\n";
            std::cout << "  Dtype Code: " << static_cast<int>(tensor.dtype_code) << "\n";
            std::cout << "  Rank: " << static_cast<int>(tensor.rank) << "\n";
            std::cout << "  Shape: ";
            for (auto s : tensor.shape) std::cout << s << " ";
            std::cout << "\n  Strides: ";
            for (auto s : tensor.strides) std::cout << s << " ";
            std::cout << "\n  Offset: " << tensor.offset;
            std::cout << "\n  Nbytes: " << tensor.nbytes << "\n\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
