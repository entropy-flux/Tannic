#ifndef PERSISTENCE_HPP
#define PERSISTENCE_HPP

#include <fstream>
#include "IO/Blobs.hpp"

inline void write(const Blob& blob, const std::string& path, uint32_t alignment = 0 ) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream) 
        throw std::runtime_error("Failed to open file for writing: " + path);
    
        stream.write(reinterpret_cast<const char*>(blob.bytes()), blob.size());
}

inline Blob read(const std::string& path, Allocator allocator = Host{}) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) 
        throw std::runtime_error("Failed to open file for reading: " + path);
    
    std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);

    Blob blob(size, allocator); 
    stream.read(reinterpret_cast<char*>(blob.bytes()), size);
    if (!stream) 
        throw std::runtime_error("Failed to read from file: " + path);

    return blob;
}

#endif // PERSISTENCE_HPP