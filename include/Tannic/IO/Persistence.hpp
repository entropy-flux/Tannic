// Copyright 2025 Eric Cardozo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// This file is part of Tannic, a machine learning tensor library for C++.

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