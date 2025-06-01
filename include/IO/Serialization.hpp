#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include "Tensor.hpp"
#include "IO/Blobs.hpp"

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
    Storage storage(metadata.rank, dsizeof(metadata.dtype), View{address});  

    std::span<Shape::size_type> shape(metadata.shape, metadata.rank);
    std::span<Strides::step_type> strides(metadata.strides, metadata.rank); 
    return Tensor(storage, Shape(shape.begin(), shape.end()), Strides(strides.begin(), strides.end()), metadata.dtype, 0);
} 


#endif // SERIALIZATION_HPP
