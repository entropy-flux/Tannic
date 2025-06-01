#include "include/Types.hpp"
#include "include/IO/Blobs.hpp"
#include "include/IO/Serialization.hpp"
#include "include/IO/Persistence.hpp" 

int main() {
    // Step 1: Create tensor
    Tensor tensor({2, 3}, float32);
    float* data_ptr = static_cast<float*>(tensor.address());
    for (int i = 0; i < 6; ++i) {
        data_ptr[i] = float(i + 1);  // Fill with 1.0 to 6.0
    }

    std::cout << "Original Tensor:\n" << tensor << "\n";

    // Step 2: Serialize tensor to a Blob in memory
    Blob serialized = serialize(tensor, 32);

    std::cout << "Serialized Blob Bytes:\n" << serialized << "\n";

    // Step 3: Save to disk
    const std::string path = "tensor.dat";
    write(serialized, "tensor.dat", 32);
    std::cout << "Tensor saved to file: " << path << "\n";

    // Step 4: Load from disk

    Blob loaded = read("tensor.dat");
    Tensor deserialized = deserialize(loaded);

    // Step 5: Display deserialized tensor
    std::cout << "Deserialized Tensor from file:\n";
    std::cout << "  DType: " << deserialized.dtype() << "\n";
    std::cout << "  Shape: " << deserialized.shape() << "\n";
    std::cout << "  Strides: " << deserialized.strides() << "\n";
    std::cout << "  Data: " << deserialized << "\n";

    return 0;
}
