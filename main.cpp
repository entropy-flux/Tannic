#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>

void* allocate_bool_storage(size_t num_bools) {
    size_t num_bytes = (num_bools + 7) / 8;  // round up
    void* ptr = std::calloc(num_bytes, 1);   // zero-initialized
    return ptr;
}

inline bool get_bit(const void* base, size_t index) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(base);
    return (data[index >> 3] >> (index & 7)) & 1;
}

inline void set_bit(void* base, size_t index, bool value) {
    uint8_t* data = reinterpret_cast<uint8_t*>(base);
    if (value)
        data[index >> 3] |=  (1u << (index & 7));
    else
        data[index >> 3] &= ~(1u << (index & 7));
}
 
#include <iostream>

int main() {
    size_t N = 20;
    void* storage = allocate_bool_storage(N);

    // set a few values
    set_bit(storage, 3, true);
    set_bit(storage, 5, true);
    set_bit(storage, 19, true);

    // check them
    for (size_t i = 0; i < N; i++) {
        std::cout << i << ": " << get_bit(storage, i) << "\n";
    }

    std::free(storage);
}
