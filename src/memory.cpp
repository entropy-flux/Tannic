#include <fstream>
#include <cstddef>
#include "Resources.hpp"

void* Host::allocate(std::size_t memory)  const { 
    return ::operator new(memory); 
}

void Host::deallocate(void* address, std::size_t size) const { 
    ::operator delete(address);
}

void Host::copy(void* address, void const* value, std::size_t size, Processor processor) const { 
    std::memcpy(address, value, size); 
}

bool Host::compare(void const* address, void const* value, std::size_t size, Processor processor) const { 
    return std::memcmp(address, value, size) == 0; 
}

unsigned long long Host::available() const {
    #ifdef __linux__
        std::ifstream meminfo("/proc/meminfo");
        if (!meminfo.is_open()) {
            throw std::runtime_error("Failed to open /proc/meminfo");
        }

        std::string line;
        unsigned long long available_kb = 0;

        while (std::getline(meminfo, line)) {
            if (line.find("MemAvailable:") == 0) {
                sscanf(line.c_str(), "MemAvailable: %llu kB", &available_kb);
                break;
            }
        }

        if (available_kb == 0) {
            throw std::runtime_error("MemAvailable not found in /proc/meminfo");
        }

        return available_kb * 1024ULL;  // Return bytes
    #else
        throw std::runtime_error("Unsupported OS for Memory::available()");
    #endif
}