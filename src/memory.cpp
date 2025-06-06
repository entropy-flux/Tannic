#include <fstream>
#include <cstddef>
#include "Memory/Resources.hpp"

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