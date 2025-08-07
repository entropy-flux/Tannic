set -e
cmake -S . -B build -DTANNIC_BUILD_MAIN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel "$(nproc)"
./build/main