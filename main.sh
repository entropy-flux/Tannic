mkdir -p build && cd build 
cmake -DTANNIC_BUILD_MAIN=ON ..
cmake --build . 
./main