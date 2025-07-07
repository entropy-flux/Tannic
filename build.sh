rm -rf build
mkdir build && cd build
cmake ..
make
ctest --rerun-failed --output-on-failure