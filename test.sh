cd tests
rm -rf build
mkdir -p build
cd build || exit 1
 
cmake .. -DTANNIC_BUILD_DIR=../../build
make
 
ctest --output-on-failure