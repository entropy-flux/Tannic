g++ -std=c++23 -Iinclude -Ibuild example.cpp -Lbuild -ltannic -lopenblas -o example
./example
rm example