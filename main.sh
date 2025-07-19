g++ -std=c++23 -Iinclude -Ibuild main.cpp -Lbuild -ltannic -lopenblas -o main
./main
rm main