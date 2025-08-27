#include <iostream>
#include <tannic.hpp>
#include <tannic/transformations.hpp>
#include <tannic/convolutions.hpp>

using namespace tannic; 

int main() { 
    Tensor X = {{
        { // batch 0
            { // channel 0
                {1.0f,  2.0f,  3.0f,  4.0f},
                {5.0f,  6.0f,  7.0f,  8.0f},
                {9.0f, 10.0f, 11.0f, 12.0f},
                {13.0f,14.0f, 15.0f, 16.0f}
            },
            { // channel 1
                {17.0f, 18.0f, 19.0f, 20.0f},
                {21.0f, 22.0f, 23.0f, 24.0f},
                {25.0f, 26.0f, 27.0f, 28.0f},
                {29.0f, 30.0f, 31.0f, 32.0f}
            },
            { // channel 2
                {33.0f, 34.0f, 35.0f, 36.0f},
                {37.0f, 38.0f, 39.0f, 40.0f},
                {41.0f, 42.0f, 43.0f, 44.0f},
                {45.0f, 46.0f, 47.0f, 48.0f}
            }
        },
        { // batch 1
            { // channel 0
                {49.0f, 50.0f, 51.0f, 52.0f},
                {53.0f, 54.0f, 55.0f, 56.0f},
                {57.0f, 58.0f, 59.0f, 60.0f},
                {61.0f, 62.0f, 63.0f, 64.0f}
            },
            { // channel 1
                {65.0f, 66.0f, 67.0f, 68.0f},
                {69.0f, 70.0f, 71.0f, 72.0f},
                {73.0f, 74.0f, 75.0f, 76.0f},
                {77.0f, 78.0f, 79.0f, 80.0f}
            },
            { // channel 2
                {81.0f, 82.0f, 83.0f, 84.0f},
                {85.0f, 86.0f, 87.0f, 88.0f},
                {89.0f, 90.0f, 91.0f, 92.0f},
                {93.0f, 94.0f, 95.0f, 96.0f}
            }
        }
    }};

  Tensor K = {{
      { // in channel 0
          {1.0f,  0.0f, -1.0f},
          {1.0f,  0.0f, -1.0f},
          {1.0f,  0.0f, -1.0f}
      },
      { // in channel 1
          {1.0f,  0.0f, -1.0f},
          {1.0f,  0.0f, -1.0f},
          {1.0f,  0.0f, -1.0f}
      },
      { // in channel 2
          {1.0f,  0.0f, -1.0f},
          {1.0f,  0.0f, -1.0f},
          {1.0f,  0.0f, -1.0f}
      }
  }};

  
    Tensor b = {0.5f};
 
    auto Y = convolve2D(X , K, /*stride=*/1, /*padding=*/1) ;

    std::cout << Y << std::endl;
    return 0;
}
