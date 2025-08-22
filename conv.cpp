// Copyright 2025 Eric Hermosis
//
// This file is part of the Tannic Tensor Library.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
  
#include <cstdint>
#include <cstddef> 

#define DIMENSIONS 8

struct shape_t {  
    size_t sizes[DIMENSIONS]; 
};

struct strides_t { 
    int64_t sizes[DIMENSIONS]; 
};

struct tensor_t {
    void* address;
    uint8_t rank;
    struct shape_t shape;
    struct strides_t strides;   
};    
  
inline int64_t calc_output_dim(int64_t input_dim, int64_t kernel_dim, int64_t pad, int64_t stride) {
    return (input_dim + 2 * pad - kernel_dim) / stride + 1;
}
 
int conv2d_naive_f32(
    const tensor_t* src,
    const tensor_t* ker,
    tensor_t* dst,
    const int64_t padding[2],
    const int64_t stride[2]
) { 
    if (!src || !ker || !dst || !src->address || !ker->address || !dst->address) {
        return -1; // Null pointer check
    }
    if (src->rank != 4 || ker->rank != 4 || dst->rank != 4) {
        return -1; // Only handle 4D tensors
    }

    // --- 2. Unpack shapes for readability ---
    // Input shape: [N, iC, iH, iW]
    const int64_t N = src->shape.sizes[0];
    const int64_t iC = src->shape.sizes[1];
    const int64_t iH = src->shape.sizes[2];
    const int64_t iW = src->shape.sizes[3];

    // Kernel shape: [oC, iC, kH, kW]
    const int64_t oC = ker->shape.sizes[0];
    const int64_t kC = ker->shape.sizes[1]; // This should equal iC
    const int64_t kH = ker->shape.sizes[2];
    const int64_t kW = ker->shape.sizes[3];

    // Output shape: [N, oC, oH, oW]
    const int64_t dst_N = dst->shape.sizes[0];
    const int64_t dst_oC = dst->shape.sizes[1];
    const int64_t oH = dst->shape.sizes[2];
    const int64_t oW = dst->shape.sizes[3];

    // --- 3. Validate shapes and parameters ---
    if (iC != kC) {
        return -1; // Input channels must match
    }
    if (N != dst_N || oC != dst_oC) {
        return -1; // Batch and output channel count must match
    }

    // Check if the calculated output size matches the provided dst tensor
    int64_t expected_oH = calc_output_dim(iH, kH, padding[0], stride[0]);
    int64_t expected_oW = calc_output_dim(iW, kW, padding[1], stride[1]);
    if (oH != expected_oH || oW != expected_oW) {
        return -1; // Output tensor has wrong dimensions
    }

    const int64_t padH = padding[0];
    const int64_t padW = padding[1];
    const int64_t strideH = stride[0];
    const int64_t strideW = stride[1];

    // --- 4. Get pointers to the actual data ---
    const float* __restrict src_data = (const float*)src->address;
    const float* __restrict ker_data = (const float*)ker->address;
    float* __restrict dst_data = (float*)dst->address;

    // --- 5. The Naive Convolution Loops ---
    // This is the direct 7-loop implementation.
    for (int64_t n = 0; n < N; ++n) {           // Loop over Batch
        for (int64_t oc = 0; oc < oC; ++oc) {    // Loop over Output Channels
            for (int64_t oh = 0; oh < oH; ++oh) { // Loop over Output Height
                for (int64_t ow = 0; ow < oW; ++ow) { // Loop over Output Width

                    // Initialize the output value for this pixel
                    float value = 0.0f;

                    // Calculate the starting point in the input for this output pixel
                    int64_t ih_start = oh * strideH - padH;
                    int64_t iw_start = ow * strideW - padW;

                    // Loop over Input Channels and Kernel spatial dimensions
                    for (int64_t ic = 0; ic < iC; ++ic) {       // Loop over Input Channels
                        for (int64_t kh = 0; kh < kH; ++kh) {    // Loop over Kernel Height
                            for (int64_t kw = 0; kw < kW; ++kw) { // Loop over Kernel Width

                                // Calculate the corresponding input location, considering padding
                                int64_t ih = ih_start + kh;
                                int64_t iw = iw_start + kw;

                                // Check if the input location is in-bounds (not in the padded area)
                                if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
                                    // Calculate the linear indices for src, ker, and dst
                                    // (This is where your strides would be crucial for non-contiguous or non-NCHW tensors)
                                    // For a contiguous NCHW tensor, the offset calculation is:
                                    size_t src_idx = n * (iC * iH * iW) + ic * (iH * iW) + ih * iW + iw;
                                    size_t ker_idx = oc * (iC * kH * kW) + ic * (kH * kW) + kh * kW + kw;

                                    // Perform the multiplication and accumulation (MAC)
                                    value += src_data[src_idx] * ker_data[ker_idx];
                                } // else, the value is treated as 0 (padding)
                            }
                        }
                    } // End of inner accumulation loops

                    // Calculate the linear index for the output and store the result
                    size_t dst_idx = n * (oC * oH * oW) + oc * (oH * oW) + oh * oW + ow;
                    dst_data[dst_idx] = value;
                }
            }
        }
    }

    return 0; // Success
}

// main.cpp
#include <iostream>
#include <vector>
#include <cstdlib> // for rand 

// Your provided function declaration
int conv2d_naive_f32(
    const tensor_t* src,
    const tensor_t* ker,
    tensor_t* dst,
    const int64_t padding[2],
    const int64_t stride[2]
);

// Helper function to create a contiguous NCHW tensor_t
tensor_t create_tensor(void* data, uint8_t rank, const std::vector<size_t>& shape_sizes) {
    tensor_t tensor;
    tensor.address = data;
    tensor.rank = rank;
    
    // Initialize shape
    for (int i = 0; i < rank; ++i) {
        tensor.shape.sizes[i] = shape_sizes[i];
    }
    
    // Initialize strides for a contiguous NCHW tensor
    // For shape [N, C, H, W], strides are [C*H*W, H*W, W, 1]
    int64_t stride = 1;
    for (int i = rank - 1; i >= 0; --i) {
        tensor.strides.sizes[i] = stride;
        stride *= shape_sizes[i];
    }
    
    return tensor;
}

// Helper function to print a small tensor (for debugging)
void print_tensor(const char* name, const float* data, const std::vector<size_t>& shape) {
    std::cout << name << ":\n";
    int64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
    
    for (int64_t ni = 0; ni < n; ++ni) {
        for (int64_t ci = 0; ci < c; ++ci) {
            std::cout << "n=" << ni << ", c=" << ci << ":\n";
            for (int64_t hi = 0; hi < h; ++hi) {
                for (int64_t wi = 0; wi < w; ++wi) {
                    size_t idx = ni*(c*h*w) + ci*(h*w) + hi*w + wi;
                    std::cout << data[idx] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
    std::cout << "----------------\n";
}

int main() {
    std::cout << "Testing conv2d_naive_f32...\n";
    
    // Define tensor dimensions (keep them small for testing)
    const int64_t N = 1;  // Batch size
    const int64_t iC = 1; // Input channels
    const int64_t iH = 4; // Input height
    const int64_t iW = 4; // Input width
    
    const int64_t oC = 1; // Output channels
    const int64_t kH = 3; // Kernel height
    const int64_t kW = 3; // Kernel width
    
    // Calculate output dimensions
    int64_t padding[2] = {0, 0};
    int64_t stride[2] = {1, 1};
    size_t oH = calc_output_dim(iH, kH, padding[0], stride[0]);
    size_t oW = calc_output_dim(iW, kW, padding[1], stride[1]);
    
    std::cout << "Output dimensions: " << oH << "x" << oW << "\n";
    
    // Create and initialize data
    std::vector<float> input_data(N * iC * iH * iW);
    std::vector<float> kernel_data(oC * iC * kH * kW);
    std::vector<float> output_data(N * oC * oH * oW, 0.0f); // Initialize to 0
    
    // Fill with simple values for easy verification
    // Input: a simple 4x4 matrix with increasing values
    for (int64_t i = 0; i < iH * iW; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }
    
    // Kernel: identity-like kernel (center is 1, others are 0)
    // This should extract the top-left 2x2 corner of the input
    for (int64_t i = 0; i < kH * kW; ++i) {
        kernel_data[i] = 0.0f;
    }
    kernel_data[4] = 1.0f; // Center of 3x3 kernel
    
    // Create tensor objects
    tensor_t input = create_tensor(input_data.data(), 4, {N, iC, iH, iW});
    tensor_t kernel = create_tensor(kernel_data.data(), 4, {oC, iC, kH, kW});
    tensor_t output = create_tensor(output_data.data(), 4, {N, oC, oH, oW});
    
    // Print initial data
    print_tensor("Input", input_data.data(), {N, iC, iH, iW});
    print_tensor("Kernel", kernel_data.data(), {oC, iC, kH, kW});
    
    // Run convolution
    int result = conv2d_naive_f32(&input, &kernel, &output, padding, stride);
    
    if (result != 0) {
        std::cerr << "Convolution failed with error: " << result << "\n";
        return 1;
    }
    
    // Print results
    print_tensor("Output", output_data.data(), {N, oC, oH, oW});
    
    std::cout << "Convolution completed successfully!\n";
    return 0;
}