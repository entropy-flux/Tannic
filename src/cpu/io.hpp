#include "runtime/tensor.h"
#include <ostream>

using namespace tannic;

namespace io {

void print(std::ostream&, const shape_t&, uint8_t rank);
void print(std::ostream&, const strides_t&, uint8_t rank);
void print(std::ostream&, const tensor_t*);

};