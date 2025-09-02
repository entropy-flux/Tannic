#pragma once

#include "resources.hpp" 
#include "tensor.hpp"
#include "runtime/tensor.h"
#include "runtime/resources.h" 
#include "runtime/status.h" 

/*
NOTE: All this functions may change and some of them will go to the runtime api when
the backend is more stable.
*/

namespace tannic {

    // OLDS FUNCTIONS TO BE REMOVED.
tensor_t structure(Tensor const&);
host_t structure(Host const&);
device_t structure(Device const&);
environment_t structure(Environment const&);    

}