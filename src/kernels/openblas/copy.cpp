#include <cblas.h>
#include "kernels/openblas/copy.h"

namespace openblas::copy {

void scopy(
    int n, 
    const void* x, 
    int incx, 
    void* y, 
    int incy
) {
    cblas_scopy(
        n, 
        static_cast<const float*>(x), incx, 
        static_cast<float*>(y), incy
    );
}

void dcopy(
    int n, 
    const void* x, int incx,
    void* y, int incy
) {
    cblas_dcopy(
        n, 
        static_cast<const double*>(x), incx, 
        static_cast<double*>(y), incy
    );
}

} // namespace openblas::copy