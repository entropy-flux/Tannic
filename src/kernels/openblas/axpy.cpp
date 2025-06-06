#include <cblas.h>
#include "kernels/openblas/axpy.h"

namespace openblas::axpy {

void saxpy(
    int n,
    double alpha,
    const void* x, int incx,
    void* y, int incy
) {
    cblas_saxpy(
        n,
        alpha,
        static_cast<const float*>(x), incx,
        static_cast<float*>(y), incy
    );
}

void daxpy(
    int n,
    double alpha,
    const void* x, int incx,
    void* y, int incy
) {
    cblas_daxpy(
        n,
        alpha,
        static_cast<const double*>(x), incx,
        static_cast<double*>(y), incy
    );
}

} // namespace openblas::axpy
