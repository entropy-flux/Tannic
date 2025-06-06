#include <cblas.h>
#include "kernels/openblas/scal.h"

namespace openblas::scal {

void sscal(
    int n,
    double alpha,
    void* x, int incx
) {
    cblas_sscal(
        n,
        static_cast<float>(alpha),
        static_cast<float*>(x), incx
    );
}

void dscal(
    int n,
    double alpha,
    void* x, int incx
) {
    cblas_dscal(
        n,
        alpha,
        static_cast<double*>(x), incx
    );
}

} // namespace openblas::scal