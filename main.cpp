#include "mcls.hpp"

int main() {
    MyClass obj;
    obj.templateFunction(42);       // Uses int version
    obj.templateFunction(3.14);     // Uses double version
    obj.regularFunction();
    return 0;
}