#ifndef MYCLASS_H
#define MYCLASS_H

class MyClass {
public:
    template <typename T>
    void templateFunction(T value);

    void regularFunction();
};

// Explicit template instantiation declarations
extern template void MyClass::templateFunction<int>(int);
extern template void MyClass::templateFunction<double>(double);

#endif // MYCLASS_H