
class Shape {
    public:

    constexpr Shape() = default;

    constexpr Shape(int a, int b, int c) {
        total += a;
        total += b;
        total += c;
    }

    int total = 0;
};


class Composed {
    public:

    constexpr Composed(int i) {
        shape = Shape(i,i,i);
    }

    Shape shape;
};

int main() {
    constexpr Composed s(3);
    static_assert(s.shape.total == 9);
}