#include <iostream>
#include <utility> 
#include <list>
#include <functional> 
#include <stack>
#include <cassert>
#include <vector>
#include <memory>
#include <string>
#include <tannic.hpp>

using namespace std;
using namespace tannic;
 
class Parameter {
public:
    using size_type = size_t;

    Parameter(type dtype, Shape shape, const string& name = "")
        : dtype_(dtype), shape_(shape), strides_(shape) {
        register_self(name);
    }

    Parameter(type dtype, Shape shape, Strides strides, const string& name = "")
        : dtype_(dtype), shape_(shape), strides_(strides) {
        register_self(name);
    }

    void initialize(Allocator allocator) const {
        storage_ = make_shared<Storage>(nbytes(), allocator);
    }

    Tensor forward() const {
        return Tensor(dtype_, shape_, strides_, 0, storage_);
    }

private:
    void register_self(const string& name);

    type dtype_;
    Shape shape_;
    Strides strides_;
    mutable shared_ptr<Storage> storage_ = nullptr;

    size_t nbytes() const {
        return shape_.size() * sizeof(float); // Assuming float for float32
    }
};

// ==== Parameters Context (Singleton Stack) ====
class Parameters {
public:
    static Parameters& current() {
        assert(!context_stack_.empty());
        return *context_stack_.top();
    }

    static void push(shared_ptr<Parameters> context) {
        context_stack_.push(move(context));
    }

    static void pop() {
        assert(!context_stack_.empty());
        context_stack_.pop();
    }

    void register_parameter(const string& name, shared_ptr<Parameter> param) {
        parameters_.emplace_back(name, param);
    }

    void print_parameters() const {
        for (const auto& [name, _] : parameters_) {
            cout << "Parameter: " << name << "\n";
        }
    }
 

private:
    static inline stack<shared_ptr<Parameters>> context_stack_ = [] {
        stack<shared_ptr<Parameters>> stack;
        stack.push(make_shared<Parameters>());  // Global default
        return stack;
    }();

    vector<pair<string, shared_ptr<Parameter>>> parameters_;
};

void Parameter::register_self(const string& name) {
    if (!name.empty()) {
        Parameters::current().register_parameter(name, make_shared<Parameter>(*this));
    }
}

// ==== RAII Scoped Context ====
class Context {
public:
    Context() {
        Parameters::push(make_shared<Parameters>());
    }
    ~Context() {
        Parameters::pop();
    }
};

// ==== Module Base ====
struct Module {
    using size_type = size_t;
    template<typename Self, typename... Args>
    auto operator()(this Self&& self, Args&&... args) -> decltype(auto) {
        return forward<Self>(self).forward(forward<Args>(args)...);
    }
};

// ==== Linear Module ====
struct Linear : Module {
    Parameter weight;
    Parameter bias;

    Linear(type dtype, size_type in, size_type out)
        : weight(dtype, Shape{in, out}, "weight"),
          bias(dtype, Shape{out}, "bias") {}

    void initialize(Allocator allocator = Host{}) {
        weight.initialize(allocator);
        bias.initialize(allocator);
    }

    Tensor forward(Tensor input) {
        return matmul(input, transpose(weight.forward(), -1, -2)) + bias.forward();
    }
};
 

// ==== Main ====
int main() { 
    { 
        Context ctx; // Scoped parameter collection 
        Linear linear(type::float32, 5, 5);
        linear.initialize(Host{}); 
        Tensor input(type::float32, {5, 5}); input.initialize(); 
        Tensor output = linear(input);
        cout << "Forward completed.\n"; 
        Parameters::current().print_parameters();  // Should list "weight", "bias"
    } 
    Parameters::current().print_parameters(); 
 
}