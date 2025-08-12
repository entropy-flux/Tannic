#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <vector>
#include <cstdlib> // for malloc/free
#include <new>     // for placement new

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <vector>

struct tensor_t {
    int size;
};

struct node_t {
    uint8_t arity; 
    node_t** priors; 
    tensor_t* target;   
};

class Tensor; // forward declaration

struct Node {
    uintptr_t id = 0;

    explicit Node(const Tensor& target);
    Node(const Tensor& target, std::span<const Node*> sources);

    ~Node() {
        if (id) {
            node_t* node = reinterpret_cast<node_t*>(id);
            delete node->target;     // free the tensor_t
            delete[] node->priors;   // free array of node_t*
            delete node;             // free the node itself
            id = 0;
        }
    }

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    Node(Node&& other) noexcept : id(other.id) {
        other.id = 0;
    }

    Node& operator=(Node&& other) noexcept {
        if (this != &other) {
            if (id) {
                node_t* node = reinterpret_cast<node_t*>(id);
                delete node->target;
                delete[] node->priors;
                delete node;
            }
            id = other.id;
            other.id = 0;
        }
        return *this;
    }

    node_t* get() const {
        return reinterpret_cast<node_t*>(id);
    }
};

class Tensor {
public:
    explicit Tensor(int size) : tensor_{size} {}

    void attachNode() {
        node_ = std::make_shared<Node>(*this);
    }

    void attachNodeWithSources(std::span<const Node*> sources) {
        node_ = std::make_shared<Node>(*this, sources);
    }

    int size() const { return tensor_.size; }

    tensor_t* get_c_tensor_ptr() { return &tensor_; } 
    const tensor_t* get_c_tensor_ptr() const { return &tensor_; }

    std::shared_ptr<Node> node() const { return node_; }

private:
    tensor_t tensor_;
    std::shared_ptr<Node> node_;
};
 
Node::Node(const Tensor& target) {
    node_t* node = new node_t;
    node->target = new tensor_t{target.size()}; 
    node->priors = nullptr;
    node->arity = 0;
    id = reinterpret_cast<uintptr_t>(node);
}
 
Node::Node(const Tensor& target, std::span<const Node*> sources) {
    node_t* node = new node_t;
    node->target = new tensor_t{target.size()};
    node->arity = sources.size();

    if (!sources.empty()) {
        node->priors = new node_t*[sources.size()];
        for (size_t i = 0; i < sources.size(); ++i) {
            node->priors[i] = sources[i]->get();
        }
    } else {
        node->priors = nullptr;
    }

    id = reinterpret_cast<uintptr_t>(node);
} 

int main() {
    Tensor t1(42), t2(7), t3(3);

    t2.attachNode();
    t3.attachNode();

    const Node* sources[] = { t2.node().get(), t3.node().get() };
    t1.attachNodeWithSources(sources);

    if (auto n = t1.node()) {
        std::cout << "Target size = " << n->get()->target->size << "\n";
        for (size_t i = 0; i < n->get()->arity; ++i) {
            std::cout << "Source " << i << " target size = "
                      << n->get()->priors[i]->target->size << "\n";
        }
    }
}
