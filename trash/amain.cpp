#include <cstddef>
#include <vector>
#include <span>
#include <type_traits> 
#include <array>
#include <cstdint>
#include <cassert>
#include <ostream>
#include <algorithm>
#include <numeric>
#include <variant>
#include <atomic>
#include <cstddef>
#include <vector>
#include <utility>



#include <iostream>
#include <cstdint>
#include <stdexcept>    
#include <any>
#include <cassert>

enum type : uint8_t {
    any,
    integer8,
    integer16,
    integer32,
    integer64,
    float32,
    float64,
    TYPES
};

struct Traits {
    using size_type = size_t;
    using retrieve_function = std::any (*)(void const*); 
    using assign_function = void(*)(void*, std::any const&);
    using compare_function = bool(*)(void const*, std::any const&);
    using print_function = std::ostream&(*)(std::ostream&, void const*); 

    const char* name;
    size_type size;
    retrieve_function retrieve;
    assign_function assign;
    compare_function compare;
    print_function print; 
};


template<typename T>
inline void assign(void* address, std::any const& value) {
    *reinterpret_cast<T*>(address) = std::any_cast<T>(value);
}


template<typename T>
inline bool compare(void const* address, std::any const& value) {
    return *reinterpret_cast<T const*>(address) == std::any_cast<T>(value);
}


template<typename T>
inline std::any retrieve(void const* address) {
    return *reinterpret_cast<T const*>(address); 
}


template<typename T>
inline std::ostream& print(std::ostream& os, void const* address) {
    return os << *reinterpret_cast<T const*>(address);
}


template<>
inline std::ostream& print<int8_t>(std::ostream& os, void const* address) {
    return os << +(*reinterpret_cast<int8_t const*>(address));
}

static constexpr Traits traits[TYPES] = {
    [any] = {
        .name = "any",
        .size = 0,
        .retrieve = nullptr,
        .assign = nullptr,
        .compare = nullptr,
        .print = nullptr 
    },

    
    [integer8] = {
        .name = "integer8",
        .size = sizeof(int8_t),
        .retrieve = retrieve<int8_t>,
        .assign = assign<int8_t>,
        .compare = compare<int8_t>,
        .print = print<int8_t>  
    },

    [integer16] = {
        .name = "integer16",
        .size = sizeof(int16_t),
        .retrieve = retrieve<int16_t>,
        .assign = assign<int16_t>,
        .compare = compare<int16_t>,
        .print = print<int16_t>
    },
    
    [integer32] = {
        .name = "integer32",
        .size = sizeof(int32_t),
        .retrieve = retrieve<int32_t>,
        .assign = assign<int32_t>,
        .compare = compare<int32_t>,
        .print = print<int32_t>
    },
    
    [integer64] = {
        .name = "integer32",
        .size = sizeof(int64_t),
        .retrieve = retrieve<int64_t>,
        .assign = assign<int64_t>,
        .compare = compare<int64_t>,
        .print = print<int64_t>
    },
    
    [float32] = {
        .name = "float32",
        .size = sizeof(float),
        .retrieve = retrieve<float>,
        .assign = assign<float>,
        .compare = compare<float>,
        .print = print<float>
    },

    
    [float64] = {
        .name = "float64",
        .size = sizeof(double),
        .retrieve = retrieve<double>,
        .assign = assign<double>,
        .compare = compare<double>,
        .print = print<double>
    },

};


template<typename T>
inline T dcast(std::any& retrieved) {
    return std::any_cast<T>(retrieved);
}


inline constexpr size_t dsizeof(type type) {
    return traits[type].size;
}


inline std::ostream& operator<<(std::ostream& os, const type type) {
    assert(type < TYPES && "Invalid type");
    os << traits[type].name;
    return os;
}


inline std::ostream& operator<<(std::ostream& os, uint8_t value) {
    return os << static_cast<unsigned int>(value);
}


constexpr type promote(type first, type second) {
    assert(first < TYPES && second < TYPES && "Invalid type");
    if (first != second) 
        throw std::runtime_error("Type promotion rules not implemented yet");
    return first;
} 


enum unit : std::size_t {
    B = 1,
    KB = 1024,
    MB = 1024 * 1024,
    GB = 1024 * 1024 * 1024,
    UNIT
};

struct Resource{};

struct Host : Resource {  
    void* allocate(std::size_t memory) const { return ::operator new(memory); }
    void deallocate(void* address, std::size_t size) const { ::operator delete(address); }
    unsigned long long available() const;
};

struct Device : Resource { 
    Device(int id) : id(id) {}
    int id;
    void* allocate(std::size_t memory);               
    void deallocate(void* address, std::size_t size); 
    unsigned long long available() const;
};

class Resources {
public:
    Resources();
    static Host host() { return Host{}; } 
    std::span<const Device> devices() const {
        return std::span<const Device>(devices_.data(), devices_.size());
    }

private:
    std::vector<Device> devices_;
};


 

template<typename T>
concept Iterable = requires(T type) {
    std::begin(type);
    std::end(type);
};

class Shape {
public:
    using index_type = int8_t;
    using rank_type = uint8_t;
    using size_type = std::size_t;
    static constexpr uint8_t limit = 8;  

    constexpr Shape() noexcept = default;

    template<typename... Sizes>
    constexpr explicit Shape(Sizes... sizes) 
    :   sizes_{static_cast<size_type>(sizes)...}
    ,   rank_(sizeof...(sizes)) {      
        if (sizeof...(sizes) > limit) {
            throw "Rank limit exceeded";
        }
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    }

    template<Iterable Sizes>
    constexpr explicit Shape(Sizes&& sizes) {
        size_type dimension = 0;
        for (auto size : sizes) {
            sizes_[dimension++] = static_cast<size_type>(size);
        }
        
        if (dimension >= limit) {
            throw "Rank limit exceeded";
        }
        rank_ = dimension;
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    } 


    template<std::input_iterator Iterator>
    constexpr Shape(Iterator begin, Iterator end) {
        size_type dimension = 0;
        for (Iterator iterator = begin; iterator != end; ++iterator) {
            if (dimension >= limit) {
                throw "Rank limit exceeded";
            }
            sizes_[dimension++] = static_cast<size_type>(*iterator);
        }
        rank_ = dimension;
        size_ = std::accumulate(sizes_.begin(), sizes_.begin() + rank_, size_type{1}, std::multiplies<size_type>()); 
    }

    constexpr rank_type rank() const noexcept { return rank_; }
    constexpr size_type operator[](index_type dimension) const noexcept { return sizes_[dimension]; }
    constexpr size_type& operator[](index_type dimension) noexcept { return sizes_[dimension]; }
    constexpr size_type size() const noexcept { return size_; }
    
    constexpr auto begin() { return sizes_.begin(); }
    constexpr auto end() { return sizes_.begin() + rank_; }

    constexpr auto begin() const { return sizes_.begin(); }
    constexpr auto end() const { return sizes_.begin() + rank_; }

    constexpr auto cbegin() const { return sizes_.cbegin(); }
    constexpr auto cend() const { return sizes_.cbegin() + rank_; }

    constexpr auto front() const { return sizes_.front(); }
    constexpr auto back() const { 
        assert(rank_ > 0 && "Cannot call back() on an empty Shape");
        return sizes_[rank_ - 1];
    }    
        
    constexpr rank_type normalize(index_type index, rank_type extra = 0) const { 
        rank_type bound = rank() + extra;
        if (index < 0) index += bound;
        assert(index >= 0  && index < bound && "Index out of bound");
        return static_cast<rank_type>(index);
    }

    constexpr Shape transpose(index_type first, index_type second) const { 
        Shape result = *this;
        std::swap(result.sizes_[normalize(first)], result.sizes_[normalize(second)]);
        return result;
    }


        
private:
    size_type size_;
    std::array<size_type, limit> sizes_{};
    size_type rank_{0};
};
 
constexpr bool operator==(const Shape& first, const Shape& second) {
    if (first.rank() != second.rank()) return false;
    for (Shape::size_type dimension = 0; dimension < first.rank(); ++dimension) {
        if (first[dimension] != second[dimension]) return false;
    }
    return true;
} 


class Strides {
public:
    using step_type = std::size_t;
    using rank_type = Shape::rank_type;
    using index_type = Shape::index_type;
    static constexpr uint8_t limit = 8;  

    constexpr Strides() noexcept = default;
 
    template<typename... Steps>
    constexpr explicit Strides(Steps... steps)
    :   steps_{static_cast<step_type>(steps)...}
    ,   rank_(sizeof...(steps)) {
        assert(sizeof...(steps) < limit && "Strides rank limit exceeded");
    }
  
    template<std::input_iterator Iterator>
    constexpr Strides(Iterator begin, Iterator end) { 
        step_type dimension = 0;
        for (auto iterator = begin; iterator != end; ++iterator) {
            assert(dimension < limit && "Strides rank limit exceeded");
            steps_[dimension++] = static_cast<step_type>(*iterator);
        }
        rank_ = dimension;
    }
 

    constexpr explicit Strides(const Shape& shape) { 
        rank_ = shape.rank();
        if (rank_ == 0) return;
        
        steps_[rank_ - 1] = 1;
        for (int step = rank_ - 2; step >= 0; --step) {
            steps_[step] = steps_[step + 1] * shape[step + 1];
        }
    }
    

    constexpr rank_type normalize(index_type index, rank_type extra = 0) const { 
        rank_type bound = rank() + extra;
        if (index < 0) index += bound;
        assert(index >= 0  && index < bound && "Index out of bound");
        return static_cast<rank_type>(index);
    }

    constexpr step_type rank() const noexcept { return rank_; }
    constexpr step_type operator[](index_type dimension) const noexcept { return steps_[normalize(dimension)]; }
    constexpr step_type& operator[](index_type dimension) noexcept { return steps_[normalize(dimension)]; }

    constexpr auto begin() { return steps_.begin(); }
    constexpr auto end() { return steps_.begin() + rank_; }

    constexpr auto begin() const { return steps_.begin(); }
    constexpr auto end() const { return steps_.begin() + rank_; }

    constexpr auto cbegin() const { return steps_.cbegin(); }
    constexpr auto cend() const { return steps_.cbegin() + rank_; }

    constexpr auto front() const { return steps_.front(); }

    
    constexpr Strides transpose(index_type first, index_type second) const { 
        Strides result = *this;
        std::swap(result.steps_[normalize(first)], result.steps_[normalize(second)]);
        return result;
    }
    
private:
    std::array<step_type, limit> steps_{};
    rank_type rank_{0};
};

using Allocator = std::variant<Host>;

class Storage {
public: 

    Storage() = default;

    
    Storage(std::size_t size, uint8_t dsize, Allocator allocator = Host{})
        : memory_(size * dsize) 
        , allocator_(allocator) {
            references_ = new std::atomic<std::size_t>(1); 
            address_ = std::visit([&](auto& allocator) -> void* {
                return allocator.allocate(memory_);
            }, allocator_);
        }
 
    Storage(const Storage& other)
        : memory_(other.memory_) 
        , allocator_(other.allocator_)
        , address_(other.address_)
        , references_(other.references_) {
            ++(*references_);
        }
 
    Storage(Storage&& other) noexcept
        : memory_(other.memory_) 
        , allocator_(std::move(other.allocator_))
        , address_(std::exchange(other.address_, nullptr))
        , references_(std::exchange(other.references_, nullptr)) {}

 
    Storage& operator=(const Storage& other) {
        if (this != &other) {
            release();
            memory_ = other.memory_; 
            allocator_ = other.allocator_;
            address_ = other.address_;
            references_ = other.references_;
            ++(*references_);
        }
        return *this;
    }
    
    Storage& operator=(Storage&& other) noexcept {
        if (this != &other) {
            release();
            memory_ = other.memory_; 
            allocator_ = std::move(other.allocator_);
            address_ = std::exchange(other.address_, nullptr);
            references_ = std::exchange(other.references_, nullptr); 
        }
        return *this;
    }

    ~Storage() {
        release();
    }

    std::size_t references() const {
        return references_ ? references_->load() : 0;
    }

    void* address() { return address_; }
    void const* address() const { return address_; }
    Allocator const& allocator() const { return allocator_; }
    std::size_t memory() const { return memory_; }

private:
    void release() {
        if (references_) {
            if (--(*references_) == 0) {
                if (address_) {
                    std::visit([&](auto& variant) {
                        variant.deallocate(address_, memory_);
                    }, allocator_);
                }
                delete references_;
            }
            references_ = nullptr;
            address_ = nullptr;
        }
    }

    std::size_t memory_ = 0; 
    Allocator allocator_ = Host{};
    void* address_ = nullptr;
    std::atomic<std::size_t>* references_ = nullptr;
};


class Tensor {
public:
    using index_type = int;
    using size_type = Shape::size_type;
    using rank_type = Shape::rank_type;
    using difference_type = std::ptrdiff_t;

    type dtype() const { return dtype_; }
    Shape const& shape() const { return shape_; }
    Strides const& strides() const { return strides_; }
    Storage const & storage() const { return storage_; }
    Shape::size_type size(Shape::index_type index) const { return shape_[index]; } 
    Strides::step_type stride(Strides::index_type index) const { return strides_[index]; }
    rank_type rank() const { return shape_.rank(); } 
    void* address() { return static_cast<std::byte*>(storage_.address()) + offset_; } 
    void const* address() const { return static_cast<const std::byte*>(storage_.address()) + offset_;   }  
    size_type size() const { return shape_.size(); }      
    Tensor() = default;
 
    template<class Size>
    Tensor(std::initializer_list<Size> sizes, type dtype, Allocator allocator = Host{})
    :   dtype_(dtype)
    ,   shape_(sizes.begin(), sizes.end()) 
    ,   strides_(shape_)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
    {}
     
    template<class Allocator>
    Tensor(Shape const& shape, type dtype, Allocator allocator = Host{})
    :   dtype_(dtype)
    ,   shape_(shape) 
    ,   strides_(shape_)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
    {}

    template<class Allocator>
    Tensor(Shape&& shape, type dtype, Allocator allocator = Host{})
    :   dtype_(dtype)
    ,   shape_(std::move(shape)) 
    ,   strides_(shape_)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
    {}


    template <class Expression, typename = std::enable_if_t<!std::is_arithmetic_v<Expression>>>
    Tensor(const Expression& expression) {
        *this = expression.forward();  
    }

    template <class Expression, typename = std::enable_if_t<!std::is_arithmetic_v<Expression>>>
    Tensor& operator=(const Expression& expression) {  
        *this = expression.forward(); 
        return *this;
    }
 
    const Tensor& forward() const {
        return *this;
    }


    constexpr index_type normalize(index_type index) const {  
        auto size = shape_.front();
        if (index < 0) index += size;
        assert(index >= 0 && index < size && "Index out of bound");
        return index;
    }

    Tensor operator[](index_type index) const { 
        assert(rank() >= 1 && "Cannot slice scalar tensor");
        difference_type offset = offset_ + normalize(index) * strides_.front() * dsizeof(dtype_);
        Strides strides(strides_.begin() + 1, strides_.end());
        Shape shape(shape_.begin() + 1, shape_.end());
        return Tensor(storage_, std::move(shape), std::move(strides), dtype_, offset);
    }

    Tensor transpose(Shape::index_type first, Shape::index_type second) const {
        return Tensor(storage_, std::move(shape_.transpose(first, second)), std::move(strides_.transpose(first, second)), dtype_, offset_);
    }

    Tensor squeeze() const {
        std::vector<Shape::size_type> shape;
        std::vector<Strides::step_type> strides;
        for (rank_type dimension = 0; dimension < shape_.rank(); ++dimension) {
            if (shape_[dimension] != 1) {
                shape.push_back(shape_[dimension]);
                strides.push_back(strides_[dimension]);
            }
        }

        if (shape.empty()) {
            shape.push_back(1);
            strides.push_back(1);
        } 
         
        return Tensor(storage_, Shape(shape.begin(), shape.end()), Strides(strides.begin(), strides.end()), dtype_, offset_);
    }
  
    Tensor unsqueeze(Shape::index_type index) const {
        std::vector<Shape::size_type> shape(shape_.begin(), shape_.end());
        std::vector<Strides::step_type> strides(strides_.begin(), strides_.end()); 
        { auto iterator = shape.begin(); std::advance(iterator, shape_.normalize(index, 1)); shape.insert(iterator, 1); }
        { auto iterator = strides.begin(); std::advance(iterator, strides_.normalize(index, 1)); strides.insert(iterator, 1); }
        return Tensor(storage_, Shape(shape.begin(), shape.end()), Strides(strides.begin(), strides.end()), dtype_, offset_);
    }


    template<typename... Indexes>
    Tensor unsqueeze(Indexes... indexes) const {
        Shape::index_type extra = sizeof...(indexes);
        std::vector<Shape::index_type> dimensions{static_cast<Shape::index_type>(indexes)...};
        std::sort(dimensions.begin(), dimensions.end());

        std::vector<Shape::size_type> shape(shape_.begin(), shape_.end());
        std::vector<Strides::step_type> strides(strides_.begin(), strides_.end()); 
        for (Shape::index_type index : dimensions) {   
            { auto iterator = shape.begin(); std::advance(iterator, shape_.normalize(index, extra)); shape.insert(iterator, 1); }
            { auto iterator = strides.begin(); std::advance(iterator, strides_.normalize(index, extra)); strides.insert(iterator, 1); }
        }
        return Tensor(storage_, Shape(shape.begin(), shape.end()), Strides(strides.begin(), strides.end()), dtype_, offset_);
    }   

    template<typename T>
    void operator=(T value) {
        assert(rank() == 0 && "Can't assign a scalar to an Array with more than one element.");
        traits[dtype_].assign(address(), value);
    }

    template<typename T> 
    bool operator==(T value) const {
        assert(rank() == 0 && "Can't compare a scalar to an Array with more than one element.");
        return traits[dtype_].compare(address(), value); 
    }

    template<typename T>
    T item() const {
        return std::any_cast<T>(traits[dtype_].retrieve(address()));
    }
 
    Tensor(const Storage& storage, Shape&& shape, Strides&& strides, type dtype, difference_type offset = 0)
    :   storage_(storage)
    ,   shape_(std::move(shape))
    ,   strides_(std::move(strides))
    ,   dtype_(dtype)
    ,   offset_(offset)
    {}

    bool is_transposed() const {
        return strides_[-1] > strides_[-2] ? true : false;
    }

    private:
    type dtype_ = any;
    Shape shape_; 
    Strides strides_;
    Storage storage_;
    difference_type offset_ = 0; 
}; 

inline std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    if (tensor.rank() == 0) {  
        traits[tensor.dtype()].print(os, const_cast<void*>(tensor.address()));
    } else {  
        os << "[";
        if (tensor.rank() == 1) {  
            for (auto index = 0; index < tensor.shape().front(); ++index) {
                if (index != 0) os << ", ";
                os << tensor[index];   
            }
        } else {  
            for (auto index = 0; index < tensor.shape().front(); ++index) {
                if (index != 0) os << ",\n ";
                os << tensor[index];   
            }
        }
        os << "]";
    }
    return os;
}



template<class Derived>
struct Module {
public:
    template<class... Operands>
    auto operator()(Operands&&... operands) const -> decltype(auto) {
        return static_cast<const Derived*>(this)->forward(std::forward<Operands>(operands)...);
    }
};



class Embedding : public Module<Embedding> {
    public: 
    template<typename Lenght, typename Dimension, class Allocator = Host>
    Embedding(Lenght lenght, Dimension dimension, type dtype, type itype = integer64, Allocator allocator = Allocator{})
    :   dtype_(dtype)
    ,   itype_(itype)
    ,   shape_(lenght, dimension)
    ,   storage_(shape_.size(), dsizeof(dtype_), allocator)
    {}

    template<class... Indexes>
    Tensor forward(Indexes... indexes) const {   
        Tensor lookup(Shape(sizeof...(indexes)), itype_, storage_.allocator());
        Tensor result(Shape(sizeof...(indexes), shape_.back()), dtype_, storage_.allocator()); 
        forward(result, lookup);
        return result;
    }

    void forward(Tensor const& result, Tensor const& lookup) const;

    private:
    type dtype_;
    type itype_;
    Shape shape_;
    Storage storage_;
};


int main() { 
    Tensor x({3,3}, float32);
    Embedding embedding(5, 5, float32);
}