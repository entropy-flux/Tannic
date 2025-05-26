#ifndef MODULE_HPP
#define MODULE_HPP

#include <iostream>
#include <utility> 

template<class Derived>
class Module {
    public:  
    template<typename... Args>
    constexpr auto operator()(Args&&... args) const -> decltype(auto) {
        auto result = static_cast<const Derived*>(this)->forward(std::forward<Args>(args)...); 
        return result;
    }
};

#endif  // MODULE_HPP