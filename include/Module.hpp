#ifndef MODULE_HPP
#define MODULE_HPP

#include <iostream>
#include <utility> 

template<class Derived>
class Module {
    public:  
    template<typename... Arguments>
    constexpr auto operator()(Arguments&&... arguments) const -> decltype(auto) {
        auto result = static_cast<const Derived*>(this)->forward(std::forward<Arguments>(arguments)...); 
        return result;
    }
};

#endif  // MODULE_HPP