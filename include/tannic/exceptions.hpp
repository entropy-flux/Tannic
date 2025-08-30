// Copyright 2025 Eric Hermosis
//
// This file is part of the Tannic Tensor Library.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions andx
// limitations under the License.
//
 
#ifndef EXCEPTIONS_HPP
#define EXCEPTIONS_HPP

/**
 * @file functions.hpp
 * @author Eric Hermosis
 * @date 2025
 * @brief Defines the excception types used in the library.  
**/

#include <exception>
#include <string>

namespace tannic {

/**
 * @class Exception
 * @brief A simple generic exception type for the Tannic Tensor Library.
 *
 * This class provides a lightweight generic exception mechanism during for 
 * the frontend of the library.  
 * 
 * This class is a simple generic Exception to avoid overengineering during
 * initial development. As the library matures, this may be extended or
 * replaced with more specialized exception types.
 */
class Exception : public std::exception {
public:
    std::string message;
    
    explicit Exception(const std::string& what)
    :   message(what) {}

    const char* what() const noexcept override {
        return message.c_str();
    }
}; 

} // namespace exception

#endif // EXCEPTIONS_HPP