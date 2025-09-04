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
 * @file exceptions.hpp
 * @author Eric Hermosis
 * @date 2025
 * @brief Defines the exceptions used in the library.  
**/

#include <exception>
#include <string>

namespace tannic { 

/**
 * @brief Base class for recoverable runtime anomalies in the framework.
 * 
 * The Exception class represents conditions that can be thrown
 * and potentially recovered from during normal program execution.
 * Examples include invalid input, file not found, or other recoverable
 * logical issues. 
 * 
 * @note Exceptions are intended to be caught and handled.
 */
class Exception : public std::exception {
public:
    std::string message;
    
    /**
     * @brief Construct an Exception with a message.
     * @param what Description of the exception.
     */
    explicit Exception(const std::string& what)
    :   message(what) {}

    /**
     * @brief Retrieve the exception message.
     * @return C-style string of the exception message.
     * @note Overrides std::exception::what().
     */
    const char* what() const noexcept override {
        return message.c_str();
    }
}; 


/**
 * @brief Represents serious, typically unrecoverable problems.
 * 
 * The Error class is used for critical failures in the program or
 * environment that should not normally be caught and handled.
 * Examples include out-of-memory conditions or hardware failures.
 * 
 * @note While technically catchable, errors are conceptually distinct 
 * from exceptions and should rarely be handled in normal application logic.
 */
class Error : public std::exception {
public:
    std::string message;
    
    /**
     * @brief Construct an Exception with a message.
     * @param what Description of the exception.
     */
    explicit Error(const std::string& what)
    :   message(what) {}

    /**
     * @brief Retrieve the exception message.
     * @return C-style string of the exception message.
     * @note Overrides std::exception::what().
     */
    const char* what() const noexcept override {
        return message.c_str();
    }
};   

} // namespace exception

#endif // EXCEPTIONS_HPP