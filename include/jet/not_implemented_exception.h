// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_NOT_IMPLEMENTED_EXCEPTION_H_
#define INCLUDE_JET_NOT_IMPLEMENTED_EXCEPTION_H_

#include <exception>
#include <string>

namespace jet {

class NotImplementedException : public std::exception {
 public:
    NotImplementedException(
        const std::string& message = "Functionality not yet implemented");

    const char* what() const noexcept override;

 private:
     std::string _message;
};

}  // namespace jet

#endif  // INCLUDE_JET_NOT_IMPLEMENTED_EXCEPTION_H_
