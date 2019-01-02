// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/exceptions.h>

namespace jet {

NotImplementedException::NotImplementedException(const std::string& message)
    : _message(message) {}

NotImplementedException::NotImplementedException(
    const NotImplementedException& other) noexcept
    : _message(other._message) {}

NotImplementedException::~NotImplementedException() {}

const char* NotImplementedException::what() const noexcept {
    return _message.c_str();
}

}  // namespace jet
