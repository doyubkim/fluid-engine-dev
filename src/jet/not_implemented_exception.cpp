// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/not_implemented_exception.h>
#include <string>

using namespace jet;

NotImplementedException::NotImplementedException(
    const std::string& message) : _message(message) {
}

const char* NotImplementedException::what() const noexcept {
    return _message.c_str();
}
