// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pch.h"

#include "frame.h"

using JET_NAMESPACE_WRAPPER_SDK;

Frame::Frame() { 
    // Create native object and store it
    JET_INITIALIZE_NATIVE_CORE; 
}

Frame::Frame(int newIndex, double newTimeIntervalInSeconds) {
    // Create native object with parameters and store it
    JET_INITIALIZE_NATIVE_CORE_2(newIndex, newTimeIntervalInSeconds);
}

Frame::Frame(Frame^ other) {
    // Create native object with parameters and store it
    JET_INITIALIZE_NATIVE_CORE_1(*other->getActualPtr());
}

Frame^ operator++(Frame^ frame) {
    frame->Advance();
    return JET_WRAPPER_NEW Frame(frame);
}

Frame^ operator++(Frame^ frame, int) {
    Frame^ result = JET_WRAPPER_NEW Frame(frame);
    frame->Advance();
    return result;
}
