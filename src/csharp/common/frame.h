// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#pragma once

#include "macros.h"

#include <jet/animation.h>

JET_BEGIN_NAMESPACE_WRAPPER_SDK {

// This demonstrates how to wrap flat types (no hiararchy).
public ref class Frame sealed {
    // Generate all the necessary code to store native object.
    JET_DEFINE_NATIVE_CORE_FOR_BASE(JET_NATIVE_SDK::Frame);
public:
    Frame();

    Frame(int newIndex, double newTimeIntervalInSeconds);

    Frame(Frame^ other);

    JET_DEFAULT_DESTRUCTOR_FOR_BASE(Frame);

    JET_MEMBER_FUNCTION_TO_GET_PROPERTY(double, timeInSeconds,
                                        TimeInSeconds);

    JET_MEMBER_VARIABLE_TO_PROPERTY(int, index, Index);

    JET_MEMBER_VARIABLE_TO_PROPERTY(double, timeIntervalInSeconds,
                                    TimeIntervalInSeconds);

    JET_MEMBER_FUNCTION_NO_RETURN(advance, Advance);

    JET_MEMBER_FUNCTION_NO_RETURN_1(advance, Advance, int, delta);
};

Frame^ operator++(Frame^ frame);

Frame^ operator++(Frame^ frame, int i);

// Define convertFromNative and convertToNative so that it can be used for automated bindings.
inline Frame^ convertFromNative(const jet::Frame& value) {
    return JET_WRAPPER_NEW Frame(value.index,
                                 value.timeIntervalInSeconds);
}

inline jet::Frame convertToNative(Frame^ value) {
    return jet::Frame(value->Index, value->TimeIntervalInSeconds);
}
}
JET_END_NAMESPACE_WRAPPER_SDK
