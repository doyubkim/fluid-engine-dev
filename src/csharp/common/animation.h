// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#pragma once

#include "frame.h"

#include <jet/animation.h>

JET_BEGIN_NAMESPACE_WRAPPER_SDK {

// High-level interface.
public interface class IAnimation {
public:
    virtual JET_OBJECT^ GetAnimationImpl() = 0;
};

// Actual implementation class that holds native object.
// In this cass, AnimationImpl is responsible for keeping native AnimationPtr
// and also destroying it in the end.
private ref class AnimationImpl {
    // Defines all the codes to store native object from a base class.
    JET_DEFINE_NATIVE_CORE_FOR_BASE(JET_NATIVE_SDK::Animation);
internal:
    // Constructs this class with native object.
    AnimationImpl(const JET_NATIVE_SDK::AnimationPtr& nativeSharedPtr);

public:
    // Defines destructor and finalize native object.
    JET_DEFAULT_DESTRUCTOR_FOR_BASE(AnimationImpl);

    // Native function wrapper.
    JET_MEMBER_FUNCTION_NO_RETURN_1(update, Update, Frame^, frame);
};
}
JET_END_NAMESPACE_WRAPPER_SDK
