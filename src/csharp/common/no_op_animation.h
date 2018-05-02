// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#pragma once

#include "animation.h"

namespace JET_NATIVE_SDK {

// Sample native class that is leaf of the hierarchy.
class NoOpAnimation final : public Animation {
 public:
     NoOpAnimation() = default;

     Frame lastFrame() const {
         return _lastFrame;
     }

 protected:
     void onUpdate(const Frame& frame) override {
         _lastFrame = frame;
     }

 private:
     Frame _lastFrame;
};

typedef std::shared_ptr<NoOpAnimation> NoOpAnimationPtr;

}

JET_BEGIN_NAMESPACE_WRAPPER_SDK{

// Impl class for NoOpAnimation class. 
// Every native class with hierarchy should have 1:1 Impl class.
private ref class NoOpAnimationImpl : public AnimationImpl {
    // Setups required code for derived class.
    JET_DEFINE_NATIVE_CORE_FOR_DERIVED(JET_NATIVE_SDK::NoOpAnimation);
internal:
    // Initialize Impl class (creates native object internally since this is the final child class).
    NoOpAnimationImpl();

public:
    // Defines destructor and finalize native object from derived class.
    JET_DEFAULT_DESTRUCTOR_FOR_DERIVED(NoOpAnimationImpl);

    // Native function wrapper(s).
    JET_MEMBER_FUNCTION_TO_GET_PROPERTY(Frame^, lastFrame, LastFrame);
};

// Actual class that will be exposed to the public.
public ref class NoOpAnimation sealed : public IAnimation {
public:
    // Default constructor.
    // This ctor will create actual native object and save it to Impl instance
    // which is also created and stored from this public class.
    NoOpAnimation();

    // IAnimation implementation
    JET_GET_IMPL(GetAnimationImpl);

    // Impl function wrapper(s)
    JET_MEMBER_FUNCTION_CALLING_IMPL_NO_RETURN_1(Update, Frame^, frame);

    JET_GET_PROPERTY_CALLING_IMPL(Frame^, LastFrame);

private:
    // Impl class instance
    NoOpAnimationImpl^ _impl;
};

}
JET_END_NAMESPACE_WRAPPER_SDK
