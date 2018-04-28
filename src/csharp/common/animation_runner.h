// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#pragma once

#include "animation.h"

namespace JET_NATIVE_SDK {

//! Simple utility class to test IAnimation from native code
class AnimationRunner {
 public:
    AnimationRunner();

    void run(AnimationPtr anim, double fps, int numFrames);
};

}  // namespace JET_NATIVE_SDK

JET_BEGIN_NAMESPACE_WRAPPER_SDK {
    //! Simple utility class to test IAnimation from managed or WinRT codes
public ref class AnimationRunner {
    JET_DEFINE_NATIVE_CORE_FOR_BASE(JET_NATIVE_SDK::AnimationRunner);

    public:
    AnimationRunner();

    JET_DEFAULT_DESTRUCTOR_FOR_BASE(AnimationRunner);

    void Run(IAnimation^ anim, double fps, int numFrames);
};
}
JET_END_NAMESPACE_WRAPPER_SDK
