// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pch.h"

#include "animation_runner.h"

namespace JET_NATIVE_SDK {

AnimationRunner::AnimationRunner() {}

void AnimationRunner::run(AnimationPtr anim, double fps, int numFrames) {
    for (Frame frame(0, 1.0 / fps); frame.index != numFrames; ++frame) {
        anim->update(frame);
    }
}

}  // namespace JET_NATIVE_SDK

JET_BEGIN_NAMESPACE_WRAPPER_SDK {
    AnimationRunner::AnimationRunner() {}

    void AnimationRunner::Run(IAnimation ^ anim, double fps, int numFrames) {
        AnimationImpl ^ animImpl =
            dynamic_cast<AnimationImpl ^>(anim->GetAnimationImpl());
        if (animImpl != nullptr) {
            getActualPtr()->run(animImpl->getNativeSharedPtr(), fps, numFrames);
        } else {
            JET_WRAPPER_THROW_INVALID_ARG("Can't find native code");
        }
    }
}
JET_END_NAMESPACE_WRAPPER_SDK
