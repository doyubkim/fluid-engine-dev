// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pch.h"

#include "animation.h"

using JET_NAMESPACE_WRAPPER_SDK;

AnimationImpl::AnimationImpl(
    const JET_NATIVE_SDK::AnimationPtr& nativeSharedPtr) {
    JET_INITIALIZE_NATIVE_CORE_WITH_SHARED_PTR(nativeSharedPtr);
}
