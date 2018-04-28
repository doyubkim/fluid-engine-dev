// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "pch.h"

#include "no_op_animation.h"

JET_BEGIN_NAMESPACE_WRAPPER_SDK{

NoOpAnimationImpl::NoOpAnimationImpl() :
    AnimationImpl(std::make_shared<JET_NATIVE_SDK::NoOpAnimation>()) {
}

//

NoOpAnimation::NoOpAnimation() {
    // Create impl object.
    _impl = JET_WRAPPER_NEW NoOpAnimationImpl();
}

}
JET_END_NAMESPACE_WRAPPER_SDK
