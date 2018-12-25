// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/persp_camera.h>

namespace jet {
namespace gfx {

PerspCamera::PerspCamera() {}

PerspCamera::PerspCamera(const CameraState& state_,
                         double fieldOfViewInRadians_)
    : Camera(state_), fieldOfViewInRadians(fieldOfViewInRadians_) {}

PerspCamera::~PerspCamera() {}

    Matrix4x4D PerspCamera::projectionMatrix() const {
    double fov_2, left, right, bottom, top;

    fov_2 = fieldOfViewInRadians * 0.5;
    top = state.nearClipPlane / (std::cos(fov_2) / std::sin(fov_2));
    bottom = -top;

    right = top * state.viewport.aspectRatio();
    left = -right;

    // https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
    double a, b, c, d;
    a = (right + left) / (right - left);
    b = (top + bottom) / (top - bottom);
    c = -(state.farClipPlane + state.nearClipPlane) /
        (state.farClipPlane - state.nearClipPlane);
    d = -(2 * state.farClipPlane * state.nearClipPlane) /
        (state.farClipPlane - state.nearClipPlane);

    return Matrix4x4D(
        2 * state.nearClipPlane / (right - left), 0, a, 0,  // 1st row
        0, 2 * state.nearClipPlane / (top - bottom), b, 0,  // 2nd row
        0, 0, c, d,                                         // 3rd row
        0, 0, -1, 0);                                       // 4th row
}

}  // namespace gfx
}  // namespace jet
