// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/ortho_camera.h>

namespace jet {
namespace gfx {

OrthoCamera::OrthoCamera() {}

OrthoCamera::OrthoCamera(const CameraState &state, float left_, float right_,
                         float bottom_, float top_)
    : Camera(state), left(left_), right(right_), bottom(bottom_), top(top_) {}

OrthoCamera::~OrthoCamera() {}

float OrthoCamera::width() const { return right - left; }

float OrthoCamera::height() const { return top - bottom; }

Vector2F OrthoCamera::center() const {
    return Vector2F(0.5f * (left + right), 0.5f * (bottom + top));
}

Matrix4x4F OrthoCamera::projectionMatrix() const {
    float dx = right - left;
    float tx = -(right + left) / dx;
    float dy = top - bottom;
    float ty = -(top + bottom) / dy;
    float dz = state.farClipPlane - state.nearClipPlane;
    float tz = -(state.farClipPlane + state.nearClipPlane) / dz;

    // https://www.opengl.org/sdk/docs/man2/xhtml/glOrtho.xml
    return Matrix4x4F(2 / dx, 0, 0, tx,   // 1st row
                      0, 2 / dy, 0, ty,   // 2nd row
                      0, 0, -2 / dz, tz,  // 3rd row
                      0, 0, 0, 1);        // 4th row
}

}  // namespace gfx
}  // namespace jet
