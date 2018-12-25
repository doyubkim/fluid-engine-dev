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

OrthoCamera::OrthoCamera(const CameraState &state, double left_, double right_,
                         double bottom_, double top_)
    : Camera(state), left(left_), right(right_), bottom(bottom_), top(top_) {}

OrthoCamera::~OrthoCamera() {}

double OrthoCamera::width() const { return right - left; }

double OrthoCamera::height() const { return top - bottom; }

Vector2D OrthoCamera::center() const {
    return Vector2D(0.5 * (left + right), 0.5 * (bottom + top));
}

Matrix4x4D OrthoCamera::projectionMatrix() const {
    double dx = right - left;
    double tx = -(right + left) / dx;
    double dy = top - bottom;
    double ty = -(top + bottom) / dy;
    double dz = state.farClipPlane - state.nearClipPlane;
    double tz = -(state.farClipPlane + state.nearClipPlane) / dz;

    // https://www.opengl.org/sdk/docs/man2/xhtml/glOrtho.xml
    return Matrix4x4D(2 / dx, 0, 0, tx,   // 1st row
                      0, 2 / dy, 0, ty,   // 2nd row
                      0, 0, -2 / dz, tz,  // 3rd row
                      0, 0, 0, 1);        // 4th row
}

}  // namespace gfx
}  // namespace jet
