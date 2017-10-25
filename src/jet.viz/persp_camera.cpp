// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/persp_camera.h>

using namespace jet;
using namespace viz;

PerspCamera::PerspCamera()
    : Camera(), _fieldOfViewInRadians(pi<double>() / 2.0) {
    updateMatrix();
}

PerspCamera::PerspCamera(const Vector3D& origin, const Vector3D& lookAt,
                         const Vector3D& lookUp, double nearClipPlane,
                         double farClipPlane, const Viewport& viewport,
                         double fieldOfViewInRadians)
    : Camera(origin, lookAt, lookUp, nearClipPlane, farClipPlane, viewport),
      _fieldOfViewInRadians(fieldOfViewInRadians) {
    updateMatrix();
}

PerspCamera::~PerspCamera() {}

void PerspCamera::updateMatrix() {
    double fov_2, left, right, bottom, top;

    fov_2 = _fieldOfViewInRadians * 0.5;
    top = _state.nearClipPlane / (std::cos(fov_2) / std::sin(fov_2));
    bottom = -top;

    right = top * _state.viewport.aspectRatio();
    left = -right;

    // https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
    double a, b, c, d;
    a = (right + left) / (right - left);
    b = (top + bottom) / (top - bottom);
    c = -(_state.farClipPlane + _state.nearClipPlane) /
        (_state.farClipPlane - _state.nearClipPlane);
    d = -(2.0 * _state.farClipPlane * _state.nearClipPlane) /
        (_state.farClipPlane - _state.nearClipPlane);

    Matrix4x4D projection(
        2.0 * _state.nearClipPlane / (right - left), 0, 0, 0,  // 1st column
        0, 2.0 * _state.nearClipPlane / (top - bottom), 0, 0,  // 2nd column
        a, b, c, -1,                                           // 3rd column
        0, 0, d, 0);                                           // 4th column

    // https://www.opengl.org/sdk/docs/man2/xhtml/gluLookAt.xml
    const Vector3D& f = _state.lookAt;
    Vector3D s = f.cross(_state.lookUp);
    Vector3D u = s.normalized().cross(f);

    Matrix4x4D view(s.x, u.x, -f.x, 0,  // 1st column
                    s.y, u.y, -f.y, 0,  // 2nd column
                    s.z, u.z, -f.z, 0,  // 3rd column
                    0, 0, 0, 1);        // 4th column

    Matrix4x4D model;
    model = Matrix4x4D::makeTranslationMatrix(-_state.origin);

    _matrix = projection * view * model;
}
