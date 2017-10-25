// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/ortho_camera.h>

using namespace jet;
using namespace viz;

OrthoCamera::OrthoCamera()
    : Camera(), _left(-1.0), _right(1.0), _bottom(-1.0), _top(1.0) {
    updateMatrix();
}

OrthoCamera::OrthoCamera(const Vector3D& origin, const Vector3D& lookAt,
                         const Vector3D& lookUp, double nearClipPlane,
                         double farClipPlane, double left, double right,
                         double bottom, double top)
    : Camera(origin, lookAt, lookUp, nearClipPlane, farClipPlane,
             Viewport(left, bottom, right - left, top - bottom)) {
    updateMatrix();
}

OrthoCamera::~OrthoCamera() {}

double OrthoCamera::left() const { return _left; }

void OrthoCamera::setLeft(double newLeft) { _left = newLeft; }

double OrthoCamera::right() const { return _right; }

void OrthoCamera::setRight(double newRight) { _right = newRight; }

double OrthoCamera::bottom() const { return _bottom; }

void OrthoCamera::setBottom(double newBottom) { _bottom = newBottom; }

double OrthoCamera::top() const { return _top; }

void OrthoCamera::setTop(double newTop) { _top = newTop; }

double OrthoCamera::width() const { return _right - _left; }

double OrthoCamera::height() const { return _top - _bottom; }

Vector2D OrthoCamera::center() const {
    return Vector2D(0.5 * (_left + _right), 0.5 * (_bottom + _top));
}

void OrthoCamera::onResize(const Viewport& viewport) {
    UNUSED_VARIABLE(viewport);
}

void OrthoCamera::updateMatrix() {
    double tx = -(_right + _left) / (_right - _left);
    double ty = -(_top + _bottom) / (_top - _bottom);
    double tz = -(_state.farClipPlane + _state.nearClipPlane) /
                (_state.farClipPlane - _state.nearClipPlane);

    // https://www.opengl.org/sdk/docs/man2/xhtml/glOrtho.xml
    Matrix4x4D projection(2.0 / (_right - _left), 0, 0, 0,  // 1st column
                          0, 2.0 / (_top - _bottom), 0, 0,  // 2nd column
                          0, 0,
                          -2.0 / (_state.farClipPlane - _state.nearClipPlane),
                          0,               // 3rd column
                          tx, ty, tz, 1);  // 4th column

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
