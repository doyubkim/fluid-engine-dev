// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/camera.h>

namespace jet {
namespace gfx {

// MARK: CameraState

Matrix4x4D CameraState::viewMatrix() const {
    // https://www.opengl.org/sdk/docs/man2/xhtml/gluLookAt.xml
    // The matrix maps the reference point to the negative z axis and the eye
    // point to the origin.

    Vector3D right = lookAt.cross(lookUp);           // normalized right vec
    Vector3D up = right.normalized().cross(lookAt);  // normalized up vec

    Matrix4x4D view(right.x, right.y, right.z, 0,        // 1st row
                    up.x, up.y, up.z, 0,                 // 2nd row
                    -lookAt.x, -lookAt.y, -lookAt.z, 0,  // 3rd row
                    0, 0, 0, 1);                         // 4th row
    Matrix4x4D translation = Matrix4x4D::makeTranslationMatrix(-origin);

    return view * translation;
}

bool CameraState::operator==(const CameraState& other) const {
    return origin == other.origin && lookAt == other.lookAt &&
           lookUp == other.lookUp && nearClipPlane == other.nearClipPlane &&
           farClipPlane == other.farClipPlane && viewport == other.viewport;
}

bool CameraState::operator!=(const CameraState& other) const {
    return !(*this == other);
}

// MARK: Camera

Camera::Camera() {}

Camera::Camera(const CameraState& state_) : state(state_) {}

Camera::~Camera() {}

}  // namespace gfx
}  // namespace jet
