// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_CAMERA_H_
#define INCLUDE_JET_VIZ_CAMERA_H_

#include <jet.viz/viewport.h>
#include <jet/matrix4x4.h>
#include <jet/ray3.h>

#include <memory>

namespace jet {

namespace viz {

//! Common camera state representation.
struct BasicCameraState {
    //! Origin of the camera.
    Vector3D origin;

    //! Look-at direction.
    Vector3D lookAt;

    //! Up vector.
    Vector3D lookUp;

    //! Distance to the near-clip plane.
    double nearClipPlane;

    //! Distance to the far-clip plane.
    double farClipPlane;

    //! Viewport.
    Viewport viewport;
};

class Camera {
 public:
    Camera();
    Camera(const Vector3D& origin, const Vector3D& lookAt,
           const Vector3D& lookUp, double nearClipPlane, double farClipPlane,
           const Viewport& viewport);
    virtual ~Camera();

    void resize(const Viewport& viewport);

    const Matrix4x4D& matrix() const;

    Matrix4x4F matrixF() const;

    const BasicCameraState& basicCameraState() const;
    void setBasicCameraState(const BasicCameraState& state);

 protected:
    Matrix4x4D _matrix;
    BasicCameraState _state;

    virtual void onResize(const Viewport& viewport);

    virtual void updateMatrix();
};

typedef std::shared_ptr<Camera> CameraPtr;

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_CAMERA_H_
