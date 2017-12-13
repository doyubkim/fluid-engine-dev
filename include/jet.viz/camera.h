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

//! Base class for cameras.
class Camera {
 public:
    //! Default constructor.
    Camera();

    //!
    //! Constructs a camera with parameters.
    //!
    //! \param origin Origin of the camera (eye position).
    //! \param lookAt Direction of the camera.
    //! \param lookUp Up-direction of the camera.
    //! \param nearClipPlane Position of the near-clip plane.
    //! \param farClipPlane Position of the far-clip plane.
    //! \param viewport The viewport.
    //!
    Camera(const Vector3D& origin, const Vector3D& lookAt,
           const Vector3D& lookUp, double nearClipPlane, double farClipPlane,
           const Viewport& viewport);

    //! Destructor.
    virtual ~Camera();

    //!
    //! Resizes the view.
    //!
    //! \param viewport New viewport to set.
    //!
    void resize(const Viewport& viewport);

    //! Returns the corresponding transformation matrix.
    const Matrix4x4D& matrix() const;

    //! Returns the camera state.
    const BasicCameraState& basicCameraState() const;

    //!
    //! Sets the camera state.
    //!
    //! \param state New camera state.
    //!
    void setBasicCameraState(const BasicCameraState& state);

 protected:
    //! Transformation matrix.
    Matrix4x4D _matrix;

    //! Camera state.
    BasicCameraState _state;

    //!
    //! Called when resize function is invoked.
    //!
    //! \param viewport New viewport.
    //!
    virtual void onResize(const Viewport& viewport);

    //! Update the matrix based on the current camera state.
    virtual void updateMatrix();
};

typedef std::shared_ptr<Camera> CameraPtr;

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_CAMERA_H_
