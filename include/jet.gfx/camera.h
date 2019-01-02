// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_CAMERA_H_
#define INCLUDE_JET_GFX_CAMERA_H_

#include <jet.gfx/viewport.h>
#include <jet/matrix.h>

#include <memory>

namespace jet {

namespace gfx {

//! \brief Camera state representation.
class CameraState {
 public:
    //! Origin of the camera.
    Vector3F origin = Vector3F(0, 0, 1);

    //! Look-at direction.
    Vector3F lookAt = Vector3F(0, 0, -1);

    //! Up vector.
    Vector3F lookUp = Vector3F(0, 1, 0);

    //! Distance to the near-clip plane.
    float nearClipPlane = 0.1f;

    //! Distance to the far-clip plane.
    float farClipPlane = 100.0f;

    //! Viewport in screen space.
    Viewport viewport;

    //!
    //! \brief Returns view matrix.
    //!
    //! This function returns the matrix maps the reference point to the
    //! negative z axis and the eye point to the origin.
    //!
    //! \return The view matrix.
    //!
    Matrix4x4F viewMatrix() const;

    //! Returns true if equal to the other state.
    bool operator==(const CameraState& other) const;

    //! Returns true if not equal to the other state.
    bool operator!=(const CameraState& other) const;
};

//! \brief Base class for cameras.
class Camera {
 public:
    //! The camera state.
    CameraState state;

    //! Default constructor.
    Camera();

    //!
    //! Constructs a camera with a camera state.
    //!
    //! \param state    Camera state.
    //!
    Camera(const CameraState& state);

    //! Destructor.
    virtual ~Camera();

    //!
    //! \brief Returns the projection matrix for this camera.
    //!
    //! This function maps camera view coordinates to the normalized device
    //! coordinates ([-1, -1, -1] x [1, 1, 1]) which is following OpenGL
    //! convention.
    //!
    //! \return The projection matrix.
    //!
    virtual Matrix4x4F projectionMatrix() const = 0;
};

//! Shared pointer type for Camera.
using CameraPtr = std::shared_ptr<Camera>;

}  // namespace gfx

}  // namespace jet

#endif  // INCLUDE_JET_GFX_CAMERA_H_
