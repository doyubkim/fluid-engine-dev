// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_ORTHO_CAMERA_H_
#define INCLUDE_JET_GFX_ORTHO_CAMERA_H_

#include <jet.gfx/camera.h>

namespace jet {

namespace gfx {

//! Orthographic camera representation.
class OrthoCamera : public Camera {
 public:
    float left = -1.0f;
    float right = 1.0f;
    float bottom = -1.0f;
    float top = 1.0f;

    //! \brief Constructs an orthographic camera with default parameters.
    OrthoCamera();

    //!
    //! \brief Constructs an orthographic camera with given view parameters.
    //!
    //! This constructor initializes a camera with common camera state as well
    //! as the view parameters which is defined in camera space.
    //!
    //! \param state    Common camera state.
    //! \param left     Left side of the view area.
    //! \param right    Right side of the view area.
    //! \param bottom   Bottom side of the view area.
    //! \param top      Top side of the view area.
    //!
    OrthoCamera(const CameraState& state, float left, float right, float bottom,
                float top);

    //! Destructor.
    virtual ~OrthoCamera();

    //! Width of the view area.
    float width() const;

    //! Height the view area.
    float height() const;

    //! Center of the view area.
    Vector2F center() const;

    //!
    //! \brief Returns the projection matrix for this camera.
    //!
    //! This function maps camera view coordinates to the normalized device
    //! coordinates ([-1, -1, -1] x [1, 1, 1]) which is following OpenGL
    //! convention.
    //!
    //! \return The projection matrix.
    //!
    Matrix4x4F projectionMatrix() const override;
};

//! Shared pointer type for OrthoCamera.
using OrthoCameraPtr = std::shared_ptr<OrthoCamera>;

}  // namespace gfx

}  // namespace jet

#endif  // INCLUDE_JET_GFX_ORTHO_CAMERA_H_
