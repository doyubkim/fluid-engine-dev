// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_PERSP_CAMERA_H_
#define INCLUDE_JET_GFX_PERSP_CAMERA_H_

#include <jet.gfx/camera.h>
#include <jet/constants.h>

namespace jet {
namespace gfx {

//! \brief Perspective camera representation.
class PerspCamera : public Camera {
 public:
    //! Camera's field of view in radians.
    float fieldOfViewInRadians = kHalfPiF;

    //! \brief Constructs an perspective camera with default camera state.
    PerspCamera();

    //!
    //! \brief Constructs an perspective camera with given parameters.
    //!
    //! \param state                Common camera state.
    //! \param fieldOfViewInRadians Perspective camera's field of view in
    //! radian.
    //!
    PerspCamera(const CameraState& state, float fieldOfViewInRadians);

    //! Destructor.
    virtual ~PerspCamera();

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

//! Shared pointer type for PerspCamera.
using PerspCameraPtr = std::shared_ptr<PerspCamera>;

}  // namespace gfx
}  // namespace jet

#endif  // INCLUDE_JET_GFX_PERSP_CAMERA_H_
