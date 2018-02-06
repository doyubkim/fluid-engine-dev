// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_PERSP_CAMERA_H_
#define INCLUDE_JET_VIZ_PERSP_CAMERA_H_

#include "camera.h"

namespace jet {
namespace viz {

//! Perspective camera representation.
class PerspCamera : public Camera {
 public:
    //!
    //! \brief Default constructor.
    //!
    //! Newly create camera will be positioned at (0, 0, 1) with look-up vector
    //! (0, 1, 0) and look-at direction (0, 0, -1).
    //!
    PerspCamera();

    //!
    //! \brief Constructs an perspective camera with given parameters.
    //!
    //! Newly create camera will be positioned at (0, 0, 1) with look-up vector
    //! (0, 1, 0) and look-at direction (0, 0, -1).
    //!
    //! \param origin Position of the camera.
    //! \param lookAt Look-at directional vector.
    //! \param lookUp Look-up directional vector.
    //! \param nearClipPlane Near clip plane position.
    //! \param farClipPlane Far clip plane position.
    //! \param viewport Viewport info.
    //! \paramfieldOfViewInRadians Field of view in randians (vertical).
    //!
    PerspCamera(const Vector3D& origin, const Vector3D& lookAt,
                const Vector3D& lookUp, double nearClipPlane,
                double farClipPlane, const Viewport& viewport = Viewport(),
                double fieldOfViewInRadians = pi<double>() / 2.0);

    //! Destructor.
    virtual ~PerspCamera();

    //! Returns the field of view in radians (vertical).
    double fieldOfViewInRadians() const;

    //! Sets the field of view in radians (vertical).
    void setFieldOfViewInRadians(double fov);

 protected:
    //! Updates the matrix.
    virtual void updateMatrix() override;

 private:
    double _fieldOfViewInRadians;
};

typedef std::shared_ptr<PerspCamera> PerspCameraPtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_PERSP_CAMERA_H_
