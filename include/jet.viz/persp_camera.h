// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_PERSP_CAMERA_H_
#define INCLUDE_JET_VIZ_PERSP_CAMERA_H_

#include "camera.h"

namespace jet { namespace viz {

class PerspCamera : public Camera {
 public:
    PerspCamera();
    PerspCamera(const Vector3D& origin, const Vector3D& lookAt,
                const Vector3D& lookUp, double nearClipPlane,
                double farClipPlane, const Viewport& viewport,
                double fieldOfViewInRadians);

    virtual ~PerspCamera();

 protected:
    virtual void updateMatrix() override;

 private:
    double _fieldOfViewInRadians;
};

typedef std::shared_ptr<PerspCamera> PerspCameraPtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_PERSP_CAMERA_H_
