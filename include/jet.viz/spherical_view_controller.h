// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_SPHERICAL_VIEW_CONTROLLER_H_
#define INCLUDE_JET_VIZ_SPHERICAL_VIEW_CONTROLLER_H_

#include "view_controller.h"

namespace jet { namespace viz {

class SphericalViewController : public ViewController {
 public:
    SphericalViewController(const CameraPtr& camera);
    virtual ~SphericalViewController();

 protected:
    virtual void onKeyDown(const KeyEvent& keyEvent) override;
    virtual void onKeyUp(const KeyEvent& keyEvent) override;

    virtual void onPointerPressed(const PointerEvent& pointerEvent) override;
    virtual void onPointerHover(const PointerEvent& pointerEvent) override;
    virtual void onPointerDragged(const PointerEvent& pointerEvent) override;
    virtual void onPointerReleased(const PointerEvent& pointerEvent) override;
    virtual void onMouseWheel(const PointerEvent& pointerEvent) override;

 private:
    Vector3D _origin;
    Vector3D _basisX = Vector3D(1, 0, 0);
    Vector3D _rotationAxis = Vector3D(0, 1, 0);
    Vector3D _basisZ = Vector3D(0, 0, 1);

    double _radialDistance = 1.0;
    double _polarAngleInRadians = pi<double>() / 2.0;
    double _azimuthalAngleInRadians = 0.0;

    double _rotateSpeed = 1.0;
    double _zoomSpeed = 1.0;
    double _panSpeed = 1.0;

    void updateCamera();
};

typedef std::shared_ptr<SphericalViewController> SphericalViewControllerPtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_SPHERICAL_VIEW_CONTROLLER_H_
