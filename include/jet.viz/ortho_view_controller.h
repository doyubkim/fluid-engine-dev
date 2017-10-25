// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_ORTHO_VIEW_CONTROLLER_H_
#define INCLUDE_JET_VIZ_ORTHO_VIEW_CONTROLLER_H_

#include "ortho_camera.h"
#include "view_controller.h"

namespace jet { namespace viz {

class OrthoViewController : public ViewController {
 public:
    OrthoViewController(const OrthoCameraPtr& camera);
    virtual ~OrthoViewController();

 protected:
    virtual void onKeyDown(const KeyEvent& keyEvent) override;
    virtual void onKeyUp(const KeyEvent& keyEvent) override;

    virtual void onPointerPressed(const PointerEvent& pointerEvent) override;
    virtual void onPointerHover(const PointerEvent& pointerEvent) override;
    virtual void onPointerDragged(const PointerEvent& pointerEvent) override;
    virtual void onPointerReleased(const PointerEvent& pointerEvent) override;
    virtual void onMouseWheel(const PointerEvent& pointerEvent) override;

 private:
    Vector3D _origin = Vector3D(0, 0, 1);
    Vector3D _basisX = Vector3D(1, 0, 0);
    Vector3D _basisY = Vector3D(0, 1, 0);

    double _viewRotateAngleInRadians = 0.0;
    double _viewHeight = 2.0;

    double _zoomSpeed = 1.0;
    double _panSpeed = 1.0;

    void updateCamera();
};

typedef std::shared_ptr<OrthoViewController> OrthoViewControllerPtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_ORTHO_VIEW_CONTROLLER_H_
