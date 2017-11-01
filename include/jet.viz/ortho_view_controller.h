// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_ORTHO_VIEW_CONTROLLER_H_
#define INCLUDE_JET_VIZ_ORTHO_VIEW_CONTROLLER_H_

#include "ortho_camera.h"
#include "view_controller.h"

namespace jet {
namespace viz {

class OrthoViewController : public ViewController {
 public:
    bool preserveAspectRatio = true;

    OrthoViewController(const OrthoCameraPtr& camera);
    virtual ~OrthoViewController();

 protected:
    void onKeyDown(const KeyEvent& keyEvent) override;
    void onKeyUp(const KeyEvent& keyEvent) override;

    void onPointerPressed(const PointerEvent& pointerEvent) override;
    void onPointerHover(const PointerEvent& pointerEvent) override;
    void onPointerDragged(const PointerEvent& pointerEvent) override;
    void onPointerReleased(const PointerEvent& pointerEvent) override;
    void onMouseWheel(const PointerEvent& pointerEvent) override;

    void onResize(const Viewport& viewport) override;

 private:
    bool _enablePan = true;
    bool _enableZoom = true;
    bool _enableRotation = true;

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

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_ORTHO_VIEW_CONTROLLER_H_
