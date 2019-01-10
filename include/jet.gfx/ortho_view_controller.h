// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_GFX_ORTHO_VIEW_CONTROLLER_H_
#define INCLUDE_JET_GFX_ORTHO_VIEW_CONTROLLER_H_

#include <jet.gfx/ortho_camera.h>
#include <jet.gfx/view_controller.h>

namespace jet {
namespace gfx {

class OrthoViewController : public ViewController {
 public:
    bool preserveAspectRatio = true;
    bool enablePan = true;
    bool enableZoom = true;
    bool enableRotation = true;

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

 private:
    Vector3F _origin = Vector3F(0, 0, 1);
    Vector3F _basisX = Vector3F(1, 0, 0);
    Vector3F _basisY = Vector3F(0, 1, 0);

    float _viewRotateAngleInRadians = 0.0f;
    float _viewHeight = 2.0f;

    float _zoomSpeed = 1.0f;
    float _panSpeed = 1.0f;

    void updateCamera();
};

typedef std::shared_ptr<OrthoViewController> OrthoViewControllerPtr;

}  // namespace gfx
}  // namespace jet

#endif  // INCLUDE_JET_GFX_ORTHO_VIEW_CONTROLLER_H_
