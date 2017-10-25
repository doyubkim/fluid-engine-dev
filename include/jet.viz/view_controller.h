// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_VIEW_CONTROLLER_H_
#define INCLUDE_JET_VIZ_VIEW_CONTROLLER_H_

#include <jet.viz/event.h>
#include <jet.viz/camera.h>
#include <jet.viz/input_events.h>

namespace jet { namespace viz {

class ViewController {
 public:
    ViewController(const CameraPtr& newCamera);
    virtual ~ViewController();

    void keyDown(const KeyEvent& keyEvent);
    void keyUp(const KeyEvent& keyEvent);

    void pointerPressed(const PointerEvent& pointerEvent);
    void pointerHover(const PointerEvent& pointerEvent);
    void pointerDragged(const PointerEvent& pointerEvent);
    void pointerReleased(const PointerEvent& pointerEvent);
    void mouseWheel(const PointerEvent& pointerEvent);

    const CameraPtr& camera() const;

    Event<ViewController*>& onBasicCameraStateChanged();

 protected:
    virtual void onKeyDown(const KeyEvent& keyEvent);
    virtual void onKeyUp(const KeyEvent& keyEvent);

    virtual void onPointerPressed(const PointerEvent& pointerEvent);
    virtual void onPointerHover(const PointerEvent& pointerEvent);
    virtual void onPointerDragged(const PointerEvent& pointerEvent);
    virtual void onPointerReleased(const PointerEvent& pointerEvent);
    virtual void onMouseWheel(const PointerEvent& pointerEvent);

    void setBasicCameraState(const BasicCameraState& newState);

 private:
    Event<ViewController*> _basicCameraStateChangedEvent;
    CameraPtr _camera;
};

typedef std::shared_ptr<ViewController> ViewControllerPtr;

} }  // namespace jet::viz

#endif  // INCLUDE_JET_VIZ_VIEW_CONTROLLER_H_
