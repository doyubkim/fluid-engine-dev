// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_VIEW_CONTROLLER_H_
#define INCLUDE_JET_VIZ_VIEW_CONTROLLER_H_

#include <jet.viz/camera.h>
#include <jet.viz/event.h>
#include <jet.viz/input_events.h>

namespace jet {

namespace viz {

//! Abstract base class for view controllers.
class ViewController {
 public:
    //! Constructs a view controller with gievn camera.
    ViewController(const CameraPtr& newCamera);

    //! Default destructor.
    virtual ~ViewController();

    //! Called when key-down event is raised.
    void keyDown(const KeyEvent& keyEvent);

    //! Called when key-up event is raised.
    void keyUp(const KeyEvent& keyEvent);

    //! Called when pointer-pressed event is raised.
    void pointerPressed(const PointerEvent& pointerEvent);

    //! Called when pointer-hover event is raised.
    void pointerHover(const PointerEvent& pointerEvent);

    //! Called when pointer-dragged event is raised.
    void pointerDragged(const PointerEvent& pointerEvent);

    //! Called when pointer released event is raised.
    void pointerReleased(const PointerEvent& pointerEvent);

    //! Called when mouse-wheel event is raised.
    void mouseWheel(const PointerEvent& pointerEvent);

    //! Called when view is resized.
    void resize(const Viewport& viewport);

    //! Returns camera pointer.
    const CameraPtr& camera() const;

    //! Returns BasicCameraState changed event object.
    Event<ViewController*>& onBasicCameraStateChanged();

 protected:
    //! Called when key-down event is handled.
    virtual void onKeyDown(const KeyEvent& keyEvent);

    //! Called when key-up event is handled.
    virtual void onKeyUp(const KeyEvent& keyEvent);

    //! Called when pointer-pressed event is handled.
    virtual void onPointerPressed(const PointerEvent& pointerEvent);

    //! Called when pointer-hover event is handled.
    virtual void onPointerHover(const PointerEvent& pointerEvent);

    //! Called when pointer-dragged event is handled.
    virtual void onPointerDragged(const PointerEvent& pointerEvent);

    //! Called when pointer-released event is handled.
    virtual void onPointerReleased(const PointerEvent& pointerEvent);

    //! Called when mouse-wheel event is handled.
    virtual void onMouseWheel(const PointerEvent& pointerEvent);

    //! Called when resize event is handled.
    virtual void onResize(const Viewport& viewport);

    //! Sets the basic camera state of the current camera.
    void setBasicCameraState(const BasicCameraState& newState);

 private:
    Event<ViewController*> _basicCameraStateChangedEvent;
    CameraPtr _camera;
};

//! Shared pointer type for ViewController.
typedef std::shared_ptr<ViewController> ViewControllerPtr;

}  // namespace viz

}  // namespace jet

#endif  // INCLUDE_JET_VIZ_VIEW_CONTROLLER_H_
