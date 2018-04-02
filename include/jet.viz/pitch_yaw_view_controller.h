// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_PITCH_YAW_VIEW_CONTROLLER_H_
#define INCLUDE_JET_VIZ_PITCH_YAW_VIEW_CONTROLLER_H_

#include "view_controller.h"

namespace jet {
namespace viz {

class PitchYawViewController : public ViewController {
 public:
    //!
    //! \brief Constructs pitch-yaw view controller.
    //!
    //! This constructor builds a pitch-yaw view controller with a camera and
    //! view controller's rotation origin. Because we need the origin of the
    //! rotation to define this view controller, which automatically defines the
    //! view direction, the look-at vector from the camera will be overriden.
    //! Also, since this controller only allows pitch and yaw, the look-up
    //! vector from the camera will also be overriden with calculated values
    //! internally.
    //!
    //! \param camera Camera object
    //! \param rotationOrigin Origin of the rotation of the view controller.
    //!
    PitchYawViewController(const CameraPtr& camera,
                           const Vector3D& rotationOrigin = Vector3D{0, 0, 0});
    virtual ~PitchYawViewController();

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

typedef std::shared_ptr<PitchYawViewController> PitchYawViewControllerPtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_PITCH_YAW_VIEW_CONTROLLER_H_
