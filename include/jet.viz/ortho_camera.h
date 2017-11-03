// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_ORTHO_CAMERA_H_
#define INCLUDE_JET_VIZ_ORTHO_CAMERA_H_

#include "camera.h"

namespace jet {
namespace viz {

class OrthoCamera : public Camera {
 public:
    OrthoCamera();

    OrthoCamera(double left, double right, double bottom, double top);

    OrthoCamera(const Vector3D& origin, const Vector3D& lookAt,
                const Vector3D& lookUp, double nearClipPlane,
                double farClipPlane, double left, double right, double bottom,
                double top);

    virtual ~OrthoCamera();

    double left() const;

    void setLeft(double newLeft);

    double right() const;

    void setRight(double newRight);

    double bottom() const;

    void setBottom(double newBottom);

    double top() const;

    void setTop(double newTop);

    double width() const;

    double height() const;

    Vector2D center() const;

 protected:
    virtual void onResize(const Viewport& viewport) override;

    virtual void updateMatrix() override;

 private:
    double _left;
    double _right;
    double _bottom;
    double _top;
};

typedef std::shared_ptr<OrthoCamera> OrthoCameraPtr;

}  // namespace viz
}  // namespace jet

#endif  // INCLUDE_JET_VIZ_ORTHO_CAMERA_H_
