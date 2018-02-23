// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VIZ_ORTHO_CAMERA_H_
#define INCLUDE_JET_VIZ_ORTHO_CAMERA_H_

#include "camera.h"

namespace jet {

namespace viz {

//! Orthographic camera representation.
class OrthoCamera : public Camera {
 public:
    //!
    //! \brief Default constructor.
    //!
    //! Newly create camera will be positioned at (0, 0, 1) with look-up vector
    //! (0, 1, 0) and look-at direction (0, 0, -1). The viewport will be
    //! covering [-1, 1] x [-1, 1].
    //!
    OrthoCamera();

    //!
    //! \brief Constructs an orthographic camera with given viewport parameters.
    //!
    //! Newly create camera will be positioned at (0, 0, 1) with look-up vector
    //! (0, 1, 0) and look-at direction (0, 0, -1).
    //!
    //! \param left Left side of the viewport.
    //! \param right Right side of the viewport.
    //! \param bottom Bottom side of the viewport.
    //! \param top Top side of the viewport.
    //!
    OrthoCamera(double left, double right, double bottom, double top);

    //!
    //! \brief Constructs an orthographic camera with given parameters.
    //!
    //! \param origin Position of the camera.
    //! \param lookAt Look-at directional vector.
    //! \param lookUp Look-up directional vector.
    //! \param nearClipPlane Near clip plane position.
    //! \param farClipPlane Far clip plane position.
    //! \param left Left side of the viewport.
    //! \param right Right side of the viewport.
    //! \param bottom Bottom side of the viewport.
    //! \param top Top side of the viewport.
    //!
    OrthoCamera(const Vector3D& origin, const Vector3D& lookAt,
                const Vector3D& lookUp, double nearClipPlane,
                double farClipPlane, double left, double right, double bottom,
                double top);

    //! Destructor.
    virtual ~OrthoCamera();

    //! Left side of the viewport.
    double left() const;

    //! Sets the left side of the viewport.
    void setLeft(double newLeft);

    //! Right side of the viewport.
    double right() const;

    //! Sets the right side of the viewport.
    void setRight(double newRight);

    //! Bottom side of the viewport.
    double bottom() const;

    //! Sets the bottom side of the viewport.
    void setBottom(double newBottom);

    //! Top side of the viewport.
    double top() const;

    //! Sets the top side of the viewport.
    void setTop(double newTop);

    //! Width of the viewport.
    double width() const;

    //! Height the viewport.
    double height() const;

    //! Center of the viewport.
    Vector2D center() const;

 protected:
    //! Called when the viewport has been resized.
    virtual void onResize(const Viewport& viewport) override;

    //! Updates the matrix.
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
