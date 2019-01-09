// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_IMPLICIT_SURFACE2_H_
#define INCLUDE_JET_IMPLICIT_SURFACE2_H_

#include <jet/surface2.h>

namespace jet {

//! Abstract base class for 2-D implicit surface.
class ImplicitSurface2 : public Surface2 {
 public:
    //! Default constructor.
    ImplicitSurface2(
        const Transform2& transform = Transform2(),
        bool isNormalFlipped = false);

    //! Copy constructor.
    ImplicitSurface2(const ImplicitSurface2& other);

    //! Default destructor.
    virtual ~ImplicitSurface2();

    //! Returns signed distance from the given point \p otherPoint.
    double signedDistance(const Vector2D& otherPoint) const;

 protected:
    //! Returns signed distance from the given point \p otherPoint in local
    //! space.
    virtual double signedDistanceLocal(const Vector2D& otherPoint) const = 0;

 private:
    double closestDistanceLocal(const Vector2D& otherPoint) const override;

    bool isInsideLocal(const Vector2D& otherPoint) const override;
};

//! Shared pointer type for the ImplicitSurface2.
typedef std::shared_ptr<ImplicitSurface2> ImplicitSurface2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE2_H_
