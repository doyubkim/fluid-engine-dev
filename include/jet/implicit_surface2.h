// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_IMPLICIT_SURFACE2_H_
#define INCLUDE_JET_IMPLICIT_SURFACE2_H_

#include <jet/surface2.h>

namespace jet {

//! Abstract base class for 2-D implicit surface.
class ImplicitSurface2 : public Surface2 {
 public:
    //! Default constructor.
    explicit ImplicitSurface2(bool isNormalFlipped = false);

    //! Copy constructor.
    ImplicitSurface2(const ImplicitSurface2& other);

    //! Default destructor.
    virtual ~ImplicitSurface2();

    //! Returns signed distance from the given point \p otherPoint.
    virtual double signedDistance(const Vector2D& otherPoint) const = 0;

    //! Returns closest distance from the given point \p otherPoint.
    double closestDistance(const Vector2D& otherPoint) const override;
};

//! Shared pointer type for the ImplicitSurface2.
typedef std::shared_ptr<ImplicitSurface2> ImplicitSurface2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE2_H_
