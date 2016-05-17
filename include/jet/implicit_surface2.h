// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_IMPLICIT_SURFACE2_H_
#define INCLUDE_JET_IMPLICIT_SURFACE2_H_

#include <jet/surface2.h>

namespace jet {

class ImplicitSurface2 : public Surface2 {
 public:
    ImplicitSurface2();

    virtual ~ImplicitSurface2();

    virtual double signedDistance(const Vector2D& otherPoint) const = 0;

    double closestDistance(const Vector2D& otherPoint) const override;
};

typedef std::shared_ptr<ImplicitSurface2> ImplicitSurface2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE2_H_
