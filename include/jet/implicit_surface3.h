// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_IMPLICIT_SURFACE3_H_
#define INCLUDE_JET_IMPLICIT_SURFACE3_H_

#include <jet/surface3.h>

namespace jet {

class ImplicitSurface3 : public Surface3 {
 public:
    ImplicitSurface3();

    virtual ~ImplicitSurface3();

    virtual double signedDistance(const Vector3D& otherPoint) const = 0;

    double closestDistance(const Vector3D& otherPoint) const override;
};

typedef std::shared_ptr<ImplicitSurface3> ImplicitSurface3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE3_H_
