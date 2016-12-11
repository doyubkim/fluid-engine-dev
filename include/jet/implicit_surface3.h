// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_IMPLICIT_SURFACE3_H_
#define INCLUDE_JET_IMPLICIT_SURFACE3_H_

#include <jet/surface3.h>

namespace jet {

//! Abstract base class for 3-D implicit surface.
class ImplicitSurface3 : public Surface3 {
 public:
    //! Default constructor.
    explicit ImplicitSurface3(bool isNormalFlipped = false);

    //! Copy constructor.
    ImplicitSurface3(const ImplicitSurface3& other);

    //! Default destructor.
    virtual ~ImplicitSurface3();

    //! Returns signed distance from the given point \p otherPoint.
    virtual double signedDistance(const Vector3D& otherPoint) const = 0;

    //! Returns closest distance from the given point \p otherPoint.
    double closestDistance(const Vector3D& otherPoint) const override;
};

//! Shared pointer type for the ImplicitSurface3.
typedef std::shared_ptr<ImplicitSurface3> ImplicitSurface3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_IMPLICIT_SURFACE3_H_
