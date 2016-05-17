// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_COLLIDER3_H_
#define INCLUDE_JET_COLLIDER3_H_

#include <jet/surface3.h>

namespace jet {

struct ColliderQueryResult3 final {
    double distance;
    Vector3D point;
    Vector3D normal;
    Vector3D velocity;
};

class Collider3 {
 public:
    Collider3();

    virtual ~Collider3();

    virtual Vector3D velocityAt(const Vector3D& point) const = 0;

    //! Resolves collision for given point.
    void resolveCollision(
        double radius,
        double restitutionCoefficient,
        Vector3D* newPosition,
        Vector3D* newVelocity);

    double frictionCoefficient() const;

    void setFrictionCoefficient(double newFrictionCoeffient);

    const Surface3Ptr& surface() const;

 protected:
    //! Assigns the surface instance from the subclass.
    void setSurface(const Surface3Ptr& newSurface);

    //! Outputs closest point's information.
    void getClosestPoint(
        const Surface3Ptr& surface,
        const Vector3D& queryPoint,
        ColliderQueryResult3* result) const;

    //! Returns true if given point is in the opposite side of the surface.
    bool isPenetrating(
        const ColliderQueryResult3& colliderPoint,
        const Vector3D& position,
        double radius);

 private:
    Surface3Ptr _surface;
    double _frictionCoeffient = 0.0;
};

typedef std::shared_ptr<Collider3> Collider3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER3_H_
