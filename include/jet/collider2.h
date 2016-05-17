// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_COLLIDER2_H_
#define INCLUDE_JET_COLLIDER2_H_

#include <jet/surface2.h>

namespace jet {

struct ColliderQueryResult2 final {
    double distance;
    Vector2D point;
    Vector2D normal;
    Vector2D velocity;
};

class Collider2 {
 public:
    Collider2();

    virtual ~Collider2();

    virtual Vector2D velocityAt(const Vector2D& point) const = 0;

    //! Resolves collision for given point.
    void resolveCollision(
        double radius,
        double restitutionCoefficient,
        Vector2D* newPosition,
        Vector2D* newVelocity);

    double frictionCoefficient() const;

    void setFrictionCoefficient(double newFrictionCoeffient);

    const Surface2Ptr& surface() const;

 protected:
    //! Assigns the surface instance from the subclass.
    void setSurface(const Surface2Ptr& newSurface);

    //! Outputs closest point's information.
    void getClosestPoint(
        const Surface2Ptr& surface,
        const Vector2D& queryPoint,
        ColliderQueryResult2* result) const;

    //! Returns true if given point is in the opposite side of the surface.
    bool isPenetrating(
        const ColliderQueryResult2& colliderPoint,
        const Vector2D& position,
        double radius);

 private:
    Surface2Ptr _surface;
    double _frictionCoeffient = 0.0;
};

typedef std::shared_ptr<Collider2> Collider2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER2_H_
