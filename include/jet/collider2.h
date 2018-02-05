// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_COLLIDER2_H_
#define INCLUDE_JET_COLLIDER2_H_

#include <jet/surface2.h>
#include <functional>

namespace jet {

//!
//! \brief Abstract base class for generic collider object.
//!
//! This class contains basic interfaces for colliders. Most of the
//! functionalities are implemented within this class, except the member
//! function Collider2::velocityAt. This class also let the subclasses to
//! provide a Surface2 instance to define collider surface using
//! Collider2::setSurface function.
//!
class Collider2 {
 public:
    //!
    //! \brief Callback function type for update calls.
    //!
    //! This type of callback function will take the collider pointer, current
    //! time, and time interval in seconds.
    //!
    typedef std::function<void(Collider2*, double, double)>
        OnBeginUpdateCallback;

    //! Default constructor.
    Collider2();

    //! Default destructor.
    virtual ~Collider2();

    //! Returns the velocity of the collider at given \p point.
    virtual Vector2D velocityAt(const Vector2D& point) const = 0;

    //!
    //! Resolves collision for given point.
    //!
    //! \param radius Radius of the colliding point.
    //! \param restitutionCoefficient Defines the restitution effect.
    //! \param position Input and output position of the point.
    //! \param position Input and output velocity of the point.
    //!
    void resolveCollision(
        double radius,
        double restitutionCoefficient,
        Vector2D* position,
        Vector2D* velocity);

    //! Returns friction coefficent.
    double frictionCoefficient() const;

    //!
    //! \brief Sets the friction coefficient.
    //!
    //! This function assigns the friction coefficient to the collider. Any
    //! negative inputs will be clamped to zero.
    //!
    void setFrictionCoefficient(double newFrictionCoeffient);

    //! Returns the surface instance.
    const Surface2Ptr& surface() const;

    //! Updates the collider state.
    void update(double currentTimeInSeconds, double timeIntervalInSeconds);

    //!
    //! \brief      Sets the callback function to be called when
    //!             Collider2::update function is invoked.
    //!
    //! The callback function takes current simulation time in seconds unit. Use
    //! this callback to track any motion or state changes related to this
    //! collider.
    //!
    //! \param[in]  callback The callback function.
    //!
    void setOnBeginUpdateCallback(const OnBeginUpdateCallback& callback);

 protected:
    //! Internal query result structure.
    struct ColliderQueryResult final {
        double distance;
        Vector2D point;
        Vector2D normal;
        Vector2D velocity;
    };

    //! Assigns the surface instance from the subclass.
    void setSurface(const Surface2Ptr& newSurface);

    //! Outputs closest point's information.
    void getClosestPoint(
        const Surface2Ptr& surface,
        const Vector2D& queryPoint,
        ColliderQueryResult* result) const;

    //! Returns true if given point is in the opposite side of the surface.
    bool isPenetrating(
        const ColliderQueryResult& colliderPoint,
        const Vector2D& position,
        double radius);

 private:
    Surface2Ptr _surface;
    double _frictionCoeffient = 0.0;
    OnBeginUpdateCallback _onUpdateCallback;
};

//! Shared pointer type for the Collider2.
typedef std::shared_ptr<Collider2> Collider2Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER2_H_
