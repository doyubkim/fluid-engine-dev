// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_COLLIDER_H_
#define INCLUDE_JET_COLLIDER_H_

#include <jet/surface.h>

#include <functional>

namespace jet {

//!
//! \brief Abstract base class for generic collider object.
//!
//! This class contains basic interfaces for colliders. Most of the
//! functionalities are implemented within this class, except the member
//! function Collider::velocityAt. This class also let the subclasses to
//! provide a Surface instance to define collider surface using
//! Collider::setSurface function.
//!
template <size_t N>
class Collider {
 public:
    //!
    //! \brief Callback function type for update calls.
    //!
    //! This type of callback function will take the collider pointer, current
    //! time, and time interval in seconds.
    //!
    typedef std::function<void(Collider*, double, double)>
        OnBeginUpdateCallback;

    //! Default constructor.
    Collider();

    //! Default destructor.
    virtual ~Collider();

    //! Returns the velocity of the collider at given \p point.
    virtual Vector<double, N> velocityAt(
        const Vector<double, N>& point) const = 0;

    //!
    //! Resolves collision for given point.
    //!
    //! \param radius Radius of the colliding point.
    //! \param restitutionCoefficient Defines the restitution effect.
    //! \param position Input and output position of the point.
    //! \param position Input and output velocity of the point.
    //!
    void resolveCollision(double radius, double restitutionCoefficient,
                          Vector<double, N>* position,
                          Vector<double, N>* velocity);

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
    const std::shared_ptr<Surface<N>>& surface() const;

    //! Updates the collider state.
    void update(double currentTimeInSeconds, double timeIntervalInSeconds);

    //!
    //! \brief      Sets the callback function to be called when
    //!             Collider::update function is invoked.
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
        Vector<double, N> point;
        Vector<double, N> normal;
        Vector<double, N> velocity;
    };

    //! Assigns the surface instance from the subclass.
    void setSurface(const std::shared_ptr<Surface<N>>& newSurface);

    //! Outputs closest point's information.
    void getClosestPoint(const std::shared_ptr<Surface<N>>& surface,
                         const Vector<double, N>& queryPoint,
                         ColliderQueryResult* result) const;

    //! Returns true if given point is in the opposite side of the surface.
    bool isPenetrating(const ColliderQueryResult& colliderPoint,
                       const Vector<double, N>& position, double radius);

 private:
    std::shared_ptr<Surface<N>> _surface;
    double _frictionCoeffient = 0.0;
    OnBeginUpdateCallback _onUpdateCallback;
};

//! 2-D collider type.
using Collider2 = Collider<2>;

//! 3-D collider type.
using Collider3 = Collider<3>;

//! Shared pointer type for the Collider2.
using Collider2Ptr = std::shared_ptr<Collider2>;

//! Shared pointer type for the Collider3.
using Collider3Ptr = std::shared_ptr<Collider3>;

}  // namespace jet

#endif  // INCLUDE_JET_COLLIDER_H_
