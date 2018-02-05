// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_QUATERNION_H_
#define INCLUDE_JET_QUATERNION_H_

#include <jet/matrix4x4.h>

namespace jet {

//!
//! \brief Quaternion class defined as q = w + xi + yj + zk.
//!
template <typename T>
class Quaternion {
 public:
    static_assert(
        std::is_floating_point<T>::value,
        "Quaternion only can be instantiated with floating point types");

    //! Real part.
    T w;

    //!< Imaginary part (i).
    T x;

    //!< Imaginary part (j).
    T y;

    //!< Imaginary part (k).
    T z;

    // MARK: Constructors

    //! Make an identity quaternion.
    Quaternion();

    //! Constructs a quaternion with given elements.
    Quaternion(T newW, T newX, T newY, T newZ);

    //! Constructs a quaternion with given elements.
    Quaternion(const std::initializer_list<T>& lst);

    //! Constructs a quaternion with given rotation axis and angle.
    Quaternion(const Vector3<T>& axis, T angle);

    //! Constructs a quaternion with from and to vectors.
    Quaternion(const Vector3<T>& from, const Vector3<T>& to);

    //! Constructs a quaternion with three basis vectors.
    Quaternion(
        const Vector3<T>& axis0,
        const Vector3<T>& axis1,
        const Vector3<T>& axis2);

    //! Constructs a quaternion with 3x3 rotational matrix.
    explicit Quaternion(const Matrix3x3<T>& m33);

    //! Copy constructor.
    Quaternion(const Quaternion& other);


    // MARK: Basic setters

    //! Sets the quaternion with other quaternion.
    void set(const Quaternion& other);

    //! Sets the quaternion with given elements.
    void set(T newW, T newX, T newY, T newZ);

    //! Sets the quaternion with given elements.
    void set(const std::initializer_list<T>& lst);

    //! Sets the quaternion with given rotation axis and angle.
    void set(const Vector3<T>& axis, T angle);

    //! Sets the quaternion with from and to vectors.
    void set(const Vector3<T>& from, const Vector3<T>& to);

    //! Sets quaternion with three basis vectors.
    void set(
        const Vector3<T>& rotationBasis0,
        const Vector3<T>& rotationBasis1,
        const Vector3<T>& rotationBasis2);

    //! Sets the quaternion with 3x3 rotational matrix.
    void set(const Matrix3x3<T>& matrix);


    // MARK: Basic getters

    //! Returns quaternion with other base type.
    template <typename U>
    Quaternion<U> castTo() const;

    //! Returns normalized quaternion.
    Quaternion normalized() const;


    // MARK: Binary operator methods - new instance = this instance (+) input

    //! Returns this quaternion * vector.
    Vector3<T> mul(const Vector3<T>& v) const;

    //! Returns this quaternion * other quaternion.
    Quaternion mul(const Quaternion& other) const;

    //! Computes the dot product with other quaternion.
    T dot(const Quaternion<T>& other);


    // MARK: Binary operator methods - new instance = input (+) this instance

    //! Returns other quaternion * this quaternion.
    Quaternion rmul(const Quaternion& other) const;

    // MARK: Augmented operator methods - this instance (+)= input

    //! Returns this quaternion *= other quaternion.
    void imul(const Quaternion& other);


    // MARK: Modifiers

    //! Makes this quaternion identity.
    void setIdentity();

    //! Rotate this quaternion with given angle in radians.
    void rotate(T angleInRadians);

    //! Normalizes the quaternion.
    void normalize();


    // MARK: Complex getters

    //! Returns the rotational axis.
    Vector3<T> axis() const;

    //! Returns the rotational angle.
    T angle() const;

    //! Returns the axis and angle.
    void getAxisAngle(Vector3<T>* axis, T* angle) const;

    //! Returns the inverse quaternion.
    Quaternion inverse() const;

    //! Converts to the 3x3 rotation matrix.
    Matrix3x3<T> matrix3() const;

    //! Converts to the 4x4 rotation matrix.
    Matrix4x4<T> matrix4() const;

    //! Returns L2 norm of this quaternion.
    T l2Norm() const;


    // MARK: Setter operators

    //! Assigns other quaternion.
    Quaternion& operator=(const Quaternion& other);

    //! Returns this quaternion *= other quaternion.
    Quaternion& operator*=(const Quaternion& other);


    // MARK: Getter operators

    //! Returns the reference to the i-th element.
    T& operator[](size_t i);

    //! Returns the const reference to the i-th element.
    const T& operator[](size_t i) const;

    //! Returns true if equal.
    bool operator==(const Quaternion& other) const;

    //! Returns true if not equal.
    bool operator!=(const Quaternion& other) const;


    // MARK: Builders

    //! Returns identity matrix.
    static Quaternion makeIdentity();
};

//! Computes spherical linear interpolation.
template <typename T>
Quaternion<T> slerp(
    const Quaternion<T>& a,
    const Quaternion<T>& b,
    T t);

//! Returns quaternion q * vector v.
template <typename T>
Vector<T, 3> operator*(const Quaternion<T>& q, const Vector<T, 3>& v);

//! Returns quaternion a times quaternion b.
template <typename T>
Quaternion<T> operator*(const Quaternion<T>& a, const Quaternion<T>& b);

//! Float-type quaternion.
typedef Quaternion<float> QuaternionF;

//! Double-type quaternion.
typedef Quaternion<double> QuaternionD;

}  // namespace jet

#include "detail/quaternion-inl.h"

#endif  // INCLUDE_JET_QUATERNION_H_
