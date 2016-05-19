// Copyright (c) 2016 Doyub Kim

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

    T w;  //!< Real part.
    T x;  //!< Imaginary part (i).
    T y;  //!< Imaginary part (j).
    T z;  //!< Imaginary part (k).

    // Constructors
    Quaternion();
    explicit Quaternion(T newW, T newX, T newY, T newZ);
    explicit Quaternion(const std::initializer_list<T>& lst);
    explicit Quaternion(const Vector3<T>& axis, T angle);
    explicit Quaternion(const Vector3<T>& from, const Vector3<T>& to);
    explicit Quaternion(
        const Vector3<T>& axis0,
        const Vector3<T>& axis1,
        const Vector3<T>& axis2);
    explicit Quaternion(const Matrix3x3<T>& m33);
    Quaternion(const Quaternion& other);


    // Basic setters
    void set(const Quaternion& other);
    void set(T newW, T newX, T newY, T newZ);
    void set(const std::initializer_list<T>& lst);
    void set(const Vector3<T>& axis, T angle);
    void set(const Vector3<T>& from, const Vector3<T>& to);
    void set(
        const Vector3<T>& rotationBasis0,
        const Vector3<T>& rotationBasis1,
        const Vector3<T>& rotationBasis2);
    void set(const Matrix3x3<T>& matrix);


    // Basic getters
    template <typename U>
    Quaternion<U> castTo() const;

    Quaternion normalized() const;


    // Binary operator methods - new instance = this instance (+) input
    Vector3<T> mul(const Vector3<T>& v) const;

    Quaternion mul(const Quaternion& other) const;

    T dot(const Quaternion<T>& other);


    // Binary operator methods - new instance = input (+) this instance
    Quaternion rmul(const Quaternion& other) const;

    // Augmented operator methods - this instance (+)= input
    void imul(const Quaternion& other);


    // Modifiers
    void makeIdentity();

    void rotate(T angleInRadians);

    void normalize();


    // Complex getters
    Vector3<T> axis() const;

    T angle() const;

    void getAxisAngle(Vector3<T>* axis, T* angle) const;

    Quaternion inverse() const;

    Matrix3x3<T> matrix3() const;

    Matrix4x4<T> matrix4() const;

    T l2Norm() const;


    // Setter operators
    Quaternion& operator=(const Quaternion& other);

    Quaternion& operator*=(const Quaternion& other);


    // Getter operators
    T& operator[](size_t i);

    const T& operator[](size_t i) const;

    bool operator==(const Quaternion& other) const;

    bool operator!=(const Quaternion& other) const;
};

template <typename T>
Quaternion<T> slerp(
    const Quaternion<T>& a,
    const Quaternion<T>& b,
    T t);

template <typename T>
Vector<T, 3> operator*(const Quaternion<T>& q, const Vector<T, 3>& v);

template <typename T>
Quaternion<T> operator*(const Quaternion<T>& a, const Quaternion<T>& b);

typedef Quaternion<float> QuaternionF;
typedef Quaternion<double> QuaternionD;

}  // namespace jet

#include "detail/quaternion-inl.h"

#endif  // INCLUDE_JET_QUATERNION_H_
