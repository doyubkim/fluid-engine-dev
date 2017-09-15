// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VECTOR_H_
#define INCLUDE_JET_VECTOR_H_

#include <jet/array_accessor1.h>
#include <jet/constants.h>
#include <jet/type_helpers.h>
#include <jet/vector_expression.h>

#include <array>
#include <type_traits>

namespace jet {

//!
//! \brief Generic statically-sized N-D vector class.
//!
//! This class defines N-D vector data where its size is determined statically
//! at compile time.
//!
//! \tparam T - Number type.
//! \tparam N - Dimension.
//!

template <typename T, size_t N>
class Vector final : public VectorExpression<T, Vector<T, N>> {
 public:
    static_assert(N > 0,
                  "Size of static-sized vector should be greater than zero.");
    static_assert(std::is_floating_point<T>::value,
                  "Vector only can be instantiated with floating point types");

    typedef std::array<T, N> ContainerType;

    //! Constructs a vector with zeros.
    Vector();

    //! Constructs vector instance with parameters.
    template <typename... Params>
    explicit Vector(Params... params);

    //! Sets all elements to \p s.
    void set(const T& s);

    //! Constructs vector instance with initializer list.
    template <typename U>
    Vector(const std::initializer_list<U>& lst);

    //! Constructs vector with expression template.
    template <typename E>
    Vector(const VectorExpression<T, E>& other);

    //! Copy constructor.
    Vector(const Vector& other);

    // MARK: Basic setters

    //! Set vector instance with initializer list.
    template <typename U>
    void set(const std::initializer_list<U>& lst);

    //! Sets vector with expression template.
    template <typename E>
    void set(const VectorExpression<T, E>& other);

    //! Swaps the content of the vector with \p other vector.
    void swap(Vector& other);

    //! Sets all elements to zero.
    void setZero();

    //! Normalizes this vector.
    void normalize();

    // MARK: Basic getters

    //! Returns the size of the vector.
    constexpr size_t size() const;

    //! Returns the raw pointer to the vector data.
    T* data();

    //! Returns the const raw pointer to the vector data.
    const T* const data() const;

    //! Returns the begin iterator of the vector.
    typename ContainerType::iterator begin();

    //! Returns the begin const iterator of the vector.
    typename ContainerType::const_iterator begin() const;

    //! Returns the end iterator of the vector.
    typename ContainerType::iterator end();

    //! Returns the end const iterator of the vector.
    typename ContainerType::const_iterator end() const;

    //! Returns the array accessor.
    ArrayAccessor1<T> accessor();

    //! Returns the const array accessor.
    ConstArrayAccessor1<T> constAccessor() const;

    //! Returns const reference to the \p i -th element of the vector.
    T at(size_t i) const;

    //! Returns reference to the \p i -th element of the vector.
    T& at(size_t i);

    //! Returns the sum of all the elements.
    T sum() const;

    //! Returns the average of all the elements.
    T avg() const;

    //! Returns the minimum element.
    T min() const;

    //! Returns the maximum element.
    T max() const;

    //! Returns the absolute minimum element.
    T absmin() const;

    //! Returns the absolute maximum element.
    T absmax() const;

    //! Returns the index of the dominant axis.
    size_t dominantAxis() const;

    //! Returns the index of the subminant axis.
    size_t subminantAxis() const;

    //! Returns normalized vector.
    VectorScalarDiv<T, Vector> normalized() const;

    //! Returns the length of the vector.
    T length() const;

    //! Returns the squared length of the vector.
    T lengthSquared() const;

    //! Returns the distance to the other vector.
    template <typename E>
    T distanceTo(const E& other) const;

    //! Returns the squared distance to the other vector.
    template <typename E>
    T distanceSquaredTo(const E& other) const;

    //! Returns a vector with different value type.
    template <typename U>
    VectorTypeCast<U, Vector<T, N>, T> castTo() const;

    //! Returns true if \p other is the same as this vector.
    template <typename E>
    bool isEqual(const E& other) const;

    //! Returns true if \p other is similar to this vector.
    template <typename E>
    bool isSimilar(const E& other,
                   T epsilon = std::numeric_limits<T>::epsilon()) const;

    // MARK: Binary operations: new instance = this (+) v

    //! Computes this + v.
    template <typename E>
    VectorAdd<T, Vector, E> add(const E& v) const;

    //! Computes this + (s, s, ... , s).
    VectorScalarAdd<T, Vector> add(const T& s) const;

    //! Computes this - v.
    template <typename E>
    VectorSub<T, Vector, E> sub(const E& v) const;

    //! Computes this - (s, s, ... , s).
    VectorScalarSub<T, Vector> sub(const T& s) const;

    //! Computes this * v.
    template <typename E>
    VectorMul<T, Vector, E> mul(const E& v) const;

    //! Computes this * (s, s, ... , s).
    VectorScalarMul<T, Vector> mul(const T& s) const;

    //! Computes this / v.
    template <typename E>
    VectorDiv<T, Vector, E> div(const E& v) const;

    //! Computes this / (s, s, ... , s).
    VectorScalarDiv<T, Vector> div(const T& s) const;

    //! Computes dot product.
    template <typename E>
    T dot(const E& v) const;

    // MARK: Binary operations: new instance = v (+) this

    //! Computes (s, s, ... , s) - this.
    VectorScalarRSub<T, Vector> rsub(const T& s) const;

    //! Computes v - this.
    template <typename E>
    VectorSub<T, Vector, E> rsub(const E& v) const;

    //! Computes (s, s, ... , s) / this.
    VectorScalarRDiv<T, Vector> rdiv(const T& s) const;

    //! Computes v / this.
    template <typename E>
    VectorDiv<T, Vector, E> rdiv(const E& v) const;

    // MARK: Augmented operations: this (+)= v

    //! Computes this += (s, s, ... , s).
    void iadd(const T& s);

    //! Computes this += v.
    template <typename E>
    void iadd(const E& v);

    //! Computes this -= (s, s, ... , s).
    void isub(const T& s);

    //! Computes this -= v.
    template <typename E>
    void isub(const E& v);

    //! Computes this *= (s, s, ... , s).
    void imul(const T& s);

    //! Computes this *= v.
    template <typename E>
    void imul(const E& v);

    //! Computes this /= (s, s, ... , s).
    void idiv(const T& s);

    //! Computes this /= v.
    template <typename E>
    void idiv(const E& v);

    // MARK: Operators

    //!
    //! \brief Iterates the vector and invoke given \p func for each element.
    //!
    //! This function iterates the vector elements and invoke the callback
    //! function \p func. The callback function takes array's element as its
    //! input. The order of execution will be 0 to N-1 where N is the size of
    //! the vector. Below is the sample usage:
    //!
    //! \code{.cpp}
    //! Vector<float, 2> vec(10, 4.f);
    //! vec.forEach([](float elem) {
    //!     printf("%d\n", elem);
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEach(Callback func) const;

    //!
    //! \brief Iterates the vector and invoke given \p func for each index.
    //!
    //! This function iterates the vector elements and invoke the callback
    //! function \p func. The callback function takes one parameter which is the
    //! index of the vector. The order of execution will be 0 to N-1 where N is
    //! the size of the array. Below is the sample usage:
    //!
    //! \code{.cpp}
    //! Vector<float, 2> vec(10, 4.f);
    //! vec.forEachIndex([&](size_t i) {
    //!     vec[i] = 4.f * i + 1.5f;
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEachIndex(Callback func) const;

    //! Returns the const reference to the \p i -th element.
    const T& operator[](size_t i) const;

    //! Returns the reference to the \p i -th element.
    T& operator[](size_t);

    //! Set vector instance with initializer list.
    template <typename U>
    Vector& operator=(const std::initializer_list<U>& lst);

    //! Sets vector with expression template.
    template <typename E>
    Vector& operator=(const VectorExpression<T, E>& other);

    //! Computes this += (s, s, ... , s)
    Vector& operator+=(const T& s);

    //! Computes this += v
    template <typename E>
    Vector& operator+=(const E& v);

    //! Computes this -= (s, s, ... , s)
    Vector& operator-=(const T& s);

    //! Computes this -= v
    template <typename E>
    Vector& operator-=(const E& v);

    //! Computes this *= (s, s, ... , s)
    Vector& operator*=(const T& s);

    //! Computes this *= v
    template <typename E>
    Vector& operator*=(const E& v);

    //! Computes this /= (s, s, ... , s)
    Vector& operator/=(const T& s);

    //! Computes this /= v
    template <typename E>
    Vector& operator/=(const E& v);

    //! Returns true if \p other is the same as this vector.
    template <typename E>
    bool operator==(const E& v) const;

    //! Returns true if \p other is the not same as this vector.
    template <typename E>
    bool operator!=(const E& v) const;

 private:
    ContainerType _elements;

    template <typename... Params>
    void setAt(size_t i, T v, Params... params);
    void setAt(size_t i, T v);
};

//! Returns the type of the value.
template <typename T, size_t N>
struct ScalarType<Vector<T, N>> {
    typedef T value;
};

}  // namespace jet

#include "detail/vector-inl.h"

#endif  // INCLUDE_JET_VECTOR_H_
