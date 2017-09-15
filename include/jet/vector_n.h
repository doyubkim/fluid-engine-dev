// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VECTOR_N_H_
#define INCLUDE_JET_VECTOR_N_H_

#include <jet/array_accessor1.h>
#include <jet/vector_expression.h>

#include <initializer_list>
#include <limits>
#include <vector>

namespace jet {

// MARK: VectorN

//!
//! \brief General purpose dynamically-sizedN-D vector class.
//!
//! This class defines N-D vector data where its size can be defined
//! dynamically.
//!
//! \tparam T Type of the element.
//!
template <typename T>
class VectorN final : public VectorExpression<T, VectorN<T>> {
 public:
    static_assert(std::is_floating_point<T>::value,
                  "VectorN only can be instantiated with floating point types");

    typedef std::vector<T> ContainerType;

    // MARK: Constructors

    //! Constructs empty vector.
    VectorN();

    //! Constructs default vector (val, val, ... , val).
    VectorN(size_t n, const T& val = 0);

    //! Constructs vector with given initializer list.
    template <typename U>
    VectorN(const std::initializer_list<U>& lst);

    //! Constructs vector with expression template.
    template <typename E>
    VectorN(const VectorExpression<T, E>& other);

    //! Copy constructor.
    VectorN(const VectorN& other);

    //! Move constructor.
    VectorN(VectorN&& other);

    // MARK: Basic setters

    //! Resizes to \p n dimensional vector with initial value \p val.
    void resize(size_t n, const T& val = 0);

    //! Sets all elements to \p s.
    void set(const T& s);

    //! Sets all elements with given initializer list.
    template <typename U>
    void set(const std::initializer_list<U>& lst);

    //! Sets vector with expression template.
    template <typename E>
    void set(const VectorExpression<T, E>& other);

    //! Swaps the content of the vector with \p other vector.
    void swap(VectorN& other);

    //! Sets all elements to zero.
    void setZero();

    //! Normalizes this vector.
    void normalize();

    // MARK: Basic getters

    //! Returns the size of the vector.
    size_t size() const;

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
    VectorScalarDiv<T, VectorN> normalized() const;

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
    VectorTypeCast<U, VectorN<T>, T> castTo() const;

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
    VectorAdd<T, VectorN, E> add(const E& v) const;

    //! Computes this + (s, s, ... , s).
    VectorScalarAdd<T, VectorN> add(const T& s) const;

    //! Computes this - v.
    template <typename E>
    VectorSub<T, VectorN, E> sub(const E& v) const;

    //! Computes this - (s, s, ... , s).
    VectorScalarSub<T, VectorN> sub(const T& s) const;

    //! Computes this * v.
    template <typename E>
    VectorMul<T, VectorN, E> mul(const E& v) const;

    //! Computes this * (s, s, ... , s).
    VectorScalarMul<T, VectorN> mul(const T& s) const;

    //! Computes this / v.
    template <typename E>
    VectorDiv<T, VectorN, E> div(const E& v) const;

    //! Computes this / (s, s, ... , s).
    VectorScalarDiv<T, VectorN> div(const T& s) const;

    //! Computes dot product.
    template <typename E>
    T dot(const E& v) const;

    // MARK: Binary operations: new instance = v (+) this

    //! Computes (s, s, ... , s) - this.
    VectorScalarRSub<T, VectorN> rsub(const T& s) const;

    //! Computes v - this.
    template <typename E>
    VectorSub<T, VectorN, E> rsub(const E& v) const;

    //! Computes (s, s, ... , s) / this.
    VectorScalarRDiv<T, VectorN> rdiv(const T& s) const;

    //! Computes v / this.
    template <typename E>
    VectorDiv<T, VectorN, E> rdiv(const E& v) const;

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
    //! VectorN<float> vec(10, 4.f);
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
    //! VectorN<float> vec(10, 4.f);
    //! vec.forEachIndex([&](size_t i) {
    //!     vec[i] = 4.f * i + 1.5f;
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void forEachIndex(Callback func) const;

    //!
    //! \brief Iterates the vector and invoke given \p func for each element in
    //!     parallel using multi-threading.
    //!
    //! This function iterates the vector elements and invoke the callback
    //! function \p func in parallel using multi-threading. The callback
    //! function takes vector's element as its input. The order of execution
    //! will be non-deterministic since it runs in parallel.
    //! Below is the sample usage:
    //!
    //! \code{.cpp}
    //! VectorN<float> vec(1000, 4.f);
    //! vec.parallelForEach([](float& elem) {
    //!     elem *= 2;
    //! });
    //! \endcode
    //!
    //! The parameter type of the callback function doesn't have to be T&, but
    //! const T& or T can be used as well.
    //!
    template <typename Callback>
    void parallelForEach(Callback func);

    //!
    //! \brief Iterates the vector and invoke given \p func for each index in
    //!     parallel using multi-threading.
    //!
    //! This function iterates the vector elements and invoke the callback
    //! function \p func in parallel using multi-threading. The callback
    //! function takes one parameter which is the index of the vector. The order
    //! of execution will be non-deterministic since it runs in parallel.
    //! Below is the sample usage:
    //!
    //! \code{.cpp}
    //! VectorN<float> vec(1000, 4.f);
    //! vec.parallelForEachIndex([](size_t i) {
    //!     array[i] *= 2;
    //! });
    //! \endcode
    //!
    template <typename Callback>
    void parallelForEachIndex(Callback func) const;

    //! Returns the \p i -th element.
    T operator[](size_t i) const;

    //! Returns the reference to the \p i -th element.
    T& operator[](size_t i);

    //! Sets vector with given initializer list.
    template <typename U>
    VectorN& operator=(const std::initializer_list<U>& lst);

    //! Sets vector with expression template.
    template <typename E>
    VectorN& operator=(const VectorExpression<T, E>& other);

    //! Copy assignment.
    VectorN& operator=(const VectorN& other);

    //! Move assignment.
    VectorN& operator=(VectorN&& other);

    //! Computes this += (s, s, ... , s)
    VectorN& operator+=(const T& s);

    //! Computes this += v
    template <typename E>
    VectorN& operator+=(const E& v);

    //! Computes this -= (s, s, ... , s)
    VectorN& operator-=(const T& s);

    //! Computes this -= v
    template <typename E>
    VectorN& operator-=(const E& v);

    //! Computes this *= (s, s, ... , s)
    VectorN& operator*=(const T& s);

    //! Computes this *= v
    template <typename E>
    VectorN& operator*=(const E& v);

    //! Computes this /= (s, s, ... , s)
    VectorN& operator/=(const T& s);

    //! Computes this /= v
    template <typename E>
    VectorN& operator/=(const E& v);

    //! Returns true if \p other is the same as this vector.
    template <typename E>
    bool operator==(const E& v) const;

    //! Returns true if \p other is the not same as this vector.
    template <typename E>
    bool operator!=(const E& v) const;

 private:
    ContainerType _elements;
};

//! Float-type N-D vector.
typedef VectorN<float> VectorNF;

//! Double-type N-D vector.
typedef VectorN<double> VectorND;

}  // namespace jet

#include "detail/vector_n-inl.h"

#endif  // INCLUDE_JET_VECTOR_N_H_
