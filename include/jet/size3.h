// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SIZE3_H_
#define INCLUDE_JET_SIZE3_H_

#include <jet/size2.h>

namespace jet {

template <>
class Size<3> final {
 public:
    size_t x;
    size_t y;
    size_t z;

    // Constructors
    Size();
    explicit Size(size_t x, size_t y, size_t z);
    explicit Size(const Size2& pt, size_t z);
    explicit Size(const std::initializer_list<size_t>& lst);
    Size(const Size& v);

    // Basic setters
    void set(size_t x, size_t y, size_t z);
    void set(const Size2& pt, size_t z);
    void set(const std::initializer_list<size_t>& lst);
    void set(const Size& v);

    void setZero();

    // Binary operations: new instance = this (+) v
    Size add(size_t v) const;
    Size add(const Size& v) const;
    Size sub(size_t v) const;
    Size sub(const Size& v) const;
    Size mul(size_t v) const;
    Size mul(const Size& v) const;
    Size div(size_t v) const;
    Size div(const Size& v) const;

    // Binary operations: new instance = v (+) this
    Size radd(size_t v) const;
    Size radd(const Size& v) const;
    Size rsub(size_t v) const;
    Size rsub(const Size& v) const;
    Size rmul(size_t v) const;
    Size rmul(const Size& v) const;
    Size rdiv(size_t v) const;
    Size rdiv(const Size& v) const;

    // Augmented operations: this (+)= v
    void iadd(size_t v);
    void iadd(const Size& v);
    void isub(size_t v);
    void isub(const Size& v);
    void imul(size_t v);
    void imul(const Size& v);
    void idiv(size_t v);
    void idiv(const Size& v);

    // Basic getters
    const size_t& at(size_t i) const;
    size_t& at(size_t i);
    size_t sum() const;
    size_t min() const;
    size_t max() const;
    size_t dominantAxis() const;
    size_t subminantAxis() const;

    bool isEqual(const Size& other) const;

    // Operators
    size_t& operator[](size_t i);
    const size_t& operator[](size_t i) const;

    Size& operator=(const Size& v);
    Size& operator+=(size_t v);
    Size& operator+=(const Size& v);
    Size& operator-=(size_t v);
    Size& operator-=(const Size& v);
    Size& operator*=(size_t v);
    Size& operator*=(const Size& v);
    Size& operator/=(size_t v);
    Size& operator/=(const Size& v);

    bool operator==(const Size& v) const;
    bool operator!=(const Size& v) const;
};

typedef Size<3> Size3;

inline Size3 operator+(const Size3& a);

inline Size3 operator+(size_t a, const Size3& b);

inline Size3 operator+(const Size3& a, const Size3& b);

inline Size3 operator-(const Size3& a, size_t b);

inline Size3 operator-(size_t a, const Size3& b);

inline Size3 operator-(const Size3& a, const Size3& b);

inline Size3 operator*(const Size3& a, size_t b);

inline Size3 operator*(size_t a, const Size3& b);

inline Size3 operator*(const Size3& a, const Size3& b);

inline Size3 operator/(const Size3& a, size_t b);

inline Size3 operator/(size_t a, const Size3& b);

inline Size3 operator/(const Size3& a, const Size3& b);

inline Size3 min(const Size3& a, const Size3& b);

inline Size3 max(const Size3& a, const Size3& b);

inline Size3 clamp(const Size3& v, const Size3& low, const Size3& high);

}  // namespace jet

#include "detail/size3-inl.h"

#endif  // INCLUDE_JET_SIZE3_H_

