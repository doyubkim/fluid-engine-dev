// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ARRAY_ACCESSOR3_H_
#define INCLUDE_JET_ARRAY_ACCESSOR3_H_

#include <jet/array_accessor.h>
#include <jet/point3.h>
#include <jet/size3.h>

namespace jet {

template <typename T>
class ArrayAccessor<T, 3> final {
 public:
    ArrayAccessor();
    explicit ArrayAccessor(const Size3& size, T* const data);
    explicit ArrayAccessor(
        size_t width, size_t height, size_t depth, T* const data);
    ArrayAccessor(const ArrayAccessor& other);

    void set(const ArrayAccessor& other);

    void reset(const Size3& size, T* const data);
    void reset(size_t width, size_t height, size_t depth, T* const data);

    T& at(size_t i);
    const T& at(size_t i) const;
    T& at(const Point3UI& pt);
    const T& at(const Point3UI& pt) const;
    T& at(size_t i, size_t j, size_t k);
    const T& at(size_t i, size_t j, size_t k) const;

    Size3 size() const;
    size_t width() const;
    size_t height() const;
    size_t depth() const;

    T* const data() const;
    void setData(const T* data);

    void swap(ArrayAccessor& other);

    size_t index(const Point3UI& pt) const;
    size_t index(size_t i, size_t j, size_t k) const;

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    T& operator()(const Point3UI& pt);
    const T& operator()(const Point3UI& pt) const;
    T& operator()(size_t i, size_t j, size_t k);
    const T& operator()(size_t i, size_t j, size_t k) const;

    ArrayAccessor& operator=(const ArrayAccessor& other);

 private:
    Size3 _size;
    T* _data;
};

template <typename T> using ArrayAccessor3 = ArrayAccessor<T, 3>;


template <typename T>
class ConstArrayAccessor<T, 3> {
 public:
    ConstArrayAccessor();
    explicit ConstArrayAccessor(const Size3& size, const T* const data);
    explicit ConstArrayAccessor(
        size_t width, size_t height, size_t depth, const T* const data);
    ConstArrayAccessor(const ArrayAccessor<T, 3>& other);
    ConstArrayAccessor(const ConstArrayAccessor& other);

    const T& at(size_t i) const;
    const T& at(const Point3UI& pt) const;
    const T& at(size_t i, size_t j, size_t k) const;

    Size3 size() const;
    size_t width() const;
    size_t height() const;
    size_t depth() const;

    const T* const data() const;

    size_t index(const Point3UI& pt) const;
    size_t index(size_t i, size_t j, size_t k) const;

    const T& operator[](size_t i) const;

    const T& operator()(const Point3UI& pt) const;
    const T& operator()(size_t i, size_t j, size_t k) const;

 private:
    Size3 _size;
    const T* _data;
};

template <typename T> using ConstArrayAccessor3 = ConstArrayAccessor<T, 3>;

}  // namespace jet

#include "detail/array_accessor3-inl.h"

#endif  // INCLUDE_JET_ARRAY_ACCESSOR3_H_
