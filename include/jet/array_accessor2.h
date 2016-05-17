// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ARRAY_ACCESSOR2_H_
#define INCLUDE_JET_ARRAY_ACCESSOR2_H_

#include <jet/array_accessor.h>
#include <jet/point2.h>
#include <jet/size2.h>

namespace jet {

template <typename T>
class ArrayAccessor<T, 2> final {
 public:
    ArrayAccessor();
    explicit ArrayAccessor(const Size2& size, T* const data);
    explicit ArrayAccessor(size_t width, size_t height, T* const data);
    ArrayAccessor(const ArrayAccessor& other);

    void set(const ArrayAccessor& other);

    void reset(const Size2& size, T* const data);
    void reset(size_t width, size_t height, T* const data);

    T& at(size_t i);
    const T& at(size_t i) const;
    T& at(const Point2UI& pt);
    const T& at(const Point2UI& pt) const;
    T& at(size_t i, size_t j);
    const T& at(size_t i, size_t j) const;

    Size2 size() const;
    size_t width() const;
    size_t height() const;

    T* const data() const;
    void setData(T* const data);

    void swap(ArrayAccessor& other);

    size_t index(const Point2UI& pt) const;
    size_t index(size_t i, size_t j) const;

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    T& operator()(const Point2UI& pt);
    const T& operator()(const Point2UI& pt) const;
    T& operator()(size_t i, size_t j);
    const T& operator()(size_t i, size_t j) const;

    ArrayAccessor& operator=(const ArrayAccessor& other);

 private:
    Size2 _size;
    T* _data;
};

template <typename T> using ArrayAccessor2 = ArrayAccessor<T, 2>;


template <typename T>
class ConstArrayAccessor<T, 2> {
 public:
    ConstArrayAccessor();
    explicit ConstArrayAccessor(const Size2& size, const T* const data);
    explicit ConstArrayAccessor(
        size_t width, size_t height, const T* const data);
    ConstArrayAccessor(const ArrayAccessor<T, 2>& other);
    ConstArrayAccessor(const ConstArrayAccessor& other);


    const T& at(size_t i) const;
    const T& at(const Point2UI& pt) const;
    const T& at(size_t i, size_t j) const;

    Size2 size() const;
    size_t width() const;
    size_t height() const;

    const T* const data() const;

    size_t index(const Point2UI& pt) const;
    size_t index(size_t i, size_t j) const;

    const T& operator[](size_t i) const;

    const T& operator()(const Point2UI& pt) const;
    const T& operator()(size_t i, size_t j) const;

 private:
    Size2 _size;
    const T* _data;
};

template <typename T> using ConstArrayAccessor2 = ConstArrayAccessor<T, 2>;

}  // namespace jet

#include "detail/array_accessor2-inl.h"

#endif  // INCLUDE_JET_ARRAY_ACCESSOR2_H_
