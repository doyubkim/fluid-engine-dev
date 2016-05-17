// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_ARRAY_ACCESSOR1_H_
#define INCLUDE_JET_ARRAY_ACCESSOR1_H_

#include <jet/array_accessor.h>

namespace jet {

template <typename T>
class ArrayAccessor<T, 1> final {
 public:
    ArrayAccessor();
    explicit ArrayAccessor(size_t size, T* const data);
    ArrayAccessor(const ArrayAccessor& other);

    void set(const ArrayAccessor& other);

    void reset(size_t size, T* const data);

    T& at(size_t i);
    const T& at(size_t i) const;

    T* const begin() const;
    T* const end() const;

    T* begin();
    T* end();

    size_t size() const;

    T* const data() const;
    void setData(T* const data);

    void swap(ArrayAccessor& other);

    T& operator[](size_t i);
    const T& operator[](size_t i) const;

    ArrayAccessor& operator=(const ArrayAccessor& other);

 private:
    size_t _size;
    T* _data;
};

template <typename T> using ArrayAccessor1 = ArrayAccessor<T, 1>;


template <typename T>
class ConstArrayAccessor<T, 1> {
 public:
    ConstArrayAccessor();
    explicit ConstArrayAccessor(size_t size, const T* const data);
    ConstArrayAccessor(const ArrayAccessor<T, 1>& other);
    ConstArrayAccessor(const ConstArrayAccessor& other);

    const T& at(size_t i) const;

    const T* const begin() const;
    const T* const end() const;

    size_t size() const;

    const T* const data() const;

    const T& operator[](size_t i) const;

 private:
    size_t _size;
    const T* _data;
};

template <typename T> using ConstArrayAccessor1 = ConstArrayAccessor<T, 1>;

}  // namespace jet

#include "detail/array_accessor1-inl.h"

#endif  // INCLUDE_JET_ARRAY_ACCESSOR1_H_
