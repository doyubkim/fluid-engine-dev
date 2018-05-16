// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY_VIEW3_H_
#define INCLUDE_JET_CUDA_ARRAY_VIEW3_H_

#include <jet/cuda_array3.h>
#include <jet/cuda_array_view.h>

namespace jet {

template <typename T>
class CudaArrayView<T, 3> final {
 public:
    typedef T value_type;
    typedef thrust::device_vector<T> ContainerType;
    typedef typename ContainerType::reference reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef thrust::device_ptr<T> iterator;
    typedef iterator const_iterator;

    CudaArrayView();

    CudaArrayView(pointer data, const Size3& size);

    CudaArrayView(const CudaArray<T, 3>& array);

    CudaArrayView(const CudaArrayView& other);

    CudaArrayView(CudaArrayView&& other);

    void set(pointer data, const Size3& size);

    void set(const CudaArray<T, 3>& array);

    void set(const CudaArrayView& other);

    void swap(CudaArrayView& other);

    const Size3& size() const;

    size_t width() const;

    size_t height() const;

    size_t depth() const;

    pointer data();

    const_pointer data() const;

    iterator begin();

    const_iterator begin() const;

    iterator end();

    const_iterator end() const;

    //! Returns the reference to i-th element.
    reference operator[](size_t i);

    //! Returns the value of the i-th element.
    value_type operator[](size_t i) const;

    reference operator()(size_t i, size_t j, size_t k);

    value_type operator()(size_t i, size_t j, size_t k) const;

    CudaArrayView& operator=(const CudaArray<T, 3>& array);

    CudaArrayView& operator=(const CudaArrayView& other);

    CudaArrayView& operator=(CudaArrayView&& other);

 private:
    thrust::device_ptr<T> _data;
    Size3 _size;
};

//! Type alias for 3-D CUDA array view.
template <typename T>
using CudaArrayView3 = CudaArrayView<T, 3>;

template <typename T>
class ConstCudaArrayView<T, 3> final {
 public:
    typedef T value_type;
    typedef typename thrust::device_ptr<T>::reference reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef thrust::device_ptr<T> iterator;
    typedef iterator const_iterator;

    ConstCudaArrayView();

    ConstCudaArrayView(const_pointer data, const Size3& size);

    ConstCudaArrayView(const CudaArray<T, 3>& array);

    ConstCudaArrayView(const CudaArrayView<T, 3>& other);

    ConstCudaArrayView(const ConstCudaArrayView& other);

    ConstCudaArrayView(ConstCudaArrayView&& other);

    const Size3& size() const;

    size_t width() const;

    size_t height() const;

    size_t depth() const;

    const_pointer data() const;

    const_iterator begin() const;

    const_iterator end() const;

    //! Returns the value of the i-th element.
    value_type operator[](size_t i) const;

    value_type operator()(size_t i, size_t j, size_t k) const;

    ConstCudaArrayView& operator=(const CudaArray<T, 3>& array);

    ConstCudaArrayView& operator=(const CudaArrayView<T, 3>& other);

    ConstCudaArrayView& operator=(const ConstCudaArrayView& other);

    ConstCudaArrayView& operator=(ConstCudaArrayView&& other);

 private:
    thrust::device_ptr<T> _data;
    Size3 _size;

    void set(const_pointer data, const Size3& size);

    void set(const CudaArray<T, 3>& array);

    void set(const CudaArrayView<T, 3>& other);

    void set(const ConstCudaArrayView& other);
};

//! Type alias for 3-D const CUDA array view.
template <typename T>
using ConstCudaArrayView3 = ConstCudaArrayView<T, 3>;

}  // namespace jet

#include "detail/cuda_array_view3-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY_VIEW3_H_

#endif  // JET_USE_CUDA
