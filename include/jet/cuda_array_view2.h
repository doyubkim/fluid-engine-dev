// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY_VIEW2_H_
#define INCLUDE_JET_CUDA_ARRAY_VIEW2_H_

#include <jet/cuda_array2.h>
#include <jet/cuda_array_view.h>

namespace jet {

template <typename T>
class CudaArrayView<T, 2> final {
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

    CudaArrayView(pointer data, const Size2& size);

    CudaArrayView(const thrust::host_vector<T>& vec, const Size2& size);

    CudaArrayView(const thrust::device_vector<T>& vec, const Size2& size);

    CudaArrayView(const CudaArray<T, 1>& array, const Size2& size);

    CudaArrayView(const CudaArray<T, 2>& array);

    CudaArrayView(const CudaArrayView<T, 1>& other, const Size2& size);

    CudaArrayView(const CudaArrayView& other);

    CudaArrayView(CudaArrayView&& other);

    void set(pointer data, const Size2& size);

    void set(const thrust::host_vector<T>& vec, const Size2& size);

    void set(const thrust::device_vector<T>& vec, const Size2& size);

    void set(const CudaArray<T, 1>& array, const Size2& size);

    void set(const CudaArray<T, 2>& array);

    void set(const CudaArrayView<T, 1>& other, const Size2& size);

    void set(const CudaArrayView& other);

    void swap(CudaArrayView& other);

    const Size2& size() const;

    size_t width() const;

    size_t height() const;

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

    reference operator()(size_t i, size_t j);

    value_type operator()(size_t i, size_t j) const;

    CudaArrayView& operator=(const CudaArray<T, 2>& array);

    CudaArrayView& operator=(const CudaArrayView& other);

    CudaArrayView& operator=(CudaArrayView&& other);

 private:
    thrust::device_ptr<T> _data;
    size_t _size = 0;
};

//! Type alias for 2-D CUDA array view.
template <typename T>
using CudaArrayView2 = CudaArrayView<T, 2>;

template <typename T>
class ConstCudaArrayView<T, 2> final {
 public:
    typedef T value_type;
    typedef typename thrust::device_ptr<T>::reference reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef thrust::device_ptr<T> iterator;
    typedef iterator const_iterator;

    ConstCudaArrayView();

    ConstCudaArrayView(const_pointer data, const Size2& size);

    ConstCudaArrayView(const thrust::host_vector<T>& vec, const Size2& size);

    ConstCudaArrayView(const thrust::device_vector<T>& vec, const Size2& size);

    ConstCudaArrayView(const CudaArray<T, 1>& array, const Size2& size);

    ConstCudaArrayView(const CudaArray<T, 2>& array);

    ConstCudaArrayView(const CudaArrayView<T, 1>& other, const Size2& size);

    ConstCudaArrayView(const CudaArrayView<T, 2>& other);

    ConstCudaArrayView(const ConstCudaArrayView<T, 1>& other,
                       const Size2& size);

    ConstCudaArrayView(const ConstCudaArrayView& other);

    ConstCudaArrayView(ConstCudaArrayView&& other);

    const Size2& size() const;

    size_t width() const;

    size_t height() const;

    const_pointer data() const;

    const_iterator begin() const;

    const_iterator end() const;

    //! Returns the value of the i-th element.
    value_type operator[](size_t i) const;

    ConstCudaArrayView& operator=(const CudaArray<T, 2>& array);

    ConstCudaArrayView& operator=(const CudaArrayView<T, 2>& other);

    ConstCudaArrayView& operator=(const ConstCudaArrayView& other);

    ConstCudaArrayView& operator=(ConstCudaArrayView&& other);

 private:
    thrust::device_ptr<T> _data;
    size_t _size = 0;

    void set(const_pointer data, const Size2& size);

    void set(const thrust::host_vector<T>& vec, const Size2& size);

    void set(const thrust::device_vector<T>& vec, const Size2& size);

    void set(const CudaArray<T, 1>& array, const Size2& size);

    void set(const CudaArray<T, 2>& array);

    void set(const CudaArrayView<T, 1>& other, const Size2& size);

    void set(const CudaArrayView<T, 2>& other);

    void set(const ConstCudaArrayView<T, 1>& other, const Size2& size);

    void set(const ConstCudaArrayView& other);
};

//! Type alias for 2-D const CUDA array view.
template <typename T>
using ConstCudaArrayView2 = ConstCudaArrayView<T, 2>;

}  // namespace jet

#include "detail/cuda_array_view2-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY_VIEW2_H_

#endif  // JET_USE_CUDA
