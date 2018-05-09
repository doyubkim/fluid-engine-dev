// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY_VIEW1_H_
#define INCLUDE_JET_CUDA_ARRAY_VIEW1_H_

#include <jet/cuda_array1.h>
#include <jet/cuda_array_view.h>

namespace jet {

template <typename T>
class CudaArrayView<T, 1> final {
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

    explicit CudaArrayView(pointer data, size_t size);

    CudaArrayView(const thrust::device_vector<T>& vec);

    CudaArrayView(const CudaArray<T, 1>& array);

    CudaArrayView(const CudaArrayView& other);

    CudaArrayView(CudaArrayView&& other);

    void set(pointer data, size_t size);

    void set(const thrust::host_vector<T>& vec);

    void set(const thrust::device_vector<T>& vec);

    void set(const CudaArray<T, 1>& array);

    void set(const CudaArrayView& other);

    void swap(CudaArrayView& other);

    size_t size() const;

    pointer data();

    const_pointer data() const;

    iterator begin();

    const_iterator begin() const;

    iterator end();

    const_iterator end() const;

    //! Returns the reference to i-th element.
    reference operator[](size_t i);

    //! Returns the const reference to i-th element.
    value_type operator[](size_t i) const;

    CudaArrayView& operator=(const thrust::device_vector<T>& vec);

    CudaArrayView& operator=(const CudaArray<T, 1>& array);

    CudaArrayView& operator=(const CudaArrayView& other);

    CudaArrayView& operator=(CudaArrayView&& other);

 private:
    thrust::device_ptr<T> _data;
    size_t _size = 0;
};

//! Type alias for 1-D CUDA array view.
template <typename T>
using CudaArrayView1 = CudaArrayView<T, 1>;

template <typename T>
class ConstCudaArrayView<T, 1> final {
 public:
    typedef T value_type;
    typedef typename thrust::device_ptr<T>::reference reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef thrust::device_ptr<T> iterator;
    typedef iterator const_iterator;

    ConstCudaArrayView();

    explicit ConstCudaArrayView(const_pointer data, size_t size);

    ConstCudaArrayView(const thrust::device_vector<T>& vec);

    ConstCudaArrayView(const CudaArray<T, 1>& array);

    ConstCudaArrayView(const CudaArrayView<T, 1>& view);

    ConstCudaArrayView(const ConstCudaArrayView& other);

    size_t size() const;

    const_pointer data() const;

    const_iterator begin() const;

    const_iterator end() const;

    //! Returns the const reference to i-th element.
    value_type operator[](size_t i) const;

    ConstCudaArrayView& operator=(const thrust::device_vector<T>& vec);

    ConstCudaArrayView& operator=(const CudaArray<T, 1>& array);

    ConstCudaArrayView& operator=(const CudaArrayView<T, 1>& view);

    ConstCudaArrayView& operator=(const ConstCudaArrayView& other);

private:
    thrust::device_ptr<T> _data;
    size_t _size = 0;

    void set(const_pointer data, size_t size);

    void set(const thrust::device_vector<T>& vec);

    void set(const CudaArray<T, 1>& array);

    void set(const CudaArrayView<T, 1>& view);

    void set(const ConstCudaArrayView& other);
};

//! Type alias for 1-D const CUDA array view.
template <typename T>
using ConstCudaArrayView1 = ConstCudaArrayView<T, 1>;

}  // namespace jet

#include "detail/cuda_array_view1-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY_VIEW1_H_

#endif  // JET_USE_CUDA
