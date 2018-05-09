// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY1_H_
#define INCLUDE_JET_CUDA_ARRAY1_H_

#include <jet/array_view1.h>
#include <jet/cuda_array.h>
#include <jet/macros.h>

#include <thrust/device_vector.h>

#include <vector>

namespace jet {

template <typename T, size_t N>
class CudaArrayView;

template <typename T, size_t N>
class ConstCudaArrayView;

template <typename T>
class CudaArray<T, 1> final {
 public:
    typedef T value_type;
    typedef thrust::device_vector<T> ContainerType;
    typedef typename ContainerType::reference reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef thrust::device_ptr<T> iterator;
    typedef iterator const_iterator;

    CudaArray();

    CudaArray(size_t size, const T& initVal = T());

    CudaArray(const ConstArrayView<T, 1>& view);

    CudaArray(const ConstCudaArrayView<T, 1>& view);

    CudaArray(const std::initializer_list<T>& lst);

    template <typename Alloc>
    CudaArray(const thrust::host_vector<T, Alloc>& vec);

    template <typename Alloc>
    CudaArray(const std::vector<T, Alloc>& vec);

    template <typename Alloc>
    CudaArray(const thrust::device_vector<T, Alloc>& vec);

    CudaArray(const CudaArray& other);

    CudaArray(CudaArray&& other);

    void set(const T& value);

    void set(const ConstArrayView<T, 1>& view);

    void set(const ConstCudaArrayView<T, 1>& view);

    void set(const std::initializer_list<T>& lst);

    template <typename Alloc>
    void set(const std::vector<T, Alloc>& vec);

    template <typename Alloc>
    void set(const thrust::host_vector<T, Alloc>& vec);

    template <typename Alloc>
    void set(const thrust::device_vector<T, Alloc>& vec);

    void set(const CudaArray& other);

    void clear();

    void resize(size_t size, const T& initVal = T());

    void swap(CudaArray& other);

    size_t size() const;

    pointer data();

    const_pointer data() const;

    iterator begin();

    const_iterator begin() const;

    iterator end();

    const_iterator end() const;

    CudaArrayView<T, 1> view();

    ConstCudaArrayView<T, 1> view() const;

    //! Returns the reference to i-th element.
    reference operator[](size_t i);

    //! Returns the value of the i-th element.
    T operator[](size_t i) const;

    CudaArray& operator=(const T& value);

    CudaArray& operator=(const ConstArrayView<T, 1>& view);

    CudaArray& operator=(const ConstCudaArrayView<T, 1>& view);

    CudaArray& operator=(const std::initializer_list<T>& lst);

    template <typename Alloc>
    JET_CUDA_HOST CudaArray& operator=(const std::vector<T, Alloc>& vec);

    template <typename Alloc>
    JET_CUDA_HOST CudaArray& operator=(
        const thrust::host_vector<T, Alloc>& vec);

    template <typename Alloc>
    CudaArray& operator=(const thrust::device_vector<T, Alloc>& vec);

    CudaArray& operator=(const CudaArray& other);

    CudaArray& operator=(CudaArray&& other);

 private:
    ContainerType _data;
};

//! Type alias for 1-D CUDA array.
template <typename T>
using CudaArray1 = CudaArray<T, 1>;

}  // namespace jet

#include "detail/cuda_array1-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY1_H_

#endif  // JET_USE_CUDA
