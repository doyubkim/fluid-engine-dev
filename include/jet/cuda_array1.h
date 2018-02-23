// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY1_H_
#define INCLUDE_JET_CUDA_ARRAY1_H_

#include <jet/array_view1.h>
#include <jet/macros.h>

#include <thrust/device_vector.h>

#include <vector>

namespace jet {

namespace experimental {

template <typename T>
class CudaArrayView1;

template <typename T>
class CudaArray1 final {
 public:
    typedef thrust::device_vector<T> ContainerType;
    typedef thrust::device_ptr<T> Iterator;

    CudaArray1();

    CudaArray1(size_t size, const T& initVal = T());

    CudaArray1(const ArrayView1<T>& view);

    CudaArray1(const CudaArrayView1<T>& view);

    CudaArray1(const std::initializer_list<T>& lst);

    template <typename Alloc>
    CudaArray1(const thrust::host_vector<T, Alloc>& vec);

    template <typename Alloc>
    CudaArray1(const std::vector<T, Alloc>& vec);

    template <typename Alloc>
    CudaArray1(const thrust::device_vector<T, Alloc>& vec);

    CudaArray1(const CudaArray1& other);

    CudaArray1(CudaArray1&& other);

    void set(const T& value);

    void set(const ArrayView1<T>& view);

    void set(const CudaArrayView1<T>& view);

    void set(const std::initializer_list<T>& lst);

    template <typename Alloc>
    void set(const std::vector<T, Alloc>& vec);

    template <typename Alloc>
    void set(const thrust::host_vector<T, Alloc>& vec);

    template <typename Alloc>
    void set(const thrust::device_vector<T, Alloc>& vec);

    void set(const CudaArray1& other);

    void clear();

    void resize(size_t size, const T& initVal = T());

    void swap(CudaArray1& other);

    size_t size() const;

    T* data();

    const T* data() const;

    Iterator begin();

    Iterator begin() const;

    Iterator end();

    Iterator end() const;

    CudaArrayView1<T> view();

    const CudaArrayView1<T> view() const;

    //! Returns the reference to i-th element.
    typename CudaArray1<T>::ContainerType::reference operator[](size_t i);

    //! Returns the const reference to i-th element.
    const T& operator[](size_t i) const;

    CudaArray1& operator=(const T& value);

    CudaArray1& operator=(const ArrayView1<T>& view);

    CudaArray1& operator=(const CudaArrayView1<T>& view);

    CudaArray1& operator=(const std::initializer_list<T>& lst);

    template <typename Alloc>
    JET_CUDA_HOST CudaArray1& operator=(const std::vector<T, Alloc>& vec);

    template <typename Alloc>
    JET_CUDA_HOST CudaArray1& operator=(
        const thrust::host_vector<T, Alloc>& vec);

    template <typename Alloc>
    CudaArray1& operator=(const thrust::device_vector<T, Alloc>& vec);

    CudaArray1& operator=(const CudaArray1& other);

    CudaArray1& operator=(CudaArray1&& other);

 private:
    ContainerType _data;
};

}  // namespace experimental

}  // namespace jet

#include "detail/cuda_array1-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY1_H_

#endif  // JET_USE_CUDA
