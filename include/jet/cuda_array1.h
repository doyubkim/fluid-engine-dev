// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY1_H_
#define INCLUDE_JET_CUDA_ARRAY1_H_

#include <thrust/device_vector.h>

namespace jet {

namespace experimental {

template <typename T>
class CudaArrayView1;

template <typename T>
class CudaArray1 final {
 public:
    typedef thrust::device_vector<T> ContainerType;

    CudaArray1();

    explicit CudaArray1(size_t size, const T& initVal = T());

    explicit CudaArray1(const CudaArrayView1<T>& view);

    explicit CudaArray1(const thrust::device_vector<T>& vec);

    CudaArray1(const CudaArray1& other);

    void set(const T& value);

    void set(const CudaArrayView1<T>& view);

    void set(const thrust::device_vector<T>& vec);

    void set(const CudaArray1& other);

    void clear();

    void resize(size_t size, const T& initVal);

    size_t size() const;

    T* data();

    const T* data() const;

    CudaArrayView1<T> view();

    const CudaArrayView1<T> view() const;

    //! Returns the reference to i-th element.
    typename CudaArray1<T>::ContainerType::reference operator[](size_t i);

    //! Returns the const reference to i-th element.
    const T& operator[](size_t i) const;

 private:
    ContainerType _data;
};

}  // namespace experimental

}  // namespace jet

#include "detail/cuda_array1-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY1_H_

#endif  // JET_USE_CUDA
