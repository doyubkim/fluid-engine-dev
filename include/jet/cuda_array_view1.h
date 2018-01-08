// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA

#ifndef INCLUDE_JET_CUDA_ARRAY_VIEW1_H_
#define INCLUDE_JET_CUDA_ARRAY_VIEW1_H_

#include <jet/cuda_array1.h>

namespace jet {

namespace experimental {

template <typename T>
class CudaArrayView1 final {
 public:
    typedef thrust::device_ptr<T> Iterator;

    CudaArrayView1();

    explicit CudaArrayView1(T* data, size_t size);

    CudaArrayView1(const CudaArray1<T>& array);

    CudaArrayView1(const thrust::device_vector<T>& vec);

    CudaArrayView1(const CudaArrayView1& other);

    CudaArrayView1(CudaArrayView1&& other);

    void set(T* data, size_t size);

    void set(const CudaArray1<T>& array);

    void set(const thrust::device_vector<T>& vec);

    void set(const CudaArrayView1& other);

    void swap(CudaArrayView1& other);

    size_t size() const;

    T* data();

    const T* data() const;

    Iterator begin();

    Iterator begin() const;

    Iterator end();

    Iterator end() const;

    //! Returns the reference to i-th element.
    typename thrust::device_ptr<T>::reference operator[](size_t i);

    //! Returns the const reference to i-th element.
    const T& operator[](size_t i) const;

    CudaArrayView1& operator=(const CudaArray1<T>& array);

    CudaArrayView1& operator=(const thrust::host_vector<T>& vec);

    CudaArrayView1& operator=(const thrust::device_vector<T>& vec);

    CudaArrayView1& operator=(const CudaArrayView1& other);

    CudaArrayView1& operator=(CudaArrayView1&& other);

 private:
    thrust::device_ptr<T> _data;
    size_t _size = 0;
};

}  // namespace experimental

}  // namespace jet

#include "detail/cuda_array_view1-inl.h"

#endif  // INCLUDE_JET_CUDA_ARRAY_VIEW1_H_

#endif  // JET_USE_CUDA
