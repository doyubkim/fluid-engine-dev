// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VECTOR_FIELD_H_
#define INCLUDE_JET_VECTOR_FIELD_H_

#include <jet/field.h>
#include <jet/matrix.h>

#include <functional>
#include <memory>

namespace jet {

template <size_t N>
struct GetCurl {};

template <>
struct GetCurl<2> {
    using type = double;
};

template <>
struct GetCurl<3> {
    using type = Vector3D;
};

//! Abstract base class for N-D vector field.
template <size_t N>
class VectorField : public Field<N> {
 public:
    //! Default constructor.
    VectorField();

    //! Default destructor.
    virtual ~VectorField();

    //! Returns sampled value at given position \p x.
    virtual Vector<double, N> sample(const Vector<double, N>& x) const = 0;

    //! Returns divergence at given position \p x.
    virtual double divergence(const Vector<double, N>& x) const;

    //! Returns curl at given position \p x.
    virtual typename GetCurl<N>::type curl(const Vector<double, N>& x) const;

    //! Returns sampler function object.
    virtual std::function<Vector<double, N>(const Vector<double, N>&)> sampler()
        const;
};

//! 2-D VectorField type.
using VectorField2 = VectorField<2>;

//! 3-D VectorField type.
using VectorField3 = VectorField<3>;

//! Shared pointer for the VectorField2 type.
using VectorField2Ptr = std::shared_ptr<VectorField2>;

//! Shared pointer for the VectorField3 type.
using VectorField3Ptr = std::shared_ptr<VectorField3>;

}  // namespace jet

#endif  // INCLUDE_JET_VECTOR_FIELD_H_
