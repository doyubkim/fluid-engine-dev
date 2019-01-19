// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SCALAR_FIELD_H_
#define INCLUDE_JET_SCALAR_FIELD_H_

#include <jet/field.h>
#include <jet/matrix.h>

#include <functional>
#include <memory>

namespace jet {

//! Abstract base class for N-D scalar field.
template <size_t N>
class ScalarField : public Field<N> {
 public:
    //! Default constructor.
    ScalarField();

    //! Default destructor.
    virtual ~ScalarField();

    //! Returns sampled value at given position \p x.
    virtual double sample(const Vector<double, N>& x) const = 0;

    //! Returns gradient vector at given position \p x.
    virtual Vector<double, N> gradient(const Vector<double, N>& x) const;

    //! Returns Laplacian at given position \p x.
    virtual double laplacian(const Vector<double, N>& x) const;

    //! Returns sampler function object.
    virtual std::function<double(const Vector<double, N>&)> sampler() const;
};

//! 2-D ScalarField type.
using ScalarField2 = ScalarField<2>;

//! 3-D ScalarField type.
using ScalarField3 = ScalarField<3>;

//! N-D shared pointer for the ScalarField type.
template <size_t N>
using ScalarFieldPtr = std::shared_ptr<ScalarField<N>>;

//! Shared pointer for the ScalarField2 type.
using ScalarField2Ptr = std::shared_ptr<ScalarField2>;

//! Shared pointer for the ScalarField3 type.
using ScalarField3Ptr = std::shared_ptr<ScalarField3>;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_FIELD_H_
