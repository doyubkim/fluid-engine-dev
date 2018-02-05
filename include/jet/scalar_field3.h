// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_SCALAR_FIELD3_H_
#define INCLUDE_JET_SCALAR_FIELD3_H_

#include <jet/field3.h>
#include <jet/vector3.h>
#include <functional>
#include <memory>

namespace jet {

//! Abstract base class for 3-D scalar field.
class ScalarField3 : public Field3 {
 public:
    //! Default constructor.
    ScalarField3();

    //! Default destructor.
    virtual ~ScalarField3();

    //! Returns sampled value at given position \p x.
    virtual double sample(const Vector3D& x) const = 0;

    //! Returns gradient vector at given position \p x.
    virtual Vector3D gradient(const Vector3D& x) const;

    //! Returns Laplacian at given position \p x.
    virtual double laplacian(const Vector3D& x) const;

    //! Returns sampler function object.
    virtual std::function<double(const Vector3D&)> sampler() const;
};

//! Shared pointer for the ScalarField3 type.
typedef std::shared_ptr<ScalarField3> ScalarField3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_FIELD3_H_
