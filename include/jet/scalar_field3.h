// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_SCALAR_FIELD3_H_
#define INCLUDE_JET_SCALAR_FIELD3_H_

#include <jet/field3.h>
#include <jet/vector3.h>
#include <functional>
#include <memory>

namespace jet {

class ScalarField3 : public Field3 {
 public:
    ScalarField3();

    virtual ~ScalarField3();

    virtual double sample(const Vector3D& x) const = 0;

    virtual Vector3D gradient(const Vector3D& x) const;

    virtual double laplacian(const Vector3D& x) const;

    virtual std::function<double(const Vector3D&)> sampler() const;
};

typedef std::shared_ptr<ScalarField3> ScalarField3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_SCALAR_FIELD3_H_
