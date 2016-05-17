// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CONSTANT_SCALAR_FIELD3_H_
#define INCLUDE_JET_CONSTANT_SCALAR_FIELD3_H_

#include <jet/scalar_field3.h>

namespace jet {

class ConstantScalarField3 final : public ScalarField3 {
 public:
    explicit ConstantScalarField3(double value);

    double sample(const Vector3D& x) const override;

    std::function<double(const Vector3D&)> sampler() const override;

 private:
    double _value = 0.0;
};

}  // namespace jet

#endif  // INCLUDE_JET_CONSTANT_SCALAR_FIELD3_H_
