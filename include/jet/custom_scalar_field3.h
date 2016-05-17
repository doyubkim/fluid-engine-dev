// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_SCALAR_FIELD3_H_
#define INCLUDE_JET_CUSTOM_SCALAR_FIELD3_H_

#include <jet/scalar_field3.h>

namespace jet {

class CustomScalarField3 final : public ScalarField3 {
 public:
    CustomScalarField3(
        const std::function<double(const Vector3D&)>& customFunction,
        double derivativeResolution = 1e-3);

    CustomScalarField3(
        const std::function<double(const Vector3D&)>& customFunction,
        const std::function<Vector3D(const Vector3D&)>& customGradientFunction,
        double derivativeResolution = 1e-3);

    CustomScalarField3(
        const std::function<double(const Vector3D&)>& customFunction,
        const std::function<Vector3D(const Vector3D&)>& customGradientFunction,
        const std::function<double(const Vector3D&)>& customLaplacianFunction);

    double sample(const Vector3D& x) const override;

    std::function<double(const Vector3D&)> sampler() const override;

    Vector3D gradient(const Vector3D& x) const override;

    double laplacian(const Vector3D& x) const override;

 private:
    std::function<double(const Vector3D&)> _customFunction;
    std::function<Vector3D(const Vector3D&)> _customGradientFunction;
    std::function<double(const Vector3D&)> _customLaplacianFunction;
    double _resolution = 1e-3;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_SCALAR_FIELD3_H_
