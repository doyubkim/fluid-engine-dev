// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_
#define INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_

#include <jet/scalar_field2.h>

namespace jet {

class CustomScalarField2 final : public ScalarField2 {
 public:
    CustomScalarField2(
        const std::function<double(const Vector2D&)>& customFunction,
        double derivativeResolution = 1e-3);

    CustomScalarField2(
        const std::function<double(const Vector2D&)>& customFunction,
        const std::function<Vector2D(const Vector2D&)>& customGradientFunction,
        double derivativeResolution = 1e-3);

    CustomScalarField2(
        const std::function<double(const Vector2D&)>& customFunction,
        const std::function<Vector2D(const Vector2D&)>& customGradientFunction,
        const std::function<double(const Vector2D&)>& customLaplacianFunction);

    double sample(const Vector2D& x) const override;

    std::function<double(const Vector2D&)> sampler() const override;

    Vector2D gradient(const Vector2D& x) const override;

    double laplacian(const Vector2D& x) const override;

 private:
    std::function<double(const Vector2D&)> _customFunction;
    std::function<Vector2D(const Vector2D&)> _customGradientFunction;
    std::function<double(const Vector2D&)> _customLaplacianFunction;
    double _resolution = 1e-3;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_SCALAR_FIELD2_H_
