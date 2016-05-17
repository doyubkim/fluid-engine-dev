// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_VECTOR_FIELD2_H_
#define INCLUDE_JET_CUSTOM_VECTOR_FIELD2_H_

#include <jet/vector_field2.h>

namespace jet {

class CustomVectorField2 final : public VectorField2 {
 public:
    CustomVectorField2(
        const std::function<Vector2D(const Vector2D&)>& customFunction,
        double derivativeResolution = 1e-3);

    CustomVectorField2(
        const std::function<Vector2D(const Vector2D&)>& customFunction,
        const std::function<double(const Vector2D&)>& customDivergenceFunction,
        double derivativeResolution = 1e-3);

    CustomVectorField2(
        const std::function<Vector2D(const Vector2D&)>& customFunction,
        const std::function<double(const Vector2D&)>& customDivergenceFunction,
        const std::function<double(const Vector2D&)>& customCurlFunction);

    virtual ~CustomVectorField2();

    Vector2D sample(const Vector2D& x) const override;

    double divergence(const Vector2D& x) const override;

    double curl(const Vector2D& x) const override;

    std::function<Vector2D(const Vector2D&)> sampler() const override;

 private:
    std::function<Vector2D(const Vector2D&)> _customFunction;
    std::function<double(const Vector2D&)> _customDivergenceFunction;
    std::function<double(const Vector2D&)> _customCurlFunction;
    double _resolution = 1e-3;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_VECTOR_FIELD2_H_
