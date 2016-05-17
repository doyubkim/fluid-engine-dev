// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_
#define INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_

#include <jet/vector_field3.h>

namespace jet {

class CustomVectorField3 final : public VectorField3 {
 public:
    CustomVectorField3(
        const std::function<Vector3D(const Vector3D&)>& customFunction,
        double derivativeResolution = 1e-3);

    CustomVectorField3(
        const std::function<Vector3D(const Vector3D&)>& customFunction,
        const std::function<double(const Vector3D&)>& customDivergenceFunction,
        double derivativeResolution = 1e-3);

    CustomVectorField3(
        const std::function<Vector3D(const Vector3D&)>& customFunction,
        const std::function<double(const Vector3D&)>& customDivergenceFunction,
        const std::function<Vector3D(const Vector3D&)>& customCurlFunction);

    virtual ~CustomVectorField3();

    Vector3D sample(const Vector3D& x) const override;

    double divergence(const Vector3D& x) const override;

    Vector3D curl(const Vector3D& x) const override;

    std::function<Vector3D(const Vector3D&)> sampler() const override;

 private:
    std::function<Vector3D(const Vector3D&)> _customFunction;
    std::function<double(const Vector3D&)> _customDivergenceFunction;
    std::function<Vector3D(const Vector3D&)> _customCurlFunction;
    double _resolution = 1e-3;
};

}  // namespace jet

#endif  // INCLUDE_JET_CUSTOM_VECTOR_FIELD3_H_
