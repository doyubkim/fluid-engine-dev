// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/array_samplers2.h>
#include <jet/cubic_semi_lagrangian2.h>

using namespace jet;

CubicSemiLagrangian2::CubicSemiLagrangian2() {
}

std::function<double(const Vector2D&)>
CubicSemiLagrangian2::getScalarSamplerFunc(const ScalarGrid2& source) const {
    auto sourceSampler = CubicArraySampler2<double, double>(
        source.constDataAccessor(),
        source.gridSpacing(),
        source.dataOrigin());
    return sourceSampler.functor();
}

std::function<Vector2D(const Vector2D&)>
CubicSemiLagrangian2::getVectorSamplerFunc(
    const CollocatedVectorGrid2& source) const {
    auto sourceSampler = CubicArraySampler2<Vector2D, double>(
        source.constDataAccessor(),
        source.gridSpacing(),
        source.dataOrigin());
    return sourceSampler.functor();
}

std::function<Vector2D(const Vector2D&)>
CubicSemiLagrangian2::getVectorSamplerFunc(
    const FaceCenteredGrid2& source) const {
    auto uSourceSampler = CubicArraySampler2<double, double>(
        source.uConstAccessor(),
        source.gridSpacing(),
        source.uOrigin());
    auto vSourceSampler = CubicArraySampler2<double, double>(
        source.vConstAccessor(),
        source.gridSpacing(),
        source.vOrigin());
    return
        [uSourceSampler, vSourceSampler](const Vector2D& x) {
            return Vector2D(
                uSourceSampler(x), vSourceSampler(x));
        };
}
