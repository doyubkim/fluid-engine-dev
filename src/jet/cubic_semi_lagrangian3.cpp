// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/array_samplers3.h>
#include <jet/cubic_semi_lagrangian3.h>

using namespace jet;

CubicSemiLagrangian3::CubicSemiLagrangian3() {
}

std::function<double(const Vector3D&)>
CubicSemiLagrangian3::getScalarSamplerFunc(const ScalarGrid3& source) const {
    auto sourceSampler = CubicArraySampler3<double, double>(
        source.constDataAccessor(),
        source.gridSpacing(),
        source.dataOrigin());
    return sourceSampler.functor();
}

std::function<Vector3D(const Vector3D&)>
CubicSemiLagrangian3::getVectorSamplerFunc(
    const CollocatedVectorGrid3& source) const {
    auto sourceSampler = CubicArraySampler3<Vector3D, double>(
        source.constDataAccessor(),
        source.gridSpacing(),
        source.dataOrigin());
    return sourceSampler.functor();
}

std::function<Vector3D(const Vector3D&)>
CubicSemiLagrangian3::getVectorSamplerFunc(
    const FaceCenteredGrid3& source) const {
    auto uSourceSampler = CubicArraySampler3<double, double>(
        source.uConstAccessor(),
        source.gridSpacing(),
        source.uOrigin());
    auto vSourceSampler = CubicArraySampler3<double, double>(
        source.vConstAccessor(),
        source.gridSpacing(),
        source.vOrigin());
    auto wSourceSampler = CubicArraySampler3<double, double>(
        source.wConstAccessor(),
        source.gridSpacing(),
        source.wOrigin());
    return
        [uSourceSampler, vSourceSampler, wSourceSampler](const Vector3D& x) {
            return Vector3D(
                uSourceSampler(x), vSourceSampler(x), wSourceSampler(x));
        };
}
