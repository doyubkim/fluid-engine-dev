// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array_samplers.h>
#include <jet/cubic_semi_lagrangian3.h>
#include <pch.h>

using namespace jet;

CubicSemiLagrangian3::CubicSemiLagrangian3() {}

std::function<double(const Vector3D&)>
CubicSemiLagrangian3::getScalarSamplerFunc(const ScalarGrid3& source) const {
    auto sourceSampler = MonotonicCatmullRomArraySampler3<double>(
        source.dataView(), source.gridSpacing(), source.dataOrigin());
    return sourceSampler.functor();
}

std::function<Vector3D(const Vector3D&)>
CubicSemiLagrangian3::getVectorSamplerFunc(
    const CollocatedVectorGrid3& source) const {
    auto sourceSampler = MonotonicCatmullRomArraySampler3<Vector3D>(
        source.dataView(), source.gridSpacing(), source.dataOrigin());
    return sourceSampler.functor();
}

std::function<Vector3D(const Vector3D&)>
CubicSemiLagrangian3::getVectorSamplerFunc(
    const FaceCenteredGrid3& source) const {
    auto uSourceSampler = MonotonicCatmullRomArraySampler3<double>(
        source.uView(), source.gridSpacing(), source.uOrigin());
    auto vSourceSampler = MonotonicCatmullRomArraySampler3<double>(
        source.vView(), source.gridSpacing(), source.vOrigin());
    auto wSourceSampler = MonotonicCatmullRomArraySampler3<double>(
        source.wView(), source.gridSpacing(), source.wOrigin());
    return [uSourceSampler, vSourceSampler, wSourceSampler](const Vector3D& x) {
        return Vector3D(uSourceSampler(x), vSourceSampler(x),
                        wSourceSampler(x));
    };
}
