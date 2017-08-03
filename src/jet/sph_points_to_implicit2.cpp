// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/sph_points_to_implicit2.h>
#include <jet/sph_system_data2.h>

using namespace jet;

SphPointsToImplicit2::SphPointsToImplicit2(double kernelRadius)
    : _kernelRadius(kernelRadius) {}

void SphPointsToImplicit2::convert(const ConstArrayAccessor1<Vector2D>& points,
                                   ScalarGrid2* output) const {
    if (output == nullptr) {
        JET_WARN << "Null scalar grid output pointer provided.";
        return;
    }

    const auto res = output->resolution();
    if (res.x * res.y == 0) {
        JET_WARN << "Empty grid is provided.";
        return;
    }

    const auto bbox = output->boundingBox();
    if (bbox.isEmpty()) {
        JET_WARN << "Empty domain is provided.";
        return;
    }

    SphSystemData2 sphParticles;
    sphParticles.addParticles(points);
    sphParticles.setKernelRadius(_kernelRadius);
    sphParticles.buildNeighborSearcher();
    sphParticles.updateDensities();

    Array1<double> constData(sphParticles.numberOfParticles(), 1.0);
    output->fill([&](const Vector2D& pt) {
        double d = sphParticles.interpolate(pt, constData);
        return 0.5 - d;
    });
}
