// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/fmm_level_set_solver2.h>
#include <jet/particle_system_data2.h>
#include <jet/zhu_bridson_points_to_implicit2.h>

using namespace jet;

inline double k(double s) { return std::max(0.0, cubic(1.0 - s * s)); }

ZhuBridsonPointsToImplicit2::ZhuBridsonPointsToImplicit2(double kernelRadius,
                                                         double cutOffThreshold,
                                                         bool isOutputSdf)
    : _kernelRadius(kernelRadius),
      _cutOffThreshold(cutOffThreshold),
      _isOutputSdf(isOutputSdf) {}

void ZhuBridsonPointsToImplicit2::convert(
    const ConstArrayAccessor1<Vector2D>& points, ScalarGrid2* output) const {
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

    ParticleSystemData2 particles;
    particles.addParticles(points);
    particles.buildNeighborSearcher(_kernelRadius);

    const auto neighborSearcher = particles.neighborSearcher();
    const double isoContValue = _cutOffThreshold * _kernelRadius;

    auto temp = output->clone();
    temp->fill([&](const Vector2D& x) -> double {
        Vector2D xAvg;
        double wSum = 0.0;
        const auto func = [&](size_t, const Vector2D& xi) {
            const double wi = k((x - xi).length() / _kernelRadius);

            wSum += wi;
            xAvg += wi * xi;
        };
        neighborSearcher->forEachNearbyPoint(x, _kernelRadius, func);

        if (wSum > 0.0) {
            xAvg /= wSum;
            return (x - xAvg).length() - isoContValue;
        } else {
            return output->boundingBox().diagonalLength();
        }
    });

    if (_isOutputSdf) {
        FmmLevelSetSolver2 solver;
        solver.reinitialize(*temp, kMaxD, output);
    } else {
        temp->swap(output);
    }
}
