// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/anisotropic_points_to_implicit3.h>
#include <jet/fmm_level_set_solver3.h>
#include <jet/jet.h>
#include <jet/sph_kernels3.h>
#include <jet/sph_system_data3.h>
#include <jet/svd.h>

using namespace jet;

inline double p(double distance) {
    const double distanceSquared = distance * distance;

    if (distanceSquared >= 1.0) {
        return 0.0;
    } else {
        const double x = 1.0 - distanceSquared;
        return x * x * x;
    }
}

inline double wij(double distance, double r) {
    if (distance < r) {
        return 1.0 - cubic(distance / r);
    } else {
        return 0.0;
    }
}

inline Matrix3x3D vvt(const Vector3D& v) {
    return Matrix3x3D(v.x * v.x, v.x * v.y, v.x * v.z, v.y * v.x, v.y * v.y,
                      v.y * v.z, v.z * v.x, v.z * v.y, v.z * v.z);
}

inline double w(const Vector3D& r, const Matrix3x3D& g, double gDet) {
    static const double sigma = 315.0 / (64 * kPiD);
    return sigma * gDet * p((g * r).length());
}

//

AnisotropicPointsToImplicit3::AnisotropicPointsToImplicit3(
    double kernelRadius, double cutOffDensity, double positionSmoothingFactor,
    size_t minNumNeighbors)
    : _kernelRadius(kernelRadius),
      _cutOffDensity(cutOffDensity),
      _positionSmoothingFactor(positionSmoothingFactor),
      _minNumNeighbors(minNumNeighbors) {}

void AnisotropicPointsToImplicit3::convert(
    const ConstArrayAccessor1<Vector3D>& points, ScalarGrid3* output) const {
    if (output == nullptr) {
        JET_WARN << "Null scalar grid output pointer provided.";
        return;
    }

    const auto res = output->resolution();
    if (res.x * res.y * res.z == 0) {
        JET_WARN << "Empty grid is provided.";
        return;
    }

    const auto bbox = output->boundingBox();
    if (bbox.isEmpty()) {
        JET_WARN << "Empty domain is provided.";
        return;
    }

    const double h = _kernelRadius;
    const double invH = 1 / h;
    const double r = 2.0 * h;

    // Mean estimator for cov. mat.
    ParticleSystemData3 meanParticles;
    meanParticles.addParticles(points);
    meanParticles.buildNeighborSearcher(r);
    const auto meanNeighborSearcher = meanParticles.neighborSearcher();

    // Compute G and xMean
    std::vector<Matrix3x3D> gs(points.size());
    std::vector<Vector3D> xMeans(points.size());

    parallelFor(kZeroSize, points.size(), [&](size_t i) {
        const auto& x = points[i];

        // Compute xMean
        Vector3D xMean;
        double wSum = 0.0;
        size_t numNeighbors = 0;
        const auto getXMean = [&](size_t, const Vector3D& xj) {
            const double wj = wij((x - xj).length(), r);
            wSum += wj;
            xMean += wj * xj;
            ++numNeighbors;
        };
        meanNeighborSearcher->forEachNearbyPoint(x, r, getXMean);

        JET_ASSERT(wSum > 0.0);
        xMean /= wSum;

        xMeans[i] = lerp(x, xMean, _positionSmoothingFactor);

        if (numNeighbors < _minNumNeighbors) {
            const auto g = Matrix3x3D::makeScaleMatrix(invH, invH, invH);
            gs[i] = g;
        } else {
            // Compute covariance matrix
            // We start with small scale matrix (h*h) in order to
            // prevent zero covariance matrix when points are all
            // perfectly lined up.
            auto cov = Matrix3x3D::makeScaleMatrix(h * h, h * h, h * h);
            wSum = 0.0;
            const auto getCov = [&](size_t, const Vector3D& xj) {
                const double wj = wij((xMean - xj).length(), r);
                wSum += wj;
                cov += wj * vvt(xj - xMean);
            };
            meanNeighborSearcher->forEachNearbyPoint(x, r, getCov);

            cov /= wSum;

            // SVD
            Matrix3x3D u;
            Vector3D v;
            Matrix3x3D w;
            svd(cov, u, v, w);

            // Constrain Sigma
            const double maxSingularVal = v.absmax();
            const double kr = 4.0;
            v[0] = std::max(v[0], maxSingularVal / kr);
            v[1] = std::max(v[1], maxSingularVal / kr);
            v[2] = std::max(v[2], maxSingularVal / kr);
            const auto invSigma = Matrix3x3D::makeScaleMatrix(1.0 / v);

            // Compute G
            const double relV = v[0] * v[1] * v[2];  // area preservation
            const Matrix3x3D g = invH * std::pow(relV, 1.0 / 3.0) *
                                 (w * invSigma * u.transposed());
            gs[i] = g;
        }
    });

    // SPH estimator
    SphSystemData3 sphParticles;
    sphParticles.addParticles(
        ConstArrayAccessor1<Vector3D>(xMeans.size(), xMeans.data()));
    sphParticles.setKernelRadius(h);
    sphParticles.buildNeighborSearcher();
    sphParticles.updateDensities();
    const auto d = sphParticles.densities();
    const double m = sphParticles.mass();

    meanParticles.resize(0);
    meanParticles.addParticles(
        ConstArrayAccessor1<Vector3D>(xMeans.size(), xMeans.data()));
    meanParticles.buildNeighborSearcher(r);
    const auto meanNeighborSearcher3 = meanParticles.neighborSearcher();

    // Compute SDF
    auto temp = output->clone();
    temp->fill([&](const Vector3D& x) {
        double sum = 0.0;
        meanNeighborSearcher3->forEachNearbyPoint(
            x, r, [&](size_t i, const Vector3D& neighborPosition) {
                sum += m / d[i] *
                       w(neighborPosition - x, gs[i], gs[i].determinant());
            });

        return _cutOffDensity - sum;
    });

    FmmLevelSetSolver3 solver;
    solver.reinitialize(*temp, kMaxD, output);
}
