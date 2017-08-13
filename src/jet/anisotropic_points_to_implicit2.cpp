// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/anisotropic_points_to_implicit2.h>
#include <jet/fmm_level_set_solver2.h>
#include <jet/jet.h>
#include <jet/sph_kernels2.h>
#include <jet/sph_system_data2.h>
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

// inline double w(double distance, double h) {
//    static const double sigma = 4.0 / kPiD;
//    const double h2 = h * h;
//    return sigma * h2 * p(distance / h);
//}

inline double wij(double distance, double r) {
    if (distance < r) {
        return 1.0 - cubic(distance / r);
    } else {
        return 0.0;
    }
}

inline Matrix2x2D vvt(const Vector2D& v) {
    return Matrix2x2D(v.x * v.x, v.x * v.y, v.y * v.x, v.y * v.y);
}

inline double w(const Vector2D& r, const Matrix2x2D& g, double gDet) {
    static const double sigma = 4.0 / kPiD;
    return sigma * gDet * p((g * r).length());
}

template <typename M>
void printMat(const M& m) {
    printf("%f, %f\n", m(0, 0), m(0, 1));
    printf("%f, %f\n", m(1, 0), m(1, 1));
}

AnisotropicPointsToImplicit2::AnisotropicPointsToImplicit2(double kernelRadius,
                                                           double cutOffDensity)
    : _kernelRadius(kernelRadius), _cutOffDensity(cutOffDensity) {}

void AnisotropicPointsToImplicit2::convert(
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

    const double h = _kernelRadius;
    const double invH = 1 / h;
    const double r = 2.0 * h;

    // Mean estimator for cov. mat.
    ParticleSystemData2 meanParticles;
    meanParticles.addParticles(points);
    meanParticles.buildNeighborSearcher(r);
    const auto meanNeighborSearcher = meanParticles.neighborSearcher();

    // Compute G and xMean
    std::vector<Matrix2x2D> gs(points.size());

    parallelFor(
        kZeroSize, points.size(),
        [&](size_t i) {
            const auto& x = points[i];
            const size_t ne = 1;

            // Compute xMean
            Vector2D xMean;
            double wSum = 0.0;
            size_t numNeighbors = 0;
            const auto getXMean = [&](size_t, const Vector2D& xj) {
                const double wj = wij((x - xj).length(), r);
                wSum += wj;
                xMean += wj * xj;
                ++numNeighbors;
            };
            meanNeighborSearcher->forEachNearbyPoint(x, r, getXMean);

            if (numNeighbors < ne) {
                const auto g = Matrix2x2D::makeScaleMatrix(invH, invH);
                gs[i] = g;
            } else {
                JET_ASSERT(wSum > 0.0);
                xMean /= wSum;

                // Compute covariance matrix
                Matrix2x2D cov;
                wSum = 0.0;
                const auto getCov = [&](size_t, const Vector2D& xj) {
                    const double wj = wij((x - xj).length(), r);
                    wSum += wj;
                    cov += wj * vvt(xj - xMean);
                };
                meanNeighborSearcher->forEachNearbyPoint(x, r, getCov);

                cov /= wSum;

                // Scaling
                cov /= square(cov.norm2());

                // SVD
                Matrix2x2D u;
                Vector2D v;
                Matrix2x2D w;
                svd(cov, u, v, w);

                // Constrain Sigma
                const double maxSingularVal = v.absmax();
                const double kr = 4.0;
                Matrix2x2D invSigma;
                invSigma(0, 0) = 1.0 / std::max(v[0], maxSingularVal / kr);
                invSigma(1, 1) = 1.0 / std::max(v[1], maxSingularVal / kr);

                // Compute G
                const double relA = v[0] * v[1];  // area preservation
                const auto g =
                    invH * std::sqrt(relA) * (w * invSigma * u.transposed());
                gs[i] = g;
            }
        },
        ExecutionPolicy::kSerial);

    // Compute SDF

    // SPH estimator
    SphSystemData2 sphParticles;
    sphParticles.addParticles(points);
    sphParticles.setKernelRadius(h);
    sphParticles.buildNeighborSearcher();
    sphParticles.updateDensities();
    const auto d = sphParticles.densities();
    const double m = sphParticles.mass();
    const auto sphNeighborSearcher = sphParticles.neighborSearcher();

    auto temp = output->clone();
    temp->fill([&](const Vector2D& x) {
        double sum = 0.0;
        sphNeighborSearcher->forEachNearbyPoint(
            x, _kernelRadius, [&](size_t i, const Vector2D& neighborPosition) {
                sum += m / d[i] *
                       w(neighborPosition - x, gs[i], gs[i].determinant());
            });

        return _cutOffDensity - sum;
    });

    FmmLevelSetSolver2 solver;
    solver.reinitialize(*temp, kMaxD, output);
}
