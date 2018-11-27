// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_samplers.h>
#include <jet/parallel.h>
#include <jet/semi_lagrangian3.h>

using namespace jet;

SemiLagrangian3::SemiLagrangian3() {}

SemiLagrangian3::~SemiLagrangian3() {}

void SemiLagrangian3::advect(const ScalarGrid3& input, const VectorField3& flow,
                             double dt, ScalarGrid3* output,
                             const ScalarField3& boundarySdf) {
    auto outputDataPos = output->dataPosition();
    auto outputDataAcc = output->dataView();
    auto inputSamplerFunc = getScalarSamplerFunc(input);
    auto inputDataPos = input.dataPosition();

    double h = min3(output->gridSpacing().x, output->gridSpacing().y,
                    output->gridSpacing().z);

    output->parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        if (boundarySdf.sample(inputDataPos(i, j, k)) > 0.0) {
            Vector3D pt =
                backTrace(flow, dt, h, outputDataPos(i, j, k), boundarySdf);
            outputDataAcc(i, j, k) = inputSamplerFunc(pt);
        }
    });
}

void SemiLagrangian3::advect(const CollocatedVectorGrid3& input,
                             const VectorField3& flow, double dt,
                             CollocatedVectorGrid3* output,
                             const ScalarField3& boundarySdf) {
    auto inputSamplerFunc = getVectorSamplerFunc(input);

    double h = min3(output->gridSpacing().x, output->gridSpacing().y,
                    output->gridSpacing().z);

    auto outputDataPos = output->dataPosition();
    auto outputDataAcc = output->dataView();
    auto inputDataPos = input.dataPosition();

    output->parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        if (boundarySdf.sample(inputDataPos(i, j, k)) > 0.0) {
            Vector3D pt =
                backTrace(flow, dt, h, outputDataPos(i, j, k), boundarySdf);
            outputDataAcc(i, j, k) = inputSamplerFunc(pt);
        }
    });
}

void SemiLagrangian3::advect(const FaceCenteredGrid3& input,
                             const VectorField3& flow, double dt,
                             FaceCenteredGrid3* output,
                             const ScalarField3& boundarySdf) {
    auto inputSamplerFunc = getVectorSamplerFunc(input);

    double h = min3(output->gridSpacing().x, output->gridSpacing().y,
                    output->gridSpacing().z);

    auto uTargetDataPos = output->uPosition();
    auto uTargetDataAcc = output->uView();
    auto uSourceDataPos = input.uPosition();

    output->parallelForEachUIndex([&](const Vector3UZ& idx) {
        if (boundarySdf.sample(uSourceDataPos(idx)) > 0.0) {
            Vector3D pt =
                backTrace(flow, dt, h, uTargetDataPos(idx), boundarySdf);
            uTargetDataAcc(idx) = inputSamplerFunc(pt).x;
        }
    });

    auto vTargetDataPos = output->vPosition();
    auto vTargetDataAcc = output->vView();
    auto vSourceDataPos = input.vPosition();

    output->parallelForEachVIndex([&](const Vector3UZ& idx) {
        if (boundarySdf.sample(vSourceDataPos(idx)) > 0.0) {
            Vector3D pt =
                backTrace(flow, dt, h, vTargetDataPos(idx), boundarySdf);
            vTargetDataAcc(idx) = inputSamplerFunc(pt).y;
        }
    });

    auto wTargetDataPos = output->wPosition();
    auto wTargetDataAcc = output->wView();
    auto wSourceDataPos = input.wPosition();

    output->parallelForEachWIndex([&](const Vector3UZ& idx) {
        if (boundarySdf.sample(wSourceDataPos(idx)) > 0.0) {
            Vector3D pt =
                backTrace(flow, dt, h, wTargetDataPos(idx), boundarySdf);
            wTargetDataAcc(idx) = inputSamplerFunc(pt).z;
        }
    });
}

Vector3D SemiLagrangian3::backTrace(const VectorField3& flow, double dt,
                                    double h, const Vector3D& startPt,
                                    const ScalarField3& boundarySdf) {
    double remainingT = dt;
    Vector3D pt0 = startPt;
    Vector3D pt1 = startPt;

    while (remainingT > kEpsilonD) {
        // Adaptive time-stepping
        Vector3D vel0 = flow.sample(pt0);
        double numSubSteps =
            std::max(std::ceil(vel0.length() * remainingT / h), 1.0);
        dt = remainingT / numSubSteps;

        // Mid-point rule
        Vector3D midPt = pt0 - 0.5 * dt * vel0;
        Vector3D midVel = flow.sample(midPt);
        pt1 = pt0 - dt * midVel;

        // Boundary handling
        double phi0 = boundarySdf.sample(pt0);
        double phi1 = boundarySdf.sample(pt1);

        if (phi0 * phi1 < 0.0) {
            double w = std::fabs(phi1) / (std::fabs(phi0) + std::fabs(phi1));
            pt1 = w * pt0 + (1.0 - w) * pt1;
            break;
        }

        remainingT -= dt;
        pt0 = pt1;
    }

    return pt1;
}

std::function<double(const Vector3D&)> SemiLagrangian3::getScalarSamplerFunc(
    const ScalarGrid3& input) const {
    return input.sampler();
}

std::function<Vector3D(const Vector3D&)> SemiLagrangian3::getVectorSamplerFunc(
    const CollocatedVectorGrid3& input) const {
    return input.sampler();
}

std::function<Vector3D(const Vector3D&)> SemiLagrangian3::getVectorSamplerFunc(
    const FaceCenteredGrid3& input) const {
    return input.sampler();
}
