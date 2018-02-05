// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/array_samplers2.h>
#include <jet/parallel.h>
#include <jet/semi_lagrangian2.h>
#include <algorithm>

using namespace jet;

SemiLagrangian2::SemiLagrangian2() {
}

SemiLagrangian2::~SemiLagrangian2() {
}

void SemiLagrangian2::advect(
    const ScalarGrid2& input,
    const VectorField2& flow,
    double dt,
    ScalarGrid2* output,
    const ScalarField2& boundarySdf) {
    auto outputDataPos = output->dataPosition();
    auto outputDataAcc = output->dataAccessor();
    auto inputSamplerFunc = getScalarSamplerFunc(input);
    auto inputDataPos = input.dataPosition();

    double h = std::min(output->gridSpacing().x, output->gridSpacing().y);

    output->parallelForEachDataPointIndex([&](size_t i, size_t j) {
        if (boundarySdf.sample(inputDataPos(i, j)) > 0.0) {
            Vector2D pt = backTrace(
                flow, dt, h, outputDataPos(i, j), boundarySdf);
            outputDataAcc(i, j) = inputSamplerFunc(pt);
        }
    });
}

void SemiLagrangian2::advect(
    const CollocatedVectorGrid2& input,
    const VectorField2& flow,
    double dt,
    CollocatedVectorGrid2* output,
    const ScalarField2& boundarySdf) {
    auto inputSamplerFunc = getVectorSamplerFunc(input);

    double h = std::min(output->gridSpacing().x, output->gridSpacing().y);

    auto outputDataPos = output->dataPosition();
    auto outputDataAcc = output->dataAccessor();
    auto inputDataPos = input.dataPosition();

    output->parallelForEachDataPointIndex([&](size_t i, size_t j) {
        if (boundarySdf.sample(inputDataPos(i, j)) > 0.0) {
            Vector2D pt = backTrace(
                flow, dt, h, outputDataPos(i, j), boundarySdf);
            outputDataAcc(i, j) = inputSamplerFunc(pt);
        }
    });
}

void SemiLagrangian2::advect(
    const FaceCenteredGrid2& input,
    const VectorField2& flow,
    double dt,
    FaceCenteredGrid2* output,
    const ScalarField2& boundarySdf) {
    auto inputSamplerFunc = getVectorSamplerFunc(input);

    double h = std::min(output->gridSpacing().x, output->gridSpacing().y);

    auto uTargetDataPos = output->uPosition();
    auto uTargetDataAcc = output->uAccessor();
    auto uSourceDataPos = input.uPosition();

    output->parallelForEachUIndex([&](size_t i, size_t j) {
        if (boundarySdf.sample(uSourceDataPos(i, j)) > 0.0) {
            Vector2D pt = backTrace(
                flow, dt, h, uTargetDataPos(i, j), boundarySdf);
            uTargetDataAcc(i, j) = inputSamplerFunc(pt).x;
        }
    });

    auto vTargetDataPos = output->vPosition();
    auto vTargetDataAcc = output->vAccessor();
    auto vSourceDataPos = input.vPosition();

    output->parallelForEachVIndex([&](size_t i, size_t j) {
        if (boundarySdf.sample(vSourceDataPos(i, j)) > 0.0) {
            Vector2D pt = backTrace(
                flow, dt, h, vTargetDataPos(i, j), boundarySdf);
            vTargetDataAcc(i, j) = inputSamplerFunc(pt).y;
        }
    });
}

Vector2D SemiLagrangian2::backTrace(
    const VectorField2& flow,
    double dt,
    double h,
    const Vector2D& startPt,
    const ScalarField2& boundarySdf) {

    double remainingT = dt;
    Vector2D pt0 = startPt;
    Vector2D pt1 = startPt;

    while (remainingT > kEpsilonD) {
        // Adaptive time-stepping
        Vector2D vel0 = flow.sample(pt0);
        double numSubSteps
            = std::max(std::ceil(vel0.length() * remainingT / h), 1.0);
        dt = remainingT / numSubSteps;

        // Mid-point rule
        Vector2D midPt = pt0 - 0.5 * dt * vel0;
        Vector2D midVel = flow.sample(midPt);
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

std::function<double(const Vector2D&)>
SemiLagrangian2::getScalarSamplerFunc(const ScalarGrid2& input) const {
    return input.sampler();
}

std::function<Vector2D(const Vector2D&)>
SemiLagrangian2::getVectorSamplerFunc(
    const CollocatedVectorGrid2& input) const {
    return input.sampler();
}

std::function<Vector2D(const Vector2D&)>
SemiLagrangian2::getVectorSamplerFunc(const FaceCenteredGrid2& input) const {
    return input.sampler();
}
