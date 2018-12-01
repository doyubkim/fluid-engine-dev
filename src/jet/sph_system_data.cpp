// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// Adopted from the sample code of:
// Bart Adams and Martin Wicke,
// "Meshless Approximation Methods and Applications in Physics Based Modeling
// and Animation", Eurographics 2009 Tutorial

#include <pch.h>

#include <fbs_helpers.h>
#include <generated/sph_system_data2_generated.h>
#include <generated/sph_system_data3_generated.h>

#include <jet/bcc_lattice_point_generator.h>
#include <jet/parallel.h>
#include <jet/sph_kernels.h>
#include <jet/sph_system_data.h>
#include <jet/triangle_point_generator.h>

namespace jet {

// MARK: Serialization helpers

template <size_t N>
struct GetFlatbuffersSphSystemData {};

template <>
struct GetFlatbuffersSphSystemData<2> {
    using Offset = flatbuffers::Offset<fbs::SphSystemData2>;
    using BaseOffset = flatbuffers::Offset<jet::fbs::ParticleSystemData2>;

    static const fbs::SphSystemData2* getSphSystemData(const uint8_t* data) {
        return fbs::GetSphSystemData2(data);
    }

    static Offset createSphSystemData(flatbuffers::FlatBufferBuilder& _fbb,
                                      BaseOffset base, double targetDensity,
                                      double targetSpacing,
                                      double kernelRadiusOverTargetSpacing,
                                      double kernelRadius, uint64_t pressureIdx,
                                      uint64_t densityIdx) {
        return fbs::CreateSphSystemData2(_fbb, base, targetDensity,
                                         targetSpacing,
                                         kernelRadiusOverTargetSpacing,
                                         kernelRadius, pressureIdx, densityIdx);
    }
};

template <>
struct GetFlatbuffersSphSystemData<3> {
    using Offset = flatbuffers::Offset<fbs::SphSystemData3>;
    using BaseOffset = flatbuffers::Offset<jet::fbs::ParticleSystemData3>;

    static const fbs::SphSystemData3* getSphSystemData(const uint8_t* data) {
        return fbs::GetSphSystemData3(data);
    }

    static Offset createSphSystemData(flatbuffers::FlatBufferBuilder& _fbb,
                                      BaseOffset base, double targetDensity,
                                      double targetSpacing,
                                      double kernelRadiusOverTargetSpacing,
                                      double kernelRadius, uint64_t pressureIdx,
                                      uint64_t densityIdx) {
        return fbs::CreateSphSystemData3(_fbb, base, targetDensity,
                                         targetSpacing,
                                         kernelRadiusOverTargetSpacing,
                                         kernelRadius, pressureIdx, densityIdx);
    }
};

// MARK: ParticleSystemData implementations

template <size_t N>
SphSystemData<N>::SphSystemData() : SphSystemData(0) {}

template <size_t N>
SphSystemData<N>::SphSystemData(size_t numberOfParticles)
    : ParticleSystemData<N>(numberOfParticles) {
    _densityIdx = addScalarData();
    _pressureIdx = addScalarData();

    setTargetSpacing(_targetSpacing);
}

template <size_t N>
SphSystemData<N>::SphSystemData(const SphSystemData& other) {
    set(other);
}

template <size_t N>
SphSystemData<N>::~SphSystemData() {}

template <size_t N>
void SphSystemData<N>::setRadius(double newRadius) {
    // Interpret it as setting target spacing
    setTargetSpacing(newRadius);
}

template <size_t N>
void SphSystemData<N>::setMass(double newMass) {
    double incRatio = newMass / mass();
    _targetDensity *= incRatio;
    ParticleSystemData<N>::setMass(newMass);
}

template <size_t N>
ConstArrayView1<double> SphSystemData<N>::densities() const {
    return scalarDataAt(_densityIdx);
}

template <size_t N>
ArrayView1<double> SphSystemData<N>::densities() {
    return scalarDataAt(_densityIdx);
}

template <size_t N>
ConstArrayView1<double> SphSystemData<N>::pressures() const {
    return scalarDataAt(_pressureIdx);
}

template <size_t N>
ArrayView1<double> SphSystemData<N>::pressures() {
    return scalarDataAt(_pressureIdx);
}

template <size_t N>
void SphSystemData<N>::updateDensities() {
    auto p = positions();
    auto d = densities();
    const double m = mass();

    parallelFor(kZeroSize, numberOfParticles(), [&](size_t i) {
        double sum = sumOfKernelNearby(p[i]);
        d[i] = m * sum;
    });
}

template <size_t N>
void SphSystemData<N>::setTargetDensity(double targetDensity) {
    _targetDensity = targetDensity;

    computeMass();
}

template <size_t N>
double SphSystemData<N>::targetDensity() const {
    return _targetDensity;
}

template <size_t N>
void SphSystemData<N>::setTargetSpacing(double spacing) {
    ParticleSystemData<N>::setRadius(spacing);

    _targetSpacing = spacing;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

template <size_t N>
double SphSystemData<N>::targetSpacing() const {
    return _targetSpacing;
}

template <size_t N>
void SphSystemData<N>::setRelativeKernelRadius(double relativeRadius) {
    _kernelRadiusOverTargetSpacing = relativeRadius;
    _kernelRadius = _kernelRadiusOverTargetSpacing * _targetSpacing;

    computeMass();
}

template <size_t N>
double SphSystemData<N>::relativeKernelRadius() const {
    return _kernelRadiusOverTargetSpacing;
}

template <size_t N>
void SphSystemData<N>::setKernelRadius(double kernelRadius) {
    _kernelRadius = kernelRadius;
    _targetSpacing = kernelRadius / _kernelRadiusOverTargetSpacing;

    computeMass();
}

template <size_t N>
double SphSystemData<N>::kernelRadius() const {
    return _kernelRadius;
}

template <size_t N>
double SphSystemData<N>::sumOfKernelNearby(
    const Vector<double, N>& origin) const {
    double sum = 0.0;
    SphStdKernel<N> kernel(_kernelRadius);
    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius,
        [&](size_t, const Vector<double, N>& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            sum += kernel(dist);
        });
    return sum;
}

template <size_t N>
double SphSystemData<N>::interpolate(
    const Vector<double, N>& origin,
    const ConstArrayView1<double>& values) const {
    double sum = 0.0;
    auto d = densities();
    SphStdKernel<N> kernel(_kernelRadius);
    const double m = mass();

    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius,
        [&](size_t i, const Vector<double, N>& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = m / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

template <size_t N>
Vector<double, N> SphSystemData<N>::interpolate(
    const Vector<double, N>& origin,
    const ConstArrayView1<Vector<double, N>>& values) const {
    Vector<double, N> sum;
    auto d = densities();
    SphStdKernel<N> kernel(_kernelRadius);
    const double m = mass();

    neighborSearcher()->forEachNearbyPoint(
        origin, _kernelRadius,
        [&](size_t i, const Vector<double, N>& neighborPosition) {
            double dist = origin.distanceTo(neighborPosition);
            double weight = m / d[i] * kernel(dist);
            sum += weight * values[i];
        });

    return sum;
}

template <size_t N>
Vector<double, N> SphSystemData<N>::gradientAt(
    size_t i, const ConstArrayView1<double>& values) const {
    Vector<double, N> sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector<double, N> origin = p[i];
    SphSpikyKernel<N> kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector<double, N> neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        if (dist > 0.0) {
            Vector<double, N> dir = (neighborPosition - origin) / dist;
            sum += d[i] * m *
                   (values[i] / square(d[i]) + values[j] / square(d[j])) *
                   kernel.gradient(dist, dir);
        }
    }

    return sum;
}

template <size_t N>
double SphSystemData<N>::laplacianAt(
    size_t i, const ConstArrayView1<double>& values) const {
    double sum = 0.0;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector<double, N> origin = p[i];
    SphSpikyKernel<N> kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector<double, N> neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum +=
            m * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

template <size_t N>
Vector<double, N> SphSystemData<N>::laplacianAt(
    size_t i, const ConstArrayView1<Vector<double, N>>& values) const {
    Vector<double, N> sum;
    auto p = positions();
    auto d = densities();
    const auto& neighbors = neighborLists()[i];
    Vector<double, N> origin = p[i];
    SphSpikyKernel<N> kernel(_kernelRadius);
    const double m = mass();

    for (size_t j : neighbors) {
        Vector<double, N> neighborPosition = p[j];
        double dist = origin.distanceTo(neighborPosition);
        sum +=
            m * (values[j] - values[i]) / d[j] * kernel.secondDerivative(dist);
    }

    return sum;
}

template <size_t N>
void SphSystemData<N>::buildNeighborSearcher() {
    ParticleSystemData<N>::buildNeighborSearcher(_kernelRadius);
}

template <size_t N>
void SphSystemData<N>::buildNeighborLists() {
    ParticleSystemData<N>::buildNeighborLists(_kernelRadius);
}

template <size_t N>
struct GetPointGenerator {};

template <>
struct GetPointGenerator<2> {
    using type = TrianglePointGenerator;
};

template <>
struct GetPointGenerator<3> {
    using type = BccLatticePointGenerator;
};

template <size_t N>
void SphSystemData<N>::computeMass() {
    Array1<Vector<double, N>> points;
    typename GetPointGenerator<N>::type pointsGenerator;
    BoundingBox<double, N> sampleBound(
        Vector<double, N>::makeConstant(-1.5 * _kernelRadius),
        Vector<double, N>::makeConstant(1.5 * _kernelRadius));

    pointsGenerator.generate(sampleBound, _targetSpacing, &points);

    double maxNumberDensity = 0.0;
    SphStdKernel<N> kernel(_kernelRadius);

    for (size_t i = 0; i < points.length(); ++i) {
        const Vector<double, N>& point = points[i];
        double sum = 0.0;

        for (size_t j = 0; j < points.length(); ++j) {
            const Vector<double, N>& neighborPoint = points[j];
            sum += kernel(neighborPoint.distanceTo(point));
        }

        maxNumberDensity = std::max(maxNumberDensity, sum);
    }

    JET_ASSERT(maxNumberDensity > 0);

    double newMass = _targetDensity / maxNumberDensity;

    ParticleSystemData<N>::setMass(newMass);
}

template <size_t N>
void SphSystemData<N>::serialize(std::vector<uint8_t>* buffer) const {
    flatbuffers::FlatBufferBuilder builder(1024);
    typename GetFlatbuffersSphSystemData<N>::BaseOffset fbsParticleSystemData;

    ParticleSystemData<N>::serialize(*this, &builder, &fbsParticleSystemData);

    auto fbsSphSystemData = GetFlatbuffersSphSystemData<N>::createSphSystemData(
        builder, fbsParticleSystemData, _targetDensity, _targetSpacing,
        _kernelRadiusOverTargetSpacing, _kernelRadius, _pressureIdx,
        _densityIdx);

    builder.Finish(fbsSphSystemData);

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    buffer->resize(size);
    memcpy(buffer->data(), buf, size);
}

template <size_t N>
void SphSystemData<N>::deserialize(const std::vector<uint8_t>& buffer) {
    auto fbsSphSystemData =
        GetFlatbuffersSphSystemData<N>::getSphSystemData(buffer.data());

    auto base = fbsSphSystemData->base();
    ParticleSystemData<N>::deserialize(base, *this);

    // SPH specific
    _targetDensity = fbsSphSystemData->targetDensity();
    _targetSpacing = fbsSphSystemData->targetSpacing();
    _kernelRadiusOverTargetSpacing =
        fbsSphSystemData->kernelRadiusOverTargetSpacing();
    _kernelRadius = fbsSphSystemData->kernelRadius();
    _pressureIdx = static_cast<size_t>(fbsSphSystemData->pressureIdx());
    _densityIdx = static_cast<size_t>(fbsSphSystemData->densityIdx());
}

template <size_t N>
void SphSystemData<N>::set(const SphSystemData& other) {
    ParticleSystemData<N>::set(other);

    _targetDensity = other._targetDensity;
    _targetSpacing = other._targetSpacing;
    _kernelRadiusOverTargetSpacing = other._kernelRadiusOverTargetSpacing;
    _kernelRadius = other._kernelRadius;
    _densityIdx = other._densityIdx;
    _pressureIdx = other._pressureIdx;
}

template <size_t N>
SphSystemData<N>& SphSystemData<N>::operator=(const SphSystemData& other) {
    set(other);
    return *this;
}

template class SphSystemData<2>;

template class SphSystemData<3>;

}  // namespace jet
