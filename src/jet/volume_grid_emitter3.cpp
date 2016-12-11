// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/collocated_vector_grid3.h>
#include <jet/face_centered_grid3.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit3.h>
#include <jet/volume_grid_emitter3.h>
#include <algorithm>

using namespace jet;

VolumeGridEmitter3::VolumeGridEmitter3(
    const ImplicitSurface3Ptr& sourceRegion,
    bool isOneShot)
: _sourceRegion(sourceRegion)
, _isOneShot(isOneShot) {
}

VolumeGridEmitter3::~VolumeGridEmitter3() {
}

void VolumeGridEmitter3::addSignedDistanceTarget(
    const ScalarGrid3Ptr& scalarGridTarget) {
    auto mapper = [] (double sdf, const Vector3D&) {
        return sdf;
    };
    auto blender = [] (double oldVal, double newVal) {
        return std::min(oldVal, newVal);
    };
    addTarget(scalarGridTarget, mapper, blender);
}

void VolumeGridEmitter3::addStepFunctionTarget(
    const ScalarGrid3Ptr& scalarGridTarget,
    double minValue,
    double maxValue) {
    double smoothingWidth = scalarGridTarget->gridSpacing().min();
    auto mapper = [minValue, maxValue, smoothingWidth, scalarGridTarget]
        (double sdf, const Vector3D&) {
            double step = 1.0 - smearedHeavisideSdf(sdf / smoothingWidth);
            return (maxValue - minValue) * step + minValue;
        };
    auto blender = [] (double oldVal, double newVal) {
        return std::max(oldVal, newVal);
    };
    addTarget(scalarGridTarget, mapper, blender);
}

void VolumeGridEmitter3::addTarget(
    const ScalarGrid3Ptr& scalarGridTarget,
    const ScalarMapper& customMapper,
    const ScalarBlender& blender) {
    _customScalarTargets.emplace_back(scalarGridTarget, customMapper, blender);
}

void VolumeGridEmitter3::addTarget(
    const VectorGrid3Ptr& vectorGridTarget,
    const VectorMapper& customMapper,
    const VectorBlender& blender) {
    _customVectorTargets.emplace_back(vectorGridTarget, customMapper, blender);
}

const ImplicitSurface3Ptr& VolumeGridEmitter3::sourceRegion() const {
    return _sourceRegion;
}

bool VolumeGridEmitter3::isOneShot() const {
    return _isOneShot;
}

void VolumeGridEmitter3::onUpdate(
    double currentTimeInSeconds,
    double timeIntervalInSeconds) {
    UNUSED_VARIABLE(currentTimeInSeconds);
    UNUSED_VARIABLE(timeIntervalInSeconds);

    if (_hasEmitted && _isOneShot) {
        return;
    }

    emit();

    _hasEmitted = true;
}

void VolumeGridEmitter3::emit() {
    for (const auto& target : _customScalarTargets) {
        const auto& grid = std::get<0>(target);
        const auto& mapper = std::get<1>(target);
        const auto& blender = std::get<2>(target);

        auto pos = grid->dataPosition();
        grid->parallelForEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                Vector3D gx = pos(i, j, k);
                double sdf = sourceRegion()->signedDistance(gx);
                (*grid)(i, j, k) = blender((*grid)(i, j, k), mapper(sdf, gx));
            });
    }

    for (const auto& target : _customVectorTargets) {
        const auto& grid = std::get<0>(target);
        const auto& mapper = std::get<1>(target);
        const auto& blender = std::get<2>(target);

        CollocatedVectorGrid3Ptr collocated
            = std::dynamic_pointer_cast<CollocatedVectorGrid3>(grid);
        if (collocated != nullptr) {
            auto pos = collocated->dataPosition();
            collocated->parallelForEachDataPointIndex(
                [&] (size_t i, size_t j, size_t k) {
                    Vector3D gx = pos(i, j, k);
                    double sdf = sourceRegion()->signedDistance(gx);
                    if (isInsideSdf(sdf)) {
                        (*collocated)(i, j, k)
                            = blender((*collocated)(i, j, k), mapper(sdf, gx));
                    }
                });
            continue;
        }

        FaceCenteredGrid3Ptr faceCentered
            = std::dynamic_pointer_cast<FaceCenteredGrid3>(grid);
        if (faceCentered != nullptr) {
            auto uPos = faceCentered->uPosition();
            auto vPos = faceCentered->vPosition();
            faceCentered->parallelForEachUIndex(
                [&] (size_t i, size_t j, size_t k) {
                    Vector3D gx = uPos(i, j, k);
                    double sdf = sourceRegion()->signedDistance(gx);
                    if (isInsideSdf(sdf)) {
                        Vector3D oldVal = faceCentered->sample(gx);
                        Vector3D newVal = mapper(sdf, gx);
                        faceCentered->u(i, j, k) = blender(oldVal, newVal).x;
                    }
                });
            faceCentered->parallelForEachVIndex(
                [&] (size_t i, size_t j, size_t k) {
                    Vector3D gx = vPos(i, j, k);
                    double sdf = sourceRegion()->signedDistance(gx);
                    if (isInsideSdf(sdf)) {
                        Vector3D oldVal = faceCentered->sample(gx);
                        Vector3D newVal = mapper(sdf, gx);
                        faceCentered->v(i, j, k) = blender(oldVal, newVal).y;
                    }
                });
            continue;
        }
    }
}

VolumeGridEmitter3::Builder VolumeGridEmitter3::builder() {
    return Builder();
}


VolumeGridEmitter3::Builder&
VolumeGridEmitter3::Builder::withSourceRegion(
    const Surface3Ptr& sourceRegion) {
    auto implicit = std::dynamic_pointer_cast<ImplicitSurface3>(sourceRegion);
    if (implicit != nullptr) {
        _sourceRegion = implicit;
    } else {
        _sourceRegion = std::make_shared<SurfaceToImplicit3>(sourceRegion);
    }
    return *this;
}

VolumeGridEmitter3::Builder&
VolumeGridEmitter3::Builder::withIsOneShot(bool isOneShot) {
    _isOneShot = isOneShot;
    return *this;
}

VolumeGridEmitter3 VolumeGridEmitter3::Builder::build() const {
    return VolumeGridEmitter3(_sourceRegion, _isOneShot);
}
