// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/collocated_vector_grid2.h>
#include <jet/face_centered_grid2.h>
#include <jet/level_set_utils.h>
#include <jet/surface_to_implicit2.h>
#include <jet/volume_grid_emitter2.h>
#include <algorithm>

using namespace jet;

VolumeGridEmitter2::VolumeGridEmitter2(
    const ImplicitSurface2Ptr& sourceRegion,
    bool isOneShot)
: _sourceRegion(sourceRegion)
, _isOneShot(isOneShot) {
}

VolumeGridEmitter2::~VolumeGridEmitter2() {
}

void VolumeGridEmitter2::addSignedDistanceTarget(
    const ScalarGrid2Ptr& scalarGridTarget) {
    auto mapper = [] (double sdf, const Vector2D&) {
        return sdf;
    };
    auto blender = [] (double oldVal, double newVal) {
        return std::min(oldVal, newVal);
    };
    addTarget(scalarGridTarget, mapper, blender);
}

void VolumeGridEmitter2::addStepFunctionTarget(
    const ScalarGrid2Ptr& scalarGridTarget,
    double minValue,
    double maxValue) {
    double smoothingWidth = scalarGridTarget->gridSpacing().min();
    auto mapper = [minValue, maxValue, smoothingWidth, scalarGridTarget]
        (double sdf, const Vector2D&) {
            double step = 1.0 - smearedHeavisideSdf(sdf / smoothingWidth);
            return (maxValue - minValue) * step + minValue;
        };
    auto blender = [] (double oldVal, double newVal) {
        return std::max(oldVal, newVal);
    };
    addTarget(scalarGridTarget, mapper, blender);
}

void VolumeGridEmitter2::addTarget(
    const ScalarGrid2Ptr& scalarGridTarget,
    const ScalarMapper& customMapper,
    const ScalarBlender& blender) {
    _customScalarTargets.emplace_back(scalarGridTarget, customMapper, blender);
}

void VolumeGridEmitter2::addTarget(
    const VectorGrid2Ptr& vectorGridTarget,
    const VectorMapper& customMapper,
    const VectorBlender& blender) {
    _customVectorTargets.emplace_back(vectorGridTarget, customMapper, blender);
}

const ImplicitSurface2Ptr& VolumeGridEmitter2::sourceRegion() const {
    return _sourceRegion;
}

bool VolumeGridEmitter2::isOneShot() const {
    return _isOneShot;
}

void VolumeGridEmitter2::onUpdate(
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

void VolumeGridEmitter2::emit() {
    for (const auto& target : _customScalarTargets) {
        const auto& grid = std::get<0>(target);
        const auto& mapper = std::get<1>(target);
        const auto& blender = std::get<2>(target);

        auto pos = grid->dataPosition();
        grid->parallelForEachDataPointIndex(
            [&] (size_t i, size_t j) {
                Vector2D gx = pos(i, j);
                double sdf = sourceRegion()->signedDistance(gx);
                (*grid)(i, j) = blender((*grid)(i, j), mapper(sdf, gx));
            });
    }

    for (const auto& target : _customVectorTargets) {
        const auto& grid = std::get<0>(target);
        const auto& mapper = std::get<1>(target);
        const auto& blender = std::get<2>(target);

        CollocatedVectorGrid2Ptr collocated
            = std::dynamic_pointer_cast<CollocatedVectorGrid2>(grid);
        if (collocated != nullptr) {
            auto pos = collocated->dataPosition();
            collocated->parallelForEachDataPointIndex(
                [&] (size_t i, size_t j) {
                    Vector2D gx = pos(i, j);
                    double sdf = sourceRegion()->signedDistance(gx);
                    if (isInsideSdf(sdf)) {
                        (*collocated)(i, j)
                            = blender((*collocated)(i, j), mapper(sdf, gx));
                    }
                });
            continue;
        }

        FaceCenteredGrid2Ptr faceCentered
            = std::dynamic_pointer_cast<FaceCenteredGrid2>(grid);
        if (faceCentered != nullptr) {
            auto uPos = faceCentered->uPosition();
            auto vPos = faceCentered->vPosition();
            faceCentered->parallelForEachUIndex(
                [&] (size_t i, size_t j) {
                    Vector2D gx = uPos(i, j);
                    double sdf = sourceRegion()->signedDistance(gx);
                    if (isInsideSdf(sdf)) {
                        Vector2D oldVal = faceCentered->sample(gx);
                        Vector2D newVal = mapper(sdf, gx);
                        faceCentered->u(i, j) = blender(oldVal, newVal).x;
                    }
                });
            faceCentered->parallelForEachVIndex(
                [&] (size_t i, size_t j) {
                    Vector2D gx = vPos(i, j);
                    double sdf = sourceRegion()->signedDistance(gx);
                    if (isInsideSdf(sdf)) {
                        Vector2D oldVal = faceCentered->sample(gx);
                        Vector2D newVal = mapper(sdf, gx);
                        faceCentered->v(i, j) = blender(oldVal, newVal).y;
                    }
                });
            continue;
        }
    }
}

VolumeGridEmitter2::Builder VolumeGridEmitter2::builder() {
    return Builder();
}


VolumeGridEmitter2::Builder&
VolumeGridEmitter2::Builder::withSourceRegion(
    const Surface2Ptr& sourceRegion) {
    auto implicit = std::dynamic_pointer_cast<ImplicitSurface2>(sourceRegion);
    if (implicit != nullptr) {
        _sourceRegion = implicit;
    } else {
        _sourceRegion = std::make_shared<SurfaceToImplicit2>(sourceRegion);
    }
    return *this;
}

VolumeGridEmitter2::Builder&
VolumeGridEmitter2::Builder::withIsOneShot(bool isOneShot) {
    _isOneShot = isOneShot;
    return *this;
}

VolumeGridEmitter2 VolumeGridEmitter2::Builder::build() const {
    return VolumeGridEmitter2(_sourceRegion, _isOneShot);
}
