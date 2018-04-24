// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/grid2.h>
#include <jet/parallel.h>
#include <jet/serial.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>  // just make cpplint happy..

using namespace jet;

Grid2::Grid2() {}

Grid2::~Grid2() {}

const Size2& Grid2::resolution() const { return _resolution; }

const Vector2D& Grid2::origin() const { return _origin; }

const Vector2D& Grid2::gridSpacing() const { return _gridSpacing; }

const BoundingBox2D& Grid2::boundingBox() const { return _boundingBox; }

Grid2::DataPositionFunc Grid2::cellCenterPosition() const {
    Vector2D h = _gridSpacing;
    Vector2D o = _origin;
    return [h, o](size_t i, size_t j) {
        return o + h * Vector2D(i + 0.5, j + 0.5);
    };
}

void Grid2::forEachCellIndex(
    const std::function<void(size_t, size_t)>& func) const {
    serialFor(kZeroSize, _resolution.x, kZeroSize, _resolution.y,
              [&func](size_t i, size_t j) { func(i, j); });
}

void Grid2::parallelForEachCellIndex(
    const std::function<void(size_t, size_t)>& func) const {
    parallelFor(kZeroSize, _resolution.x, kZeroSize, _resolution.y,
                [&func](size_t i, size_t j) { func(i, j); });
}

bool Grid2::hasSameShape(const Grid2& other) const {
    return _resolution.x == other._resolution.x &&
           _resolution.y == other._resolution.y &&
           similar(_gridSpacing.x, other._gridSpacing.x) &&
           similar(_gridSpacing.y, other._gridSpacing.y) &&
           similar(_origin.x, other._origin.x) &&
           similar(_origin.y, other._origin.y);
}

void Grid2::setSizeParameters(const Size2& resolution,
                              const Vector2D& gridSpacing,
                              const Vector2D& origin) {
    _resolution = resolution;
    _origin = origin;
    _gridSpacing = gridSpacing;

    Vector2D resolutionD = Vector2D(static_cast<double>(resolution.x),
                                    static_cast<double>(resolution.y));

    _boundingBox = BoundingBox2D(origin, origin + gridSpacing * resolutionD);
}

void Grid2::swapGrid(Grid2* other) {
    std::swap(_resolution, other->_resolution);
    std::swap(_gridSpacing, other->_gridSpacing);
    std::swap(_origin, other->_origin);
    std::swap(_boundingBox, other->_boundingBox);
}

void Grid2::setGrid(const Grid2& other) {
    _resolution = other._resolution;
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _boundingBox = other._boundingBox;
}
