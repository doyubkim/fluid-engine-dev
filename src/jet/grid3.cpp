// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/grid3.h>
#include <jet/parallel.h>
#include <jet/serial.h>

#include <algorithm>
#include <fstream>
#include <iostream>

using namespace jet;

Grid3::Grid3() {
}

Grid3::~Grid3() {
}

const Size3& Grid3::resolution() const {
    return _resolution;
}

const Vector3D& Grid3::origin() const {
    return _origin;
}

const Vector3D& Grid3::gridSpacing() const {
    return _gridSpacing;
}

const BoundingBox3D& Grid3::boundingBox() const {
    return _boundingBox;
}

Grid3::DataPositionFunc Grid3::cellCenterPosition() const {
    Vector3D h = _gridSpacing;
    Vector3D o = _origin;
    return [h, o](size_t i, size_t j, size_t k) {
        return o + h * Vector3D(i + 0.5, j + 0.5, k + 0.5);
    };
}

void Grid3::forEachCellIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    serialFor(
        kZeroSize, _resolution.x,
        kZeroSize, _resolution.y,
        kZeroSize, _resolution.z,
        [this, &func](size_t i, size_t j, size_t k) {
            func(i, j, k);
        });
}

void Grid3::parallelForEachCellIndex(
    const std::function<void(size_t, size_t, size_t)>& func) const {
    parallelFor(
        kZeroSize, _resolution.x,
        kZeroSize, _resolution.y,
        kZeroSize, _resolution.z,
        [this, &func](size_t i, size_t j, size_t k) {
            func(i, j, k);
        });
}

bool Grid3::hasSameShape(const Grid3& other) const {
    return _resolution.x == other._resolution.x
        && _resolution.y == other._resolution.y
        && _resolution.z == other._resolution.z
        && similar(_gridSpacing.x, other._gridSpacing.x)
        && similar(_gridSpacing.y, other._gridSpacing.y)
        && similar(_gridSpacing.z, other._gridSpacing.z)
        && similar(_origin.x, other._origin.x)
        && similar(_origin.y, other._origin.y)
        && similar(_origin.z, other._origin.z);
}

void Grid3::setSizeParameters(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin) {
    _resolution = resolution;
    _origin = origin;
    _gridSpacing = gridSpacing;

    Vector3D resolutionD = Vector3D(
        static_cast<double>(resolution.x),
        static_cast<double>(resolution.y),
        static_cast<double>(resolution.z));

    _boundingBox = BoundingBox3D(
        origin,
        origin + gridSpacing * resolutionD);
}

void Grid3::swapGrid(Grid3* other) {
    std::swap(_resolution, other->_resolution);
    std::swap(_gridSpacing, other->_gridSpacing);
    std::swap(_origin, other->_origin);
    std::swap(_boundingBox, other->_boundingBox);
}

void Grid3::setGrid(const Grid3& other) {
    _resolution = other._resolution;
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _boundingBox = other._boundingBox;
}

void Grid3::serializeGrid(std::ostream* strm) const {
    uint64_t res64[3] = { _resolution.x, _resolution.y, _resolution.z };

    const char* resAsBytes = reinterpret_cast<const char*>(&res64);
    const char* gsAsBytes = reinterpret_cast<const char*>(&_gridSpacing);
    const char* orgAsBytes = reinterpret_cast<const char*>(&_origin);
    const char* boxAsBytes = reinterpret_cast<const char*>(&_boundingBox);

    strm->write(resAsBytes, 3 * sizeof(uint64_t));
    strm->write(gsAsBytes, 3 * sizeof(double));
    strm->write(orgAsBytes, 3 * sizeof(double));
    strm->write(boxAsBytes, 6 * sizeof(double));
}

void Grid3::deserializeGrid(std::istream* strm) {
    uint64_t res64[3];
    char* resAsBytes = reinterpret_cast<char*>(res64);
    char* gsAsBytes = reinterpret_cast<char*>(&_gridSpacing);
    char* orgAsBytes = reinterpret_cast<char*>(&_origin);
    char* boxAsBytes = reinterpret_cast<char*>(&_boundingBox);

    strm->read(resAsBytes, 3 * sizeof(uint64_t));
    strm->read(gsAsBytes, 3 * sizeof(double));
    strm->read(orgAsBytes, 3 * sizeof(double));
    strm->read(boxAsBytes, 6 * sizeof(double));

    _resolution = Size3(
        static_cast<size_t>(res64[0]),
        static_cast<size_t>(res64[1]),
        static_cast<size_t>(res64[2]));
}
