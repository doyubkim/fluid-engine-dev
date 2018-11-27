// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/grid.h>
#include <jet/iteration_utils.h>

#include <algorithm>
#include <fstream>
#include <iostream>

namespace jet {

template <size_t N>
Grid<N>::Grid() {}

template <size_t N>
Grid<N>::~Grid() {}

template <size_t N>
const Vector<size_t, N> &Grid<N>::resolution() const {
    return _resolution;
}

template <size_t N>
const Vector<double, N> &Grid<N>::origin() const {
    return _origin;
}

template <size_t N>
const Vector<double, N> &Grid<N>::gridSpacing() const {
    return _gridSpacing;
}

template <size_t N>
const BoundingBox<double, N> &Grid<N>::boundingBox() const {
    return _boundingBox;
}

template <size_t N>
GridDataPositionFunc<N> Grid<N>::cellCenterPosition() const {
    Vector<double, N> h = _gridSpacing;
    Vector<double, N> o = _origin;
    return GridDataPositionFunc<N>(
        [h, o](const Vector<size_t, N> &idx) -> Vector<double, N> {
            return o + elemMul(h, idx.template castTo<double>() +
                                      Vector<double, N>::makeConstant(0.5));
        });
}

template <size_t N>
void Grid<N>::forEachCellIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    forEachIndex(_resolution, [&func](auto... indices) {
        func(Vector<size_t, N>(indices...));
    });
}

template <size_t N>
void Grid<N>::parallelForEachCellIndex(
    const std::function<void(const Vector<size_t, N> &)> &func) const {
    parallelForEachIndex(
        Vector<size_t, N>::makeZero(), _resolution,
        [&func](auto... indices) { func(Vector<size_t, N>(indices...)); });
}

template <size_t N>
bool Grid<N>::hasSameShape(const Grid &other) const {
    return _resolution == other._resolution &&
           _gridSpacing.isSimilar(other._gridSpacing) &&
           _origin.isSimilar(other._origin);
}

template <size_t N>
void Grid<N>::setSizeParameters(const Vector<size_t, N> &resolution,
                                const Vector<double, N> &gridSpacing,
                                const Vector<double, N> &origin) {
    _resolution = resolution;
    _origin = origin;
    _gridSpacing = gridSpacing;

    Vector<double, N> resolutionD = resolution.template castTo<double>();

    _boundingBox = BoundingBox<double, N>(
        origin, origin + elemMul(gridSpacing, resolutionD));
}

template <size_t N>
void Grid<N>::swapGrid(Grid *other) {
    std::swap(_resolution, other->_resolution);
    std::swap(_gridSpacing, other->_gridSpacing);
    std::swap(_origin, other->_origin);
    std::swap(_boundingBox, other->_boundingBox);
}

template <size_t N>
void Grid<N>::setGrid(const Grid &other) {
    _resolution = other._resolution;
    _gridSpacing = other._gridSpacing;
    _origin = other._origin;
    _boundingBox = other._boundingBox;
}

template class Grid<2>;

template class Grid<3>;

}  // namespace jet
