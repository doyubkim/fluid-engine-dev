// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/array_samplers.h>
#include <jet/parallel.h>
#include <jet/vertex_centered_vector_grid.h>

namespace jet {

template <size_t N>
VertexCenteredVectorGrid<N>::VertexCenteredVectorGrid() {}

template <size_t N>
VertexCenteredVectorGrid<N>::VertexCenteredVectorGrid(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &origin, const Vector<double, N> &initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

template <size_t N>
VertexCenteredVectorGrid<N>::VertexCenteredVectorGrid(
    const VertexCenteredVectorGrid &other) {
    set(other);
}

template <size_t N>
Vector<size_t, N> VertexCenteredVectorGrid<N>::dataSize() const {
    if (resolution() != Vector<size_t, N>()) {
        return resolution() + Vector<size_t, N>::makeConstant(1);
    } else {
        return Vector<size_t, N>();
    }
}

template <size_t N>
Vector<double, N> VertexCenteredVectorGrid<N>::dataOrigin() const {
    return origin();
}

template <size_t N>
void VertexCenteredVectorGrid<N>::swap(Grid<N> *other) {
    VertexCenteredVectorGrid *sameType =
        dynamic_cast<VertexCenteredVectorGrid *>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

template <size_t N>
void VertexCenteredVectorGrid<N>::set(const VertexCenteredVectorGrid &other) {
    setCollocatedVectorGrid(other);
}

template <size_t N>
VertexCenteredVectorGrid<N> &VertexCenteredVectorGrid<N>::operator=(
    const VertexCenteredVectorGrid &other) {
    set(other);
    return *this;
}

template <size_t N>
void VertexCenteredVectorGrid<N>::fill(const Vector<double, N> &value,
                                       ExecutionPolicy policy) {
    Vector<size_t, N> size = dataSize();
    auto view = dataView();
    parallelForEachIndex(
        Vector<size_t, N>(), size,
        [&view, &value](auto... indices) { view(indices...) = value; }, policy);
}

template <size_t N>
void VertexCenteredVectorGrid<N>::fill(
    const std::function<Vector<double, N>(const Vector<double, N> &)> &func,
    ExecutionPolicy policy) {
    Vector<size_t, N> size = dataSize();
    auto view = dataView();
    DataPositionFunc pos = dataPosition();
    parallelForEachIndex(Vector<size_t, N>::makeZero(), size,
                         [&func, &view, &pos](auto... indices) {
                             view(indices...) =
                                 func(pos(Vector<size_t, N>(indices...)));
                         },
                         policy);
}

template <size_t N>
std::shared_ptr<VectorGrid<N>> VertexCenteredVectorGrid<N>::clone() const {
    return CLONE_W_CUSTOM_DELETER(VertexCenteredVectorGrid);
}

template <size_t N>
typename VertexCenteredVectorGrid<N>::Builder
VertexCenteredVectorGrid<N>::builder() {
    return Builder();
}

template <size_t N>
typename VertexCenteredVectorGrid<N>::Builder &
VertexCenteredVectorGrid<N>::Builder::withResolution(
    const Vector<size_t, N> &resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
typename VertexCenteredVectorGrid<N>::Builder &
VertexCenteredVectorGrid<N>::Builder::withGridSpacing(
    const Vector<double, N> &gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

template <size_t N>
typename VertexCenteredVectorGrid<N>::Builder &
VertexCenteredVectorGrid<N>::Builder::withOrigin(
    const Vector<double, N> &gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

template <size_t N>
typename VertexCenteredVectorGrid<N>::Builder &
VertexCenteredVectorGrid<N>::Builder::withInitialValue(
    const Vector<double, N> &initialVal) {
    _initialVal = initialVal;
    return *this;
}

template <size_t N>
VertexCenteredVectorGrid<N> VertexCenteredVectorGrid<N>::Builder::build()
    const {
    return VertexCenteredVectorGrid(_resolution, _gridSpacing, _gridOrigin,
                                    _initialVal);
}

template <size_t N>
std::shared_ptr<VertexCenteredVectorGrid<N>>
VertexCenteredVectorGrid<N>::Builder::makeShared() const {
    return std::shared_ptr<VertexCenteredVectorGrid>(
        new VertexCenteredVectorGrid(_resolution, _gridSpacing, _gridOrigin,
                                     _initialVal),
        [](VertexCenteredVectorGrid *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<VectorGrid<N>> VertexCenteredVectorGrid<N>::Builder::build(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &gridOrigin,
    const Vector<double, N> &initialVal) const {
    return std::shared_ptr<VertexCenteredVectorGrid>(
        new VertexCenteredVectorGrid(resolution, gridSpacing, gridOrigin,
                                     initialVal),
        [](VertexCenteredVectorGrid *obj) { delete obj; });
}

template class VertexCenteredVectorGrid<2>;

template class VertexCenteredVectorGrid<3>;

}  // namespace jet
