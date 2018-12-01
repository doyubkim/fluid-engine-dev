// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/vertex_centered_scalar_grid.h>

namespace jet {

template <size_t N>
VertexCenteredScalarGrid<N>::VertexCenteredScalarGrid() {}

template <size_t N>
VertexCenteredScalarGrid<N>::VertexCenteredScalarGrid(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &origin, double initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

template <size_t N>
VertexCenteredScalarGrid<N>::VertexCenteredScalarGrid(
    const VertexCenteredScalarGrid &other) {
    set(other);
}

template <size_t N>
Vector<size_t, N> VertexCenteredScalarGrid<N>::dataSize() const {
    if (resolution() != Vector<size_t, N>()) {
        return resolution() + Vector<size_t, N>::makeConstant(1);
    } else {
        return Vector<size_t, N>();
    }
}

template <size_t N>
Vector<double, N> VertexCenteredScalarGrid<N>::dataOrigin() const {
    return origin();
}

template <size_t N>
std::shared_ptr<ScalarGrid<N>> VertexCenteredScalarGrid<N>::clone() const {
    return CLONE_W_CUSTOM_DELETER(VertexCenteredScalarGrid);
}

template <size_t N>
void VertexCenteredScalarGrid<N>::swap(Grid<N> *other) {
    VertexCenteredScalarGrid *sameType =
        dynamic_cast<VertexCenteredScalarGrid *>(other);
    if (sameType != nullptr) {
        swapScalarGrid(sameType);
    }
}

template <size_t N>
void VertexCenteredScalarGrid<N>::set(const VertexCenteredScalarGrid &other) {
    setScalarGrid(other);
}

template <size_t N>
VertexCenteredScalarGrid<N> &VertexCenteredScalarGrid<N>::operator=(
    const VertexCenteredScalarGrid &other) {
    set(other);
    return *this;
}

template <size_t N>
typename VertexCenteredScalarGrid<N>::Builder
VertexCenteredScalarGrid<N>::builder() {
    return Builder();
}

template <size_t N>
typename VertexCenteredScalarGrid<N>::Builder &
VertexCenteredScalarGrid<N>::Builder::withResolution(
    const Vector<size_t, N> &resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
typename VertexCenteredScalarGrid<N>::Builder &
VertexCenteredScalarGrid<N>::Builder::withGridSpacing(
    const Vector<double, N> &gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

template <size_t N>
typename VertexCenteredScalarGrid<N>::Builder &
VertexCenteredScalarGrid<N>::Builder::withOrigin(
    const Vector<double, N> &gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

template <size_t N>
typename VertexCenteredScalarGrid<N>::Builder &
VertexCenteredScalarGrid<N>::Builder::withInitialValue(double initialVal) {
    _initialVal = initialVal;
    return *this;
}

template <size_t N>
VertexCenteredScalarGrid<N> VertexCenteredScalarGrid<N>::Builder::build()
    const {
    return VertexCenteredScalarGrid(_resolution, _gridSpacing, _gridOrigin,
                                    _initialVal);
}

template <size_t N>
std::shared_ptr<VertexCenteredScalarGrid<N>>
VertexCenteredScalarGrid<N>::Builder::makeShared() const {
    return std::shared_ptr<VertexCenteredScalarGrid>(
        new VertexCenteredScalarGrid(_resolution, _gridSpacing, _gridOrigin,
                                     _initialVal),
        [](VertexCenteredScalarGrid *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<ScalarGrid<N>> VertexCenteredScalarGrid<N>::Builder::build(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &gridOrigin, double initialVal) const {
    return std::shared_ptr<VertexCenteredScalarGrid>(
        new VertexCenteredScalarGrid(resolution, gridSpacing, gridOrigin,
                                     initialVal),
        [](VertexCenteredScalarGrid *obj) { delete obj; });
}

template class VertexCenteredScalarGrid<2>;

template class VertexCenteredScalarGrid<3>;

}  // namespace jet
