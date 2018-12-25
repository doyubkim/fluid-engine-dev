// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/cell_centered_vector_grid.h>
#include <jet/parallel.h>

namespace jet {

template <size_t N>
CellCenteredVectorGrid<N>::CellCenteredVectorGrid() {}

template <size_t N>
CellCenteredVectorGrid<N>::CellCenteredVectorGrid(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &origin, const Vector<double, N> &initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

template <size_t N>
CellCenteredVectorGrid<N>::CellCenteredVectorGrid(
    const CellCenteredVectorGrid &other) {
    set(other);
}

template <size_t N>
Vector<size_t, N> CellCenteredVectorGrid<N>::dataSize() const {
    return resolution();
}

template <size_t N>
Vector<double, N> CellCenteredVectorGrid<N>::dataOrigin() const {
    return origin() + 0.5 * gridSpacing();
}

template <size_t N>
void CellCenteredVectorGrid<N>::swap(Grid<N> *other) {
    CellCenteredVectorGrid *sameType =
        dynamic_cast<CellCenteredVectorGrid *>(other);
    if (sameType != nullptr) {
        swapCollocatedVectorGrid(sameType);
    }
}

template <size_t N>
void CellCenteredVectorGrid<N>::set(const CellCenteredVectorGrid &other) {
    setCollocatedVectorGrid(other);
}

template <size_t N>
CellCenteredVectorGrid<N> &CellCenteredVectorGrid<N>::operator=(
    const CellCenteredVectorGrid &other) {
    set(other);
    return *this;
}

template <size_t N>
void CellCenteredVectorGrid<N>::fill(const Vector<double, N> &value,
                                     ExecutionPolicy policy) {
    Vector<size_t, N> size = dataSize();
    auto view = dataView();
    parallelForEachIndex(
        Vector<size_t, N>(), size,
        [&view, &value](auto... indices) { view(indices...) = value; }, policy);
}

template <size_t N>
void CellCenteredVectorGrid<N>::fill(
    const std::function<Vector<double, N>(const Vector<double, N> &)> &func,
    ExecutionPolicy policy) {
    Vector<size_t, N> size = dataSize();
    auto view = dataView();
    auto pos = dataPosition();
    parallelForEachIndex(Vector<size_t, N>::makeZero(), size,
                         [&func, &view, &pos](auto... indices) {
                             view(indices...) = func(pos(indices...));
                         },
                         policy);
}

template <size_t N>
std::shared_ptr<VectorGrid<N>> CellCenteredVectorGrid<N>::clone() const {
    return CLONE_W_CUSTOM_DELETER(CellCenteredVectorGrid);
}

template <size_t N>
typename CellCenteredVectorGrid<N>::Builder
CellCenteredVectorGrid<N>::builder() {
    return Builder();
}

template <size_t N>
typename CellCenteredVectorGrid<N>::Builder &
CellCenteredVectorGrid<N>::Builder::withResolution(
    const Vector<size_t, N> &resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
typename CellCenteredVectorGrid<N>::Builder &
CellCenteredVectorGrid<N>::Builder::withGridSpacing(
    const Vector<double, N> &gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

template <size_t N>
typename CellCenteredVectorGrid<N>::Builder &
CellCenteredVectorGrid<N>::Builder::withOrigin(
    const Vector<double, N> &gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

template <size_t N>
typename CellCenteredVectorGrid<N>::Builder &
CellCenteredVectorGrid<N>::Builder::withInitialValue(
    const Vector<double, N> &initialVal) {
    _initialVal = initialVal;
    return *this;
}

template <size_t N>
CellCenteredVectorGrid<N> CellCenteredVectorGrid<N>::Builder::build() const {
    return CellCenteredVectorGrid(_resolution, _gridSpacing, _gridOrigin,
                                  _initialVal);
}

template <size_t N>
std::shared_ptr<VectorGrid<N>> CellCenteredVectorGrid<N>::Builder::build(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &gridOrigin,
    const Vector<double, N> &initialVal) const {
    return std::shared_ptr<CellCenteredVectorGrid>(
        new CellCenteredVectorGrid(resolution, gridSpacing, gridOrigin,
                                   initialVal),
        [](CellCenteredVectorGrid *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<CellCenteredVectorGrid<N>>
CellCenteredVectorGrid<N>::Builder::makeShared() const {
    return std::shared_ptr<CellCenteredVectorGrid>(
        new CellCenteredVectorGrid(_resolution, _gridSpacing, _gridOrigin,
                                   _initialVal),
        [](CellCenteredVectorGrid *obj) { delete obj; });
}

template class CellCenteredVectorGrid<2>;

template class CellCenteredVectorGrid<3>;

}  // namespace jet