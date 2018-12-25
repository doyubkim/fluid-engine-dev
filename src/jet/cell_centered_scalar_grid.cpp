// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/cell_centered_scalar_grid.h>

namespace jet {

template <size_t N>
CellCenteredScalarGrid<N>::CellCenteredScalarGrid() {}

template <size_t N>
CellCenteredScalarGrid<N>::CellCenteredScalarGrid(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &origin, double initialValue) {
    resize(resolution, gridSpacing, origin, initialValue);
}

template <size_t N>
CellCenteredScalarGrid<N>::CellCenteredScalarGrid(
    const CellCenteredScalarGrid &other) {
    set(other);
}

template <size_t N>
Vector<size_t, N> CellCenteredScalarGrid<N>::dataSize() const {
    // The size of the data should be the same as the grid resolution.
    return resolution();
}

template <size_t N>
Vector<double, N> CellCenteredScalarGrid<N>::dataOrigin() const {
    return origin() + 0.5 * gridSpacing();
}

template <size_t N>
std::shared_ptr<ScalarGrid<N>> CellCenteredScalarGrid<N>::clone() const {
    return CLONE_W_CUSTOM_DELETER(CellCenteredScalarGrid<N>);
}

template <size_t N>
void CellCenteredScalarGrid<N>::swap(Grid<N> *other) {
    CellCenteredScalarGrid<N> *sameType =
        dynamic_cast<CellCenteredScalarGrid<N> *>(other);
    if (sameType != nullptr) {
        swapScalarGrid(sameType);
    }
}

template <size_t N>
void CellCenteredScalarGrid<N>::set(const CellCenteredScalarGrid &other) {
    setScalarGrid(other);
}

template <size_t N>
CellCenteredScalarGrid<N> &CellCenteredScalarGrid<N>::operator=(
    const CellCenteredScalarGrid &other) {
    set(other);
    return *this;
}

template <size_t N>
typename CellCenteredScalarGrid<N>::Builder
CellCenteredScalarGrid<N>::builder() {
    return Builder();
}

template <size_t N>
typename CellCenteredScalarGrid<N>::Builder &
CellCenteredScalarGrid<N>::Builder::withResolution(
    const Vector<size_t, N> &resolution) {
    _resolution = resolution;
    return *this;
}

template <size_t N>
typename CellCenteredScalarGrid<N>::Builder &
CellCenteredScalarGrid<N>::Builder::withGridSpacing(
    const Vector<double, N> &gridSpacing) {
    _gridSpacing = gridSpacing;
    return *this;
}

template <size_t N>
typename CellCenteredScalarGrid<N>::Builder &
CellCenteredScalarGrid<N>::Builder::withOrigin(
    const Vector<double, N> &gridOrigin) {
    _gridOrigin = gridOrigin;
    return *this;
}

template <size_t N>
typename CellCenteredScalarGrid<N>::Builder &
CellCenteredScalarGrid<N>::Builder::withInitialValue(double initialVal) {
    _initialVal = initialVal;
    return *this;
}

template <size_t N>
CellCenteredScalarGrid<N> CellCenteredScalarGrid<N>::Builder::build() const {
    return CellCenteredScalarGrid(_resolution, _gridSpacing, _gridOrigin,
                                  _initialVal);
}

template <size_t N>
std::shared_ptr<ScalarGrid<N>> CellCenteredScalarGrid<N>::Builder::build(
    const Vector<size_t, N> &resolution, const Vector<double, N> &gridSpacing,
    const Vector<double, N> &gridOrigin, double initialVal) const {
    return std::shared_ptr<CellCenteredScalarGrid>(
        new CellCenteredScalarGrid(resolution, gridSpacing, gridOrigin,
                                   initialVal),
        [](CellCenteredScalarGrid *obj) { delete obj; });
}

template <size_t N>
std::shared_ptr<CellCenteredScalarGrid<N>>
CellCenteredScalarGrid<N>::Builder::makeShared() const {
    return std::shared_ptr<CellCenteredScalarGrid>(
        new CellCenteredScalarGrid(_resolution, _gridSpacing, _gridOrigin,
                                   _initialVal),
        [](CellCenteredScalarGrid *obj) { delete obj; });
}

template class CellCenteredScalarGrid<2>;

template class CellCenteredScalarGrid<3>;

}  // namespace jet
