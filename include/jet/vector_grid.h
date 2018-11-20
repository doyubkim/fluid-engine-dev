// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_VECTOR_GRID_H_
#define INCLUDE_JET_VECTOR_GRID_H_

#include <jet/array_view.h>
#include <jet/grid.h>
#include <jet/parallel.h>
#include <jet/vector_field.h>

namespace jet {

//! Abstract base class for N-D vector grid structure.
template <size_t N>
class VectorGrid : public VectorField<N>, public Grid<N> {
 public:
    //! Read-write array view type.
    using VectorDataView = ArrayView<Vector<double, N>, N>;

    //! Read-only array view type.
    using ConstVectorDataView = ArrayView<const Vector<double, N>, N>;

    using Grid<N>::resolution;
    using Grid<N>::gridSpacing;
    using Grid<N>::origin;

    //! Constructs an empty grid.
    VectorGrid();

    //! Default destructor.
    virtual ~VectorGrid();

    //! Clears the contents of the grid.
    void clear();

    //! Resizes the grid using given parameters.
    void resize(const Vector<size_t, N>& resolution,
                const Vector<double, N>& gridSpacing = Vector<double, N>(1, 1),
                const Vector<double, N>& origin = Vector<double, N>(),
                const Vector<double, N>& initialValue = Vector<double, N>());

    //! Resizes the grid using given parameters.
    void resize(const Vector<double, N>& gridSpacing,
                const Vector<double, N>& origin);

    //! Fills the grid with given value.
    virtual void fill(const Vector<double, N>& value,
                      ExecutionPolicy policy = ExecutionPolicy::kParallel) = 0;

    //! Fills the grid with given position-to-value mapping function.
    virtual void fill(
        const std::function<Vector<double, N>(const Vector<double, N>&)>& func,
        ExecutionPolicy policy = ExecutionPolicy::kParallel) = 0;

    //! Returns the copy of the grid instance.
    virtual std::shared_ptr<VectorGrid<N>> clone() const = 0;

    //! Serializes the grid instance to the output buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the input buffer to the grid instance.
    void deserialize(const std::vector<uint8_t>& buffer) override;

 protected:
    using Grid<N>::setSizeParameters;
    using Grid<N>::getData;
    using Grid<N>::setData;

    //!
    //! \brief Invoked when the resizing happens.
    //!
    //! This callback function is called when the grid gets resized. The
    //! overriding class should allocate the internal storage based on its
    //! data layout scheme.
    //!
    virtual void onResize(const Vector<size_t, N>& resolution,
                          const Vector<double, N>& gridSpacing,
                          const Vector<double, N>& origin,
                          const Vector<double, N>& initialValue) = 0;
};

//! 2-D VectorGrid type.
using VectorGrid2 = VectorGrid<2>;

//! 3-D VectorGrid type.
using VectorGrid3 = VectorGrid<3>;

//! Shared pointer for the VectorGrid2 type.
using VectorGrid2Ptr = std::shared_ptr<VectorGrid2>;

//! Shared pointer for the VectorGrid3 type.
using VectorGrid3Ptr = std::shared_ptr<VectorGrid3>;

//! Abstract base class for N-D vector grid builder.
template <size_t N>
class VectorGridBuilder {
 public:
    //! Creates a builder.
    VectorGridBuilder();

    //! Default destructor.
    virtual ~VectorGridBuilder();

    //! Returns N-D vector grid with given parameters.
    virtual std::shared_ptr<VectorGrid<N>> build(
        const Vector<size_t, N>& resolution,
        const Vector<double, N>& gridSpacing,
        const Vector<double, N>& gridOrigin,
        const Vector<double, N>& initialVal) const = 0;
};

//! 2-D VectorGridBuilder type.
using VectorGridBuilder2 = VectorGridBuilder<2>;

//! 3-D VectorGridBuilder type.
using VectorGridBuilder3 = VectorGridBuilder<3>;

//! Shared pointer for the VectorGridBuilder2 type.
using VectorGridBuilder2Ptr = std::shared_ptr<VectorGridBuilder2>;

//! Shared pointer for the VectorGridBuilder3 type.
using VectorGridBuilder3Ptr = std::shared_ptr<VectorGridBuilder3>;

}  // namespace jet

#endif  // INCLUDE_JET_VECTOR_GRID_H_
