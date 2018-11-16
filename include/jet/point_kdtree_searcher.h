// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_KDTREE_SEARCHER_H
#define INCLUDE_JET_POINT_KDTREE_SEARCHER_H

#include <jet/kdtree.h>
#include <jet/matrix.h>
#include <jet/point_neighbor_searcher.h>

#include <vector>

namespace jet {

//!
//! \brief KdTree-based N-D point searcher.
//!
//! This class implements N-D point searcher by using KdTree for its internal
//! acceleration data structure.
//!
template <size_t N>
class PointKdTreeSearcher final : public PointNeighborSearcher<N> {
 public:
    JET_NEIGHBOR_SEARCHER_TYPE_NAME(PointKdTreeSearcher, N)

    class Builder;

    using typename PointNeighborSearcher<N>::ForEachNearbyPointFunc;

    //! Constructs an empty kD-tree instance.
    PointKdTreeSearcher();

    //! Copy constructor.
    PointKdTreeSearcher(const PointKdTreeSearcher& other);

    //! Builds internal acceleration structure for given points list.
    void build(const ConstArrayView1<Vector<double, N>>& points) override;

    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    void forEachNearbyPoint(
        const Vector<double, N>& origin, double radius,
        const ForEachNearbyPointFunc& callback) const override;

    //!
    //! Returns true if there are any nearby points for given origin within
    //! radius.
    //!
    //! \param[in]  origin The origin.
    //! \param[in]  radius The radius.
    //!
    //! \return     True if has nearby point, false otherwise.
    //!
    bool hasNearbyPoint(const Vector<double, N>& origin,
                        double radius) const override;

    //!
    //! \brief      Creates a new instance of the object with same properties
    //!             than original.
    //!
    //! \return     Copy of this object.
    //!
    std::shared_ptr<PointNeighborSearcher<N>> clone() const override;

    //! Assignment operator.
    PointKdTreeSearcher& operator=(const PointKdTreeSearcher& other);

    //! Copy from the other instance.
    void set(const PointKdTreeSearcher& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointKdTreeSearcher.
    static Builder builder();

 private:
    KdTree<double, N> _tree;

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> serialize(
        const PointKdTreeSearcher<2>& searcher, std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> serialize(
        const PointKdTreeSearcher<3>& searcher, std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> deserialize(
        const std::vector<uint8_t>& buffer, PointKdTreeSearcher<2>& searcher);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> deserialize(
        const std::vector<uint8_t>& buffer, PointKdTreeSearcher<3>& searcher);
};

//! 2-D PointKdTreeSearcher type.
using PointKdTreeSearcher2 = PointKdTreeSearcher<2>;

//! 3-D PointKdTreeSearcher type.
using PointKdTreeSearcher3 = PointKdTreeSearcher<3>;

//! Shared pointer for the PointKdTreeSearcher2 type.
using PointKdTreeSearcher2Ptr = std::shared_ptr<PointKdTreeSearcher2>;

//! Shared pointer for the PointKdTreeSearcher3 type.
using PointKdTreeSearcher3Ptr = std::shared_ptr<PointKdTreeSearcher3>;

//!
//! \brief Front-end to create PointKdTreeSearcher objects step by step.
//!
template <size_t N>
class PointKdTreeSearcher<N>::Builder final
    : public PointNeighborSearcherBuilder<N> {
 public:
    //! Builds PointKdTreeSearcher instance.
    PointKdTreeSearcher build() const;

    //! Builds shared pointer of PointKdTreeSearcher instance.
    std::shared_ptr<PointKdTreeSearcher<N>> makeShared() const;

    //! Returns shared pointer of PointNeighborSearcher3 type.
    std::shared_ptr<PointNeighborSearcher<N>> buildPointNeighborSearcher()
        const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_KDTREE_SEARCHER_H
