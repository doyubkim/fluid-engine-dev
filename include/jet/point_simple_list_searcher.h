// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER_H_
#define INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER_H_

#include <jet/array.h>
#include <jet/point_neighbor_searcher.h>

namespace jet {

//!
//! \brief Simple ad-hoc N-D point searcher.
//!
//! This class implements N-D point searcher simply by looking up every point in
//! the list. Thus, this class is not ideal for searches involing large number
//! of points, but only for small set of items.
//!
template <size_t N>
class PointSimpleListSearcher final : public PointNeighborSearcher<N> {
 public:
    JET_NEIGHBOR_SEARCHER_TYPE_NAME(PointSimpleListSearcher, N)

    class Builder;

    using typename PointNeighborSearcher<N>::ForEachNearbyPointFunc;
    using PointNeighborSearcher<N>::build;

    //! Default constructor.
    PointSimpleListSearcher();

    //! Copy constructor.
    PointSimpleListSearcher(const PointSimpleListSearcher& other);

    //!
    //! \brief      Builds internal structure for given points list and max search radius.
    //!
    //! For this class, this function simply copies the given point list to the
    //! internal list. The max search radius will be unused.
    //!
    //! \param[in]  points          The points to search.
    //! \param[in]  maxSearchRadius Max search radius (ignored).
    //!
    void build(const ConstArrayView1<Vector<double, N>>& points, double maxSearchRadius) override;

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
    PointSimpleListSearcher& operator=(const PointSimpleListSearcher& other);

    //! Copy from the other instance.
    void set(const PointSimpleListSearcher& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointSimpleListSearcher.
    static Builder builder();

 private:
    Array1<Vector<double, N>> _points;

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> serialize(
        const PointSimpleListSearcher<2>& searcher,
        std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> serialize(
        const PointSimpleListSearcher<3>& searcher,
        std::vector<uint8_t>* buffer);

    template <size_t M = N>
    static std::enable_if_t<M == 2, void> deserialize(
        const std::vector<uint8_t>& buffer,
        PointSimpleListSearcher<2>& searcher);

    template <size_t M = N>
    static std::enable_if_t<M == 3, void> deserialize(
        const std::vector<uint8_t>& buffer,
        PointSimpleListSearcher<3>& searcher);
};

//! 2-D PointSimpleListSearcher type.
using PointSimpleListSearcher2 = PointSimpleListSearcher<2>;

//! 3-D PointSimpleListSearcher type.
using PointSimpleListSearcher3 = PointSimpleListSearcher<3>;

//! Shared pointer for the PointSimpleListSearcher2 type.
using PointSimpleListSearcher2Ptr = std::shared_ptr<PointSimpleListSearcher<2>>;

//! Shared pointer for the PointSimpleListSearcher3 type.
using PointSimpleListSearcher3Ptr = std::shared_ptr<PointSimpleListSearcher<3>>;

//!
//! \brief Front-end to create PointSimpleListSearcher objects step by step.
//!
template <size_t N>
class PointSimpleListSearcher<N>::Builder final
    : public PointNeighborSearcherBuilder<N> {
 public:
    //! Builds PointSimpleListSearcher instance.
    PointSimpleListSearcher<N> build() const;

    //! Builds shared pointer of PointSimpleListSearcher instance.
    std::shared_ptr<PointSimpleListSearcher<N>> makeShared() const;

    //! Returns shared pointer of PointNeighborSearcher type.
    std::shared_ptr<PointNeighborSearcher<N>> buildPointNeighborSearcher()
        const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER_H_
