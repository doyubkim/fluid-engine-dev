// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER2_H_
#define INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER2_H_

#include <jet/point_neighbor_searcher2.h>
#include <vector>

namespace jet {

//!
//! \brief Simple ad-hoc 2-D point searcher.
//!
//! This class implements 2-D point searcher simply by looking up every point in
//! the list. Thus, this class is not ideal for searches involing large number
//! of points, but only for small set of items.
//!
class PointSimpleListSearcher2 final : public PointNeighborSearcher2 {
 public:
    JET_NEIGHBOR_SEARCHER2_TYPE_NAME(PointSimpleListSearcher2)

    class Builder;

    //! Default constructor.
    PointSimpleListSearcher2();

    //! Copy constructor.
    PointSimpleListSearcher2(const PointSimpleListSearcher2& other);

    //!
    //! \brief      Builds internal structure for given points list.
    //!
    //! For this class, this function simply copies the given point list to the
    //! internal list.
    //!
    //! \param[in]  points The points to search.
    //!
    void build(const ConstArrayAccessor1<Vector2D>& points) override;

    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    void forEachNearbyPoint(
        const Vector2D& origin,
        double radius,
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
    bool hasNearbyPoint(
        const Vector2D& origin, double radius) const override;

    //!
    //! \brief      Creates a new instance of the object with same properties
    //!             than original.
    //!
    //! \return     Copy of this object.
    //!
    PointNeighborSearcher2Ptr clone() const override;

    //! Assignment operator.
    PointSimpleListSearcher2& operator=(const PointSimpleListSearcher2& other);

    //! Copy from the other instance.
    void set(const PointSimpleListSearcher2& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointSimpleListSearcher2.
    static Builder builder();

 private:
    std::vector<Vector2D> _points;
};

//! Shared pointer for the PointSimpleListSearcher2 type.
typedef std::shared_ptr<PointSimpleListSearcher2> PointSimpleListSearcher2Ptr;

//!
//! \brief Front-end to create PointSimpleListSearcher2 objects step by step.
//!
class PointSimpleListSearcher2::Builder final
    : public PointNeighborSearcherBuilder2 {
 public:
    //! Builds PointSimpleListSearcher2 instance.
    PointSimpleListSearcher2 build() const;

    //! Builds shared pointer of PointSimpleListSearcher2 instance.
    PointSimpleListSearcher2Ptr makeShared() const;

    //! Returns shared pointer of PointNeighborSearcher3 type.
    PointNeighborSearcher2Ptr buildPointNeighborSearcher() const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_SIMPLE_LIST_SEARCHER2_H_
