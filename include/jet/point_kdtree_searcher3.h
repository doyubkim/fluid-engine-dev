// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_KDTREE_SEARCHER3_H
#define INCLUDE_JET_POINT_KDTREE_SEARCHER3_H

#include <jet/kdtree.h>
#include <jet/point3.h>
#include <jet/point_neighbor_searcher3.h>
#include <jet/size3.h>

#include <vector>

namespace jet {

//!
//! \brief KdTree-based 3-D point searcher.
//!
//! This class implements 3-D point searcher by using KdTree for its internal
//! acceleration data structure.
//!
class PointKdTreeSearcher3 final : public PointNeighborSearcher3 {
 public:
    JET_NEIGHBOR_SEARCHER3_TYPE_NAME(PointKdTreeSearcher3)

    class Builder;

    //! Constructs an empty kD-tree instance.
    PointKdTreeSearcher3();

    //! Copy constructor.
    PointKdTreeSearcher3(const PointKdTreeSearcher3& other);

    //! Builds internal acceleration structure for given points list.
    void build(const ConstArrayAccessor1<Vector3D>& points) override;

    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    void forEachNearbyPoint(
        const Vector3D& origin, double radius,
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
    bool hasNearbyPoint(const Vector3D& origin, double radius) const override;

    //!
    //! \brief      Creates a new instance of the object with same properties
    //!             than original.
    //!
    //! \return     Copy of this object.
    //!
    PointNeighborSearcher3Ptr clone() const override;

    //! Assignment operator.
    PointKdTreeSearcher3& operator=(const PointKdTreeSearcher3& other);

    //! Copy from the other instance.
    void set(const PointKdTreeSearcher3& other);

    //! Serializes the neighbor searcher into the buffer.
    void serialize(std::vector<uint8_t>* buffer) const override;

    //! Deserializes the neighbor searcher from the buffer.
    void deserialize(const std::vector<uint8_t>& buffer) override;

    //! Returns builder fox PointKdTreeSearcher3.
    static Builder builder();

 private:
    KdTree<double, 3> _tree;
};

//! Shared pointer for the PointKdTreeSearcher3 type.
typedef std::shared_ptr<PointKdTreeSearcher3> PointKdTreeSearcher3Ptr;

//!
//! \brief Front-end to create PointKdTreeSearcher3 objects step by step.
//!
class PointKdTreeSearcher3::Builder final
    : public PointNeighborSearcherBuilder3 {
 public:
    //! Builds PointKdTreeSearcher3 instance.
    PointKdTreeSearcher3 build() const;

    //! Builds shared pointer of PointKdTreeSearcher3 instance.
    PointKdTreeSearcher3Ptr makeShared() const;

    //! Returns shared pointer of PointNeighborSearcher3 type.
    PointNeighborSearcher3Ptr buildPointNeighborSearcher() const override;
};

}  // namespace jet

#endif  // INCLUDE_JET_POINT_KDTREE_SEARCHER3_H
