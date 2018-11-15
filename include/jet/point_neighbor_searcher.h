// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_POINT_NEIGHBOR_SEARCHER_H_
#define INCLUDE_JET_POINT_NEIGHBOR_SEARCHER_H_

#include <jet/array_view.h>
#include <jet/matrix.h>
#include <jet/serialization.h>

#include <functional>
#include <memory>
#include <string>

namespace jet {

//!
//! \brief Abstract base class for N-D neighbor point searcher.
//!
//! This class provides interface for N-D neighbor point searcher. For given
//! list of points, the class builds internal cache to accelerate the search.
//! Once built, the data structure is used to search nearby points for given
//! origin point.
//!
template <size_t N>
class PointNeighborSearcher : public Serializable {
 public:
    //! Callback function for nearby search query. The first parameter is the
    //! index of the nearby point, and the second is the position of the point.
    using ForEachNearbyPointFunc = std::function<void(size_t, const Vector<double, N>&)>;

    //! Default constructor.
    PointNeighborSearcher();

    //! Destructor.
    virtual ~PointNeighborSearcher();

    //! Returns the type name of the derived class.
    virtual std::string typeName() const = 0;

    //! Builds internal acceleration structure for given points list.
    virtual void build(const ConstArrayView1<Vector<double, N>>& points) = 0;

    //!
    //! Invokes the callback function for each nearby point around the origin
    //! within given radius.
    //!
    //! \param[in]  origin   The origin position.
    //! \param[in]  radius   The search radius.
    //! \param[in]  callback The callback function.
    //!
    virtual void forEachNearbyPoint(
        const Vector<double, N>& origin, double radius,
        const ForEachNearbyPointFunc& callback) const = 0;

    //!
    //! Returns true if there are any nearby points for given origin within
    //! radius.
    //!
    //! \param[in]  origin The origin.
    //! \param[in]  radius The radius.
    //!
    //! \return     True if has nearby point, false otherwise.
    //!
    virtual bool hasNearbyPoint(const Vector<double, N>& origin,
                                double radius) const = 0;

    //!
    //! \brief      Creates a new instance of the object with same properties
    //!             than original.
    //!
    //! \return     Copy of this object.
    //!
    virtual std::shared_ptr<PointNeighborSearcher> clone() const = 0;
};

//! 2-D PointNeighborSearcher type.
using PointNeighborSearcher2 = PointNeighborSearcher<2>;

//! 3-D PointNeighborSearcher type.
using PointNeighborSearcher3 = PointNeighborSearcher<3>;

//! Shared pointer for the PointNeighborSearcher2 type.
using PointNeighborSearcher2Ptr = std::shared_ptr<PointNeighborSearcher2>;

//! Shared pointer for the PointNeighborSearcher3 type.
using PointNeighborSearcher3Ptr = std::shared_ptr<PointNeighborSearcher3>;

//! Abstract base class for N-D point neighbor searcher builders.
template <size_t N>
class PointNeighborSearcherBuilder {
 public:
    //! Returns shared pointer of PointNeighborSearcher type.
    virtual std::shared_ptr<PointNeighborSearcher<N>>
    buildPointNeighborSearcher() const = 0;
};

//! 2-D PointNeighborSearcherBuilder type.
using PointNeighborSearcherBuilder2 = PointNeighborSearcherBuilder<2>;

//! 3-D PointNeighborSearcherBuilder type.
using PointNeighborSearcherBuilder3 = PointNeighborSearcherBuilder<3>;

//! Shared pointer for the PointNeighborSearcher2 type.
using PointNeighborSearcherBuilder2Ptr =
    std::shared_ptr<PointNeighborSearcherBuilder2>;

//! Shared pointer for the PointNeighborSearcher3 type.
using PointNeighborSearcherBuilder3Ptr =
    std::shared_ptr<PointNeighborSearcherBuilder3>;

#define JET_NEIGHBOR_SEARCHER_TYPE_NAME(DerivedClassName, N) \
    std::string typeName() const override { return #DerivedClassName + std::to_string(N); }

}  // namespace jet

#endif  // INCLUDE_JET_POINT_NEIGHBOR_SEARCHER_H_
