// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_FACTORY_H_
#define SRC_JET_FACTORY_H_

#include <jet/scalar_grid2.h>
#include <jet/scalar_grid3.h>
#include <jet/vector_grid2.h>
#include <jet/vector_grid3.h>
#include <jet/point_neighbor_searcher2.h>
#include <jet/point_neighbor_searcher3.h>
#include <string>

namespace jet {

class Factory {
 public:
    static ScalarGrid2Ptr buildScalarGrid2(const std::string& name);

    static ScalarGrid3Ptr buildScalarGrid3(const std::string& name);

    static VectorGrid2Ptr buildVectorGrid2(const std::string& name);

    static VectorGrid3Ptr buildVectorGrid3(const std::string& name);

    static PointNeighborSearcher2Ptr buildPointNeighborSearcher2(
        const std::string& name);

    static PointNeighborSearcher3Ptr buildPointNeighborSearcher3(
        const std::string& name);
};

}  // namespace jet

#endif  // SRC_JET_FACTORY_H_
