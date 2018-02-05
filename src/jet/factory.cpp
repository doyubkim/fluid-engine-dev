// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <factory.h>

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/cell_centered_vector_grid3.h>
#include <jet/face_centered_grid2.h>
#include <jet/face_centered_grid3.h>
#include <jet/point_hash_grid_searcher2.h>
#include <jet/point_hash_grid_searcher3.h>
#include <jet/point_kdtree_searcher2.h>
#include <jet/point_kdtree_searcher3.h>
#include <jet/point_parallel_hash_grid_searcher2.h>
#include <jet/point_parallel_hash_grid_searcher3.h>
#include <jet/point_simple_list_searcher2.h>
#include <jet/point_simple_list_searcher3.h>
#include <jet/vertex_centered_scalar_grid2.h>
#include <jet/vertex_centered_scalar_grid3.h>
#include <jet/vertex_centered_vector_grid2.h>
#include <jet/vertex_centered_vector_grid3.h>

#include <string>
#include <unordered_map>

using namespace jet;

namespace {

std::unordered_map<std::string, ScalarGridBuilder2Ptr> sScalarGrid2Builders;
std::unordered_map<std::string, ScalarGridBuilder3Ptr> sScalarGrid3Builders;
std::unordered_map<std::string, VectorGridBuilder2Ptr> sVectorGrid2Builders;
std::unordered_map<std::string, VectorGridBuilder3Ptr> sVectorGrid3Builders;

std::unordered_map<std::string, PointNeighborSearcherBuilder2Ptr>
    sPointNeighborSearcher2Builders;
std::unordered_map<std::string, PointNeighborSearcherBuilder3Ptr>
    sPointNeighborSearcher3Builders;

}  // namespace

#define REGISTER_BUILDER(map, ClassName) \
    map.emplace(#ClassName, std::make_shared<ClassName::Builder>());

#define REGISTER_SCALAR_GRID2_BUILDER(ClassName) \
    REGISTER_BUILDER(sScalarGrid2Builders, ClassName)

#define REGISTER_SCALAR_GRID3_BUILDER(ClassName) \
    REGISTER_BUILDER(sScalarGrid3Builders, ClassName)

#define REGISTER_VECTOR_GRID2_BUILDER(ClassName) \
    REGISTER_BUILDER(sVectorGrid2Builders, ClassName)

#define REGISTER_VECTOR_GRID3_BUILDER(ClassName) \
    REGISTER_BUILDER(sVectorGrid3Builders, ClassName)

#define REGISTER_POINT_NEIGHBOR_SEARCHER2_BUILDER(ClassName) \
    REGISTER_BUILDER(sPointNeighborSearcher2Builders, ClassName)

#define REGISTER_POINT_NEIGHBOR_SEARCHER3_BUILDER(ClassName) \
    REGISTER_BUILDER(sPointNeighborSearcher3Builders, ClassName)

class Registry {
 public:
    Registry() {
        REGISTER_SCALAR_GRID2_BUILDER(CellCenteredScalarGrid2)
        REGISTER_SCALAR_GRID2_BUILDER(VertexCenteredScalarGrid2)

        REGISTER_SCALAR_GRID3_BUILDER(CellCenteredScalarGrid3)
        REGISTER_SCALAR_GRID3_BUILDER(VertexCenteredScalarGrid3)

        REGISTER_VECTOR_GRID2_BUILDER(CellCenteredVectorGrid2)
        REGISTER_VECTOR_GRID2_BUILDER(FaceCenteredGrid2)
        REGISTER_VECTOR_GRID2_BUILDER(VertexCenteredVectorGrid2)

        REGISTER_VECTOR_GRID3_BUILDER(CellCenteredVectorGrid3)
        REGISTER_VECTOR_GRID3_BUILDER(FaceCenteredGrid3)
        REGISTER_VECTOR_GRID3_BUILDER(VertexCenteredVectorGrid3)

        REGISTER_POINT_NEIGHBOR_SEARCHER2_BUILDER(PointHashGridSearcher2)
        REGISTER_POINT_NEIGHBOR_SEARCHER2_BUILDER(
            PointParallelHashGridSearcher2)
        REGISTER_POINT_NEIGHBOR_SEARCHER2_BUILDER(PointSimpleListSearcher2)
        REGISTER_POINT_NEIGHBOR_SEARCHER2_BUILDER(PointKdTreeSearcher2)

        REGISTER_POINT_NEIGHBOR_SEARCHER3_BUILDER(PointHashGridSearcher3)
        REGISTER_POINT_NEIGHBOR_SEARCHER3_BUILDER(
            PointParallelHashGridSearcher3)
        REGISTER_POINT_NEIGHBOR_SEARCHER3_BUILDER(PointSimpleListSearcher3)
        REGISTER_POINT_NEIGHBOR_SEARCHER3_BUILDER(PointKdTreeSearcher3)
    }
};

static Registry sRegistry;

ScalarGrid2Ptr Factory::buildScalarGrid2(const std::string& name) {
    auto result = sScalarGrid2Builders.find(name);
    if (result != sScalarGrid2Builders.end()) {
        auto builder = result->second;
        return builder->build({0, 0}, {1, 1}, {0, 0}, 0.0);
    } else {
        return nullptr;
    }
}

ScalarGrid3Ptr Factory::buildScalarGrid3(const std::string& name) {
    auto result = sScalarGrid3Builders.find(name);
    if (result != sScalarGrid3Builders.end()) {
        auto builder = result->second;
        return builder->build({0, 0, 0}, {1, 1, 1}, {0, 0, 0}, 0.0);
    } else {
        return nullptr;
    }
}

VectorGrid2Ptr Factory::buildVectorGrid2(const std::string& name) {
    auto result = sVectorGrid2Builders.find(name);
    if (result != sVectorGrid2Builders.end()) {
        auto builder = result->second;
        return builder->build({0, 0}, {1, 1}, {0, 0}, {0, 0});
    } else {
        return nullptr;
    }
}

VectorGrid3Ptr Factory::buildVectorGrid3(const std::string& name) {
    auto result = sVectorGrid3Builders.find(name);
    if (result != sVectorGrid3Builders.end()) {
        auto builder = result->second;
        return builder->build({0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0});
    } else {
        return nullptr;
    }
}

PointNeighborSearcher2Ptr Factory::buildPointNeighborSearcher2(
    const std::string& name) {
    auto result = sPointNeighborSearcher2Builders.find(name);
    if (result != sPointNeighborSearcher2Builders.end()) {
        auto builder = result->second;
        return builder->buildPointNeighborSearcher();
    } else {
        return nullptr;
    }
}

PointNeighborSearcher3Ptr Factory::buildPointNeighborSearcher3(
    const std::string& name) {
    auto result = sPointNeighborSearcher3Builders.find(name);
    if (result != sPointNeighborSearcher3Builders.end()) {
        auto builder = result->second;
        return builder->buildPointNeighborSearcher();
    } else {
        return nullptr;
    }
}
