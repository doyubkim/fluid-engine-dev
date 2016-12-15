// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <factory.h>
#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/cell_centered_vector_grid3.h>
#include <jet/face_centered_grid2.h>
#include <jet/face_centered_grid3.h>
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

}


#define REGISTER_GRID_BUILDER(map, ClassName) \
    map.emplace(#ClassName, std::make_shared<ClassName::Builder>());

#define REGISTER_SCALAR_GRID2_BUILDER(ClassName) \
    REGISTER_GRID_BUILDER(sScalarGrid2Builders, ClassName)

#define REGISTER_SCALAR_GRID3_BUILDER(ClassName) \
    REGISTER_GRID_BUILDER(sScalarGrid3Builders, ClassName)

#define REGISTER_VECTOR_GRID2_BUILDER(ClassName) \
    REGISTER_GRID_BUILDER(sVectorGrid2Builders, ClassName)

#define REGISTER_VECTOR_GRID3_BUILDER(ClassName) \
    REGISTER_GRID_BUILDER(sVectorGrid3Builders, ClassName)

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
