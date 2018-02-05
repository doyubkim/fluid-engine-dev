// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_vector_grid2.h>
#include <jet/vertex_centered_scalar_grid2.h>
#include <jet/vertex_centered_vector_grid2.h>

#include <jet/grid_system_data2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(GridSystemData2, Constructors) {
    GridSystemData2 grids1;
    EXPECT_EQ(0u, grids1.resolution().x);
    EXPECT_EQ(0u, grids1.resolution().y);
    EXPECT_EQ(1.0, grids1.gridSpacing().x);
    EXPECT_EQ(1.0, grids1.gridSpacing().y);
    EXPECT_EQ(0.0, grids1.origin().x);
    EXPECT_EQ(0.0, grids1.origin().y);

    GridSystemData2 grids2(
        {32, 64},
        {1.0, 2.0},
        {-5.0, 4.5});

    EXPECT_EQ(32u, grids2.resolution().x);
    EXPECT_EQ(64u, grids2.resolution().y);
    EXPECT_EQ(1.0, grids2.gridSpacing().x);
    EXPECT_EQ(2.0, grids2.gridSpacing().y);
    EXPECT_EQ(-5.0, grids2.origin().x);
    EXPECT_EQ(4.5, grids2.origin().y);

    GridSystemData2 grids3(grids2);

    EXPECT_EQ(32u, grids3.resolution().x);
    EXPECT_EQ(64u, grids3.resolution().y);
    EXPECT_EQ(1.0, grids3.gridSpacing().x);
    EXPECT_EQ(2.0, grids3.gridSpacing().y);
    EXPECT_EQ(-5.0, grids3.origin().x);
    EXPECT_EQ(4.5, grids3.origin().y);

    EXPECT_TRUE(grids2.velocity() != grids3.velocity());
}

TEST(GridSystemData2, Serialize) {
    std::vector<uint8_t> buffer;

    GridSystemData2 grids(
        {32, 64},
        {1.0, 2.0},
        {-5.0, 4.5});

    size_t scalarIdx0 = grids.addScalarData(
        std::make_shared<CellCenteredScalarGrid2::Builder>());
    size_t vectorIdx0 = grids.addVectorData(
        std::make_shared<CellCenteredVectorGrid2::Builder>());
    size_t scalarIdx1 = grids.addAdvectableScalarData(
        std::make_shared<VertexCenteredScalarGrid2::Builder>());
    size_t vectorIdx1 = grids.addAdvectableVectorData(
        std::make_shared<VertexCenteredVectorGrid2::Builder>());

    auto scalar0 = grids.scalarDataAt(scalarIdx0);
    auto vector0 = grids.vectorDataAt(vectorIdx0);
    auto scalar1 = grids.advectableScalarDataAt(scalarIdx1);
    auto vector1 = grids.advectableVectorDataAt(vectorIdx1);

    scalar0->fill([] (const Vector2D& pt) {
        return pt.length();
    });

    vector0->fill([] (const Vector2D& pt) {
        return pt;
    });

    scalar1->fill([] (const Vector2D& pt) {
        return (pt - Vector2D(1, 2)).length();
    });

    vector1->fill([] (const Vector2D& pt) {
        return pt - Vector2D(1, 2);
    });

    grids.serialize(&buffer);

    GridSystemData2 grids2;
    grids2.deserialize(buffer);

    EXPECT_EQ(32u, grids2.resolution().x);
    EXPECT_EQ(64u, grids2.resolution().y);
    EXPECT_EQ(1.0, grids2.gridSpacing().x);
    EXPECT_EQ(2.0, grids2.gridSpacing().y);
    EXPECT_EQ(-5.0, grids2.origin().x);
    EXPECT_EQ(4.5, grids2.origin().y);

    EXPECT_EQ(1u, grids2.numberOfScalarData());
    EXPECT_EQ(1u, grids2.numberOfVectorData());
    EXPECT_EQ(1u, grids2.numberOfAdvectableScalarData());
    EXPECT_EQ(2u, grids2.numberOfAdvectableVectorData());

    auto scalar0_2 = grids2.scalarDataAt(scalarIdx0);
    EXPECT_TRUE(
        std::dynamic_pointer_cast<CellCenteredScalarGrid2>(scalar0_2)
        != nullptr);
    EXPECT_EQ(scalar0->resolution(), scalar0_2->resolution());
    EXPECT_EQ(scalar0->gridSpacing(), scalar0_2->gridSpacing());
    EXPECT_EQ(scalar0->origin(), scalar0_2->origin());
    EXPECT_EQ(scalar0->dataSize(), scalar0_2->dataSize());
    EXPECT_EQ(scalar0->dataOrigin(), scalar0_2->dataOrigin());
    scalar0->forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_EQ((*scalar0)(i, j), (*scalar0_2)(i, j));
    });

    auto vector0_2 = grids2.vectorDataAt(vectorIdx0);
    auto cell_vector0
        = std::dynamic_pointer_cast<CellCenteredVectorGrid2>(vector0);
    auto cell_vector0_2
        = std::dynamic_pointer_cast<CellCenteredVectorGrid2>(vector0_2);
    EXPECT_TRUE(cell_vector0_2 != nullptr);
    EXPECT_EQ(vector0->resolution(), vector0_2->resolution());
    EXPECT_EQ(vector0->gridSpacing(), vector0_2->gridSpacing());
    EXPECT_EQ(vector0->origin(), vector0_2->origin());
    EXPECT_EQ(cell_vector0->dataSize(), cell_vector0_2->dataSize());
    EXPECT_EQ(cell_vector0->dataOrigin(), cell_vector0_2->dataOrigin());
    cell_vector0->forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_EQ((*cell_vector0)(i, j), (*cell_vector0_2)(i, j));
    });

    auto scalar1_2 = grids2.advectableScalarDataAt(scalarIdx1);
    EXPECT_TRUE(
        std::dynamic_pointer_cast<VertexCenteredScalarGrid2>(scalar1_2)
        != nullptr);
    EXPECT_EQ(scalar1->resolution(), scalar1_2->resolution());
    EXPECT_EQ(scalar1->gridSpacing(), scalar1_2->gridSpacing());
    EXPECT_EQ(scalar1->origin(), scalar1_2->origin());
    EXPECT_EQ(scalar1->dataSize(), scalar1_2->dataSize());
    EXPECT_EQ(scalar1->dataOrigin(), scalar1_2->dataOrigin());
    scalar1->forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_EQ((*scalar1)(i, j), (*scalar1_2)(i, j));
    });

    auto vector1_2 = grids2.advectableVectorDataAt(vectorIdx1);
    auto vert_vector1
        = std::dynamic_pointer_cast<VertexCenteredVectorGrid2>(vector1);
    auto vert_vector1_2
        = std::dynamic_pointer_cast<VertexCenteredVectorGrid2>(vector1_2);
    EXPECT_TRUE(vert_vector1_2 != nullptr);
    EXPECT_EQ(vector1->resolution(), vector1_2->resolution());
    EXPECT_EQ(vector1->gridSpacing(), vector1_2->gridSpacing());
    EXPECT_EQ(vector1->origin(), vector1_2->origin());
    EXPECT_EQ(vert_vector1->dataSize(), vert_vector1_2->dataSize());
    EXPECT_EQ(vert_vector1->dataOrigin(), vert_vector1_2->dataOrigin());
    vert_vector1->forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_EQ((*vert_vector1)(i, j), (*vert_vector1_2)(i, j));
    });

    auto velocity = grids.velocity();
    auto velocity2 = grids2.velocity();
    EXPECT_EQ(velocity->resolution(), velocity2->resolution());
    EXPECT_EQ(velocity->gridSpacing(), velocity2->gridSpacing());
    EXPECT_EQ(velocity->origin(), velocity2->origin());
    EXPECT_EQ(velocity->uSize(), velocity2->uSize());
    EXPECT_EQ(velocity->vSize(), velocity2->vSize());
    EXPECT_EQ(velocity->uOrigin(), velocity2->uOrigin());
    EXPECT_EQ(velocity->vOrigin(), velocity2->vOrigin());
    velocity->forEachUIndex([&] (size_t i, size_t j) {
        EXPECT_EQ(velocity->u(i, j), velocity2->u(i, j));
    });
    velocity->forEachVIndex([&] (size_t i, size_t j) {
        EXPECT_EQ(velocity->v(i, j), velocity2->v(i, j));
    });
}
