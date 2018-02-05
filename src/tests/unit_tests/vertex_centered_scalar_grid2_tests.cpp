// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/vertex_centered_scalar_grid2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(VertexCenteredScalarGrid2, Constructors) {
    // Default constructors
    VertexCenteredScalarGrid2 grid1;
    EXPECT_EQ(0u, grid1.resolution().x);
    EXPECT_EQ(0u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().y);
    EXPECT_EQ(0u, grid1.dataSize().x);
    EXPECT_EQ(0u, grid1.dataSize().y);
    EXPECT_DOUBLE_EQ(0.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(0.0, grid1.dataOrigin().y);

    // Constructor with params
    VertexCenteredScalarGrid2 grid2(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0);
    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.origin().y);
    EXPECT_EQ(6u, grid2.dataSize().x);
    EXPECT_EQ(5u, grid2.dataSize().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.dataOrigin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.dataOrigin().y);
    grid2.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(5.0, grid2(i, j));
    });

    // Copy constructor
    VertexCenteredScalarGrid2 grid3(grid2);
    EXPECT_EQ(5u, grid3.resolution().x);
    EXPECT_EQ(4u, grid3.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid3.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid3.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid3.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid3.origin().y);
    EXPECT_EQ(6u, grid3.dataSize().x);
    EXPECT_EQ(5u, grid3.dataSize().y);
    EXPECT_DOUBLE_EQ(3.0, grid3.dataOrigin().x);
    EXPECT_DOUBLE_EQ(4.0, grid3.dataOrigin().y);
    grid3.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(5.0, grid3(i, j));
    });
}

TEST(VertexCenteredScalarGrid2, Swap) {
    VertexCenteredScalarGrid2 grid1(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0);
    VertexCenteredScalarGrid2 grid2(3, 8, 2.0, 3.0, 1.0, 5.0, 4.0);
    grid1.swap(&grid2);

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().y);
    EXPECT_EQ(4u, grid1.dataSize().x);
    EXPECT_EQ(9u, grid1.dataSize().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.dataOrigin().y);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(4.0, grid1(i, j));
    });

    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.origin().y);
    EXPECT_EQ(6u, grid2.dataSize().x);
    EXPECT_EQ(5u, grid2.dataSize().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.dataOrigin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.dataOrigin().y);
    grid2.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(5.0, grid2(i, j));
    });
}

TEST(VertexCenteredScalarGrid2, Set) {
    VertexCenteredScalarGrid2 grid1(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0);
    VertexCenteredScalarGrid2 grid2(3, 8, 2.0, 3.0, 1.0, 5.0, 4.0);
    grid1.set(grid2);

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().y);
    EXPECT_EQ(4u, grid1.dataSize().x);
    EXPECT_EQ(9u, grid1.dataSize().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.dataOrigin().y);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(4.0, grid1(i, j));
    });
}

TEST(VertexCenteredScalarGrid2, AssignmentOperator) {
    VertexCenteredScalarGrid2 grid1(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0);
    VertexCenteredScalarGrid2 grid2(3, 8, 2.0, 3.0, 1.0, 5.0, 4.0);
    grid1 = grid2;

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().y);
    EXPECT_EQ(4u, grid1.dataSize().x);
    EXPECT_EQ(9u, grid1.dataSize().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.dataOrigin().y);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(4.0, grid1(i, j));
    });
}

TEST(VertexCenteredScalarGrid2, Clone) {
    VertexCenteredScalarGrid2 grid2(3, 8, 2.0, 3.0, 1.0, 5.0, 4.0);
    auto grid1 = grid2.clone();

    EXPECT_EQ(3u, grid1->resolution().x);
    EXPECT_EQ(8u, grid1->resolution().y);
    EXPECT_DOUBLE_EQ(2.0, grid1->gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1->gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1->origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1->origin().y);
    EXPECT_EQ(4u, grid1->dataSize().x);
    EXPECT_EQ(9u, grid1->dataSize().y);
    EXPECT_DOUBLE_EQ(1.0, grid1->dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1->dataOrigin().y);
    grid1->forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(4.0, (*grid1)(i, j));
    });
}

TEST(VertexCenteredScalarGrid2, Builder) {
    {
        auto grid1 = VertexCenteredScalarGrid2::builder().build(
            Size2(3, 8), Vector2D(2.0, 3.0), Vector2D(1.0, 5.0), 4.0);

        auto grid2 = std::dynamic_pointer_cast<VertexCenteredScalarGrid2>(grid1);
        EXPECT_TRUE(grid2 != nullptr);

        EXPECT_EQ(3u, grid1->resolution().x);
        EXPECT_EQ(8u, grid1->resolution().y);
        EXPECT_DOUBLE_EQ(2.0, grid1->gridSpacing().x);
        EXPECT_DOUBLE_EQ(3.0, grid1->gridSpacing().y);
        EXPECT_DOUBLE_EQ(1.0, grid1->origin().x);
        EXPECT_DOUBLE_EQ(5.0, grid1->origin().y);
        EXPECT_EQ(4u, grid1->dataSize().x);
        EXPECT_EQ(9u, grid1->dataSize().y);
        EXPECT_DOUBLE_EQ(1.0, grid1->dataOrigin().x);
        EXPECT_DOUBLE_EQ(5.0, grid1->dataOrigin().y);
        grid1->forEachDataPointIndex([&] (size_t i, size_t j) {
            EXPECT_DOUBLE_EQ(4.0, (*grid1)(i, j));
        });
    }

    {
        auto grid1 = VertexCenteredScalarGrid2::builder()
            .withResolution(3, 8)
            .withGridSpacing(2, 3)
            .withOrigin(1, 5)
            .withInitialValue(4)
            .build();

        EXPECT_EQ(3u, grid1.resolution().x);
        EXPECT_EQ(8u, grid1.resolution().y);
        EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
        EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
        EXPECT_DOUBLE_EQ(1.0, grid1.origin().x);
        EXPECT_DOUBLE_EQ(5.0, grid1.origin().y);
        EXPECT_EQ(4u, grid1.dataSize().x);
        EXPECT_EQ(9u, grid1.dataSize().y);
        EXPECT_DOUBLE_EQ(1.0, grid1.dataOrigin().x);
        EXPECT_DOUBLE_EQ(5.0, grid1.dataOrigin().y);
        grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
            EXPECT_DOUBLE_EQ(4.0, grid1(i, j));
        });
    }
}

TEST(VertexCenteredScalarGrid2, Fill) {
    VertexCenteredScalarGrid2 grid(5, 4, 1.0, 1.0, 0.0, 0.0, 0.0);
    grid.fill(42.0);

    for (size_t j = 0; j < grid.dataSize().y; ++j) {
        for (size_t i = 0; i < grid.dataSize().x; ++i) {
            EXPECT_DOUBLE_EQ(42.0, grid(i, j));
        }
    }

    auto func = [](const Vector2D& x) {
        if (x.x < 3.0) {
            return 2.0;
        } else {
            return 5.0;
        }
    };
    grid.fill(func);

    for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 5; ++i) {
            if (i < 3) {
                EXPECT_DOUBLE_EQ(2.0, grid(i, j));
            } else {
                EXPECT_DOUBLE_EQ(5.0, grid(i, j));
            }
        }
    }
}

TEST(VertexCenteredScalarGrid2, Serialization) {
    VertexCenteredScalarGrid2 grid1(5, 4, 1.0, 2.0, -5.0, 3.0);
    grid1.fill([&] (const Vector2D& pt) {
        return pt.x + pt.y;
    });

    // Serialize to in-memoery stream
    std::vector<uint8_t> buffer1;
    grid1.serialize(&buffer1);

    // Deserialize to non-zero array
    VertexCenteredScalarGrid2 grid2(1, 2, 0.5, 1.0, 0.5, 2.0);
    grid2.deserialize(buffer1);
    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_DOUBLE_EQ(-5.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(3.0, grid2.origin().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(-5.0, grid2.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(3.0, grid2.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(0.0, grid2.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(11.0, grid2.boundingBox().upperCorner.y);

    grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(grid1(i, j), grid2(i, j));
    });

    // Serialize zero-sized array
    VertexCenteredScalarGrid2 grid3;
    std::vector<uint8_t> buffer2;
    grid3.serialize(&buffer2);

    // Deserialize to non-zero array
    grid2.deserialize(buffer2);
    EXPECT_EQ(0u, grid2.resolution().x);
    EXPECT_EQ(0u, grid2.resolution().y);
    EXPECT_DOUBLE_EQ(0.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(0.0, grid2.origin().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().y);
}
