// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_vector_grid2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(CellCenteredVectorGrid2, Constructors) {
    // Default constructors
    CellCenteredVectorGrid2 grid1;
    EXPECT_EQ(0u, grid1.resolution().x);
    EXPECT_EQ(0u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().y);
    EXPECT_EQ(0u, grid1.dataSize().x);
    EXPECT_EQ(0u, grid1.dataSize().y);
    EXPECT_DOUBLE_EQ(0.5, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(0.5, grid1.dataOrigin().y);

    // Constructor with params
    CellCenteredVectorGrid2 grid2(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.origin().y);
    EXPECT_EQ(5u, grid2.dataSize().x);
    EXPECT_EQ(4u, grid2.dataSize().y);
    EXPECT_DOUBLE_EQ(3.5, grid2.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid2.dataOrigin().y);
    grid2.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(5.0, grid2(i, j).x);
        EXPECT_DOUBLE_EQ(6.0, grid2(i, j).y);
    });

    // Copy constructor
    CellCenteredVectorGrid2 grid3(grid2);
    EXPECT_EQ(5u, grid3.resolution().x);
    EXPECT_EQ(4u, grid3.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid3.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid3.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid3.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid3.origin().y);
    EXPECT_EQ(5u, grid3.dataSize().x);
    EXPECT_EQ(4u, grid3.dataSize().y);
    EXPECT_DOUBLE_EQ(3.5, grid3.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid3.dataOrigin().y);
    grid3.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(5.0, grid3(i, j).x);
        EXPECT_DOUBLE_EQ(6.0, grid3(i, j).y);
    });
}

TEST(CellCenteredVectorGrid2, Swap) {
    CellCenteredVectorGrid2 grid1(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    CellCenteredVectorGrid2 grid2(3, 8, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0);
    grid1.swap(&grid2);

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().y);
    EXPECT_EQ(3u, grid1.dataSize().x);
    EXPECT_EQ(8u, grid1.dataSize().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(6.5, grid1.dataOrigin().y);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(4.0, grid1(i, j).x);
        EXPECT_DOUBLE_EQ(7.0, grid1(i, j).y);
    });

    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.origin().y);
    EXPECT_EQ(5u, grid2.dataSize().x);
    EXPECT_EQ(4u, grid2.dataSize().y);
    EXPECT_DOUBLE_EQ(3.5, grid2.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid2.dataOrigin().y);
    grid2.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(5.0, grid2(i, j).x);
        EXPECT_DOUBLE_EQ(6.0, grid2(i, j).y);
    });
}

TEST(CellCenteredVectorGrid2, Set) {
    CellCenteredVectorGrid2 grid1(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    CellCenteredVectorGrid2 grid2(3, 8, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0);
    grid1.set(grid2);

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().y);
    EXPECT_EQ(3u, grid1.dataSize().x);
    EXPECT_EQ(8u, grid1.dataSize().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(6.5, grid1.dataOrigin().y);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(4.0, grid1(i, j).x);
        EXPECT_DOUBLE_EQ(7.0, grid1(i, j).y);
    });
}

TEST(CellCenteredVectorGrid2, AssignmentOperator) {
    CellCenteredVectorGrid2 grid1(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    CellCenteredVectorGrid2 grid2(3, 8, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0);
    grid1 = grid2;

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().y);
    EXPECT_EQ(3u, grid1.dataSize().x);
    EXPECT_EQ(8u, grid1.dataSize().y);
    EXPECT_DOUBLE_EQ(2.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(6.5, grid1.dataOrigin().y);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(4.0, grid1(i, j).x);
        EXPECT_DOUBLE_EQ(7.0, grid1(i, j).y);
    });
}

TEST(CellCenteredVectorGrid2, Clone) {
    CellCenteredVectorGrid2 grid2(3, 8, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0);
    auto grid1 = grid2.clone();

    auto grid3 = std::dynamic_pointer_cast<CellCenteredVectorGrid2>(grid1);
    EXPECT_TRUE(grid3 != nullptr);

    EXPECT_EQ(3u, grid1->resolution().x);
    EXPECT_EQ(8u, grid1->resolution().y);
    EXPECT_DOUBLE_EQ(2.0, grid1->gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1->gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1->origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid1->origin().y);
    EXPECT_EQ(3u, grid3->dataSize().x);
    EXPECT_EQ(8u, grid3->dataSize().y);
    EXPECT_DOUBLE_EQ(2.0, grid3->dataOrigin().x);
    EXPECT_DOUBLE_EQ(6.5, grid3->dataOrigin().y);
    grid3->forEachDataPointIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(4.0, (*grid3)(i, j).x);
        EXPECT_DOUBLE_EQ(7.0, (*grid3)(i, j).y);
    });
}

TEST(CellCenteredVectorGrid2, Builder) {
    {
        auto grid1 = CellCenteredVectorGrid2::builder().build(
            Size2(3, 8), Vector2D(2.0, 3.0), Vector2D(1.0, 5.0), {4.0, 7.0});

        auto grid2 = std::dynamic_pointer_cast<CellCenteredVectorGrid2>(grid1);
        EXPECT_TRUE(grid2 != nullptr);

        EXPECT_EQ(3u, grid1->resolution().x);
        EXPECT_EQ(8u, grid1->resolution().y);
        EXPECT_DOUBLE_EQ(2.0, grid1->gridSpacing().x);
        EXPECT_DOUBLE_EQ(3.0, grid1->gridSpacing().y);
        EXPECT_DOUBLE_EQ(1.0, grid1->origin().x);
        EXPECT_DOUBLE_EQ(5.0, grid1->origin().y);
        EXPECT_EQ(3u, grid2->dataSize().x);
        EXPECT_EQ(8u, grid2->dataSize().y);
        EXPECT_DOUBLE_EQ(2.0, grid2->dataOrigin().x);
        EXPECT_DOUBLE_EQ(6.5, grid2->dataOrigin().y);
        grid2->forEachDataPointIndex([&] (size_t i, size_t j) {
            EXPECT_DOUBLE_EQ(4.0, (*grid2)(i, j).x);
            EXPECT_DOUBLE_EQ(7.0, (*grid2)(i, j).y);
        });
    }

    {
        auto grid1 = CellCenteredVectorGrid2::builder().withResolution(3, 8)
                                                       .withGridSpacing(2, 3)
                                                       .withOrigin(1, 5)
                                                       .withInitialValue(4, 7)
                                                       .build();

        EXPECT_EQ(3u, grid1.resolution().x);
        EXPECT_EQ(8u, grid1.resolution().y);
        EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
        EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
        EXPECT_DOUBLE_EQ(1.0, grid1.origin().x);
        EXPECT_DOUBLE_EQ(5.0, grid1.origin().y);
        EXPECT_EQ(3u, grid1.dataSize().x);
        EXPECT_EQ(8u, grid1.dataSize().y);
        EXPECT_DOUBLE_EQ(2.0, grid1.dataOrigin().x);
        EXPECT_DOUBLE_EQ(6.5, grid1.dataOrigin().y);
        grid1.forEachDataPointIndex([&] (size_t i, size_t j) {
            EXPECT_DOUBLE_EQ(4.0, grid1(i, j).x);
            EXPECT_DOUBLE_EQ(7.0, grid1(i, j).y);
        });
    }
}

TEST(CellCenteredVectorGrid2, Fill) {
    CellCenteredVectorGrid2 grid(5, 4, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    grid.fill(Vector2D(42.0, 27.0));

    for (size_t j = 0; j < grid.dataSize().y; ++j) {
        for (size_t i = 0; i < grid.dataSize().x; ++i) {
            EXPECT_DOUBLE_EQ(42.0, grid(i, j).x);
            EXPECT_DOUBLE_EQ(27.0, grid(i, j).y);
        }
    }

    auto func = [](const Vector2D& x) {
        if (x.x < 3.0) {
            return Vector2D(2.0, 3.0);
        } else {
            return Vector2D(5.0, 7.0);
        }
    };
    grid.fill(func);

    for (size_t j = 0; j < grid.dataSize().y; ++j) {
        for (size_t i = 0; i < grid.dataSize().x; ++i) {
            if (i < 3) {
                EXPECT_DOUBLE_EQ(2.0, grid(i, j).x);
                EXPECT_DOUBLE_EQ(3.0, grid(i, j).y);
            } else {
                EXPECT_DOUBLE_EQ(5.0, grid(i, j).x);
                EXPECT_DOUBLE_EQ(7.0, grid(i, j).y);
            }
        }
    }
}

TEST(CellCenteredVectorGrid2, DivergenceAtDataPoint) {
    CellCenteredVectorGrid2 grid(5, 8, 2.0, 3.0);

    grid.fill(Vector2D(1.0, -2.0));

    for (size_t j = 0; j < grid.resolution().y; ++j) {
        for (size_t i = 0; i < grid.resolution().x; ++i) {
            EXPECT_DOUBLE_EQ(0.0, grid.divergenceAtDataPoint(i, j));
        }
    }

    grid.fill([](const Vector2D& x) { return x; });

    for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
        for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
            EXPECT_NEAR(2.0, grid.divergenceAtDataPoint(i, j), 1e-6);
        }
    }
}

TEST(CellCenteredVectorGrid2, CurlAtAtDataPoint) {
    CellCenteredVectorGrid2 grid(5, 8, 2.0, 3.0);

    grid.fill(Vector2D(1.0, -2.0));

    for (size_t j = 0; j < grid.resolution().y; ++j) {
        for (size_t i = 0; i < grid.resolution().x; ++i) {
            EXPECT_DOUBLE_EQ(0.0, grid.curlAtDataPoint(i, j));
        }
    }

    grid.fill([](const Vector2D& x) { return Vector2D(-x.y, x.x); });

    for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
        for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
            EXPECT_NEAR(2.0, grid.curlAtDataPoint(i, j), 1e-6);
        }
    }
}

TEST(CellCenteredVectorGrid2, Serialization) {
    CellCenteredVectorGrid2 grid1(5, 4, 1.0, 2.0, -5.0, 3.0);
    grid1.fill([&] (const Vector2D& pt) {
        return Vector2D(pt.x, pt.y);
    });

    // Serialize to in-memoery stream
    std::vector<uint8_t> buffer1;
    grid1.serialize(&buffer1);

    // Deserialize to non-zero array
    CellCenteredVectorGrid2 grid2(1, 2, 0.5, 1.0, 0.5, 2.0);
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
        EXPECT_DOUBLE_EQ(grid1(i, j).x, grid2(i, j).x);
        EXPECT_DOUBLE_EQ(grid1(i, j).y, grid2(i, j).y);
    });

    // Serialize zero-sized array
    CellCenteredVectorGrid2 grid3;
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
