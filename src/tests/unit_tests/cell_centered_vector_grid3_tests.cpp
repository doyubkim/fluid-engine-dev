// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/cell_centered_vector_grid3.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(CellCenteredVectorGrid3, Constructors) {
    // Default constructors
    CellCenteredVectorGrid3 grid1;
    EXPECT_EQ(0u, grid1.resolution().x);
    EXPECT_EQ(0u, grid1.resolution().y);
    EXPECT_EQ(0u, grid1.resolution().z);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().z);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().y);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().z);
    EXPECT_EQ(0u, grid1.dataSize().x);
    EXPECT_EQ(0u, grid1.dataSize().y);
    EXPECT_EQ(0u, grid1.dataSize().z);
    EXPECT_DOUBLE_EQ(0.5, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(0.5, grid1.dataOrigin().y);
    EXPECT_DOUBLE_EQ(0.5, grid1.dataOrigin().z);

    // Constructor with params
    CellCenteredVectorGrid3 grid2(
        5, 4, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_EQ(3u, grid2.resolution().z);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.gridSpacing().z);
    EXPECT_DOUBLE_EQ(4.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid2.origin().y);
    EXPECT_DOUBLE_EQ(6.0, grid2.origin().z);
    EXPECT_EQ(5u, grid2.dataSize().x);
    EXPECT_EQ(4u, grid2.dataSize().y);
    EXPECT_EQ(3u, grid2.dataSize().z);
    EXPECT_DOUBLE_EQ(4.5, grid2.dataOrigin().x);
    EXPECT_DOUBLE_EQ(6.0, grid2.dataOrigin().y);
    EXPECT_DOUBLE_EQ(7.5, grid2.dataOrigin().z);
    grid2.forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        EXPECT_DOUBLE_EQ(7.0, grid2(i, j, k).x);
        EXPECT_DOUBLE_EQ(8.0, grid2(i, j, k).y);
        EXPECT_DOUBLE_EQ(9.0, grid2(i, j, k).z);
    });

    // Copy constructor
    CellCenteredVectorGrid3 grid3(grid2);
    EXPECT_EQ(5u, grid3.resolution().x);
    EXPECT_EQ(4u, grid3.resolution().y);
    EXPECT_EQ(3u, grid3.resolution().z);
    EXPECT_DOUBLE_EQ(1.0, grid3.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid3.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid3.gridSpacing().z);
    EXPECT_DOUBLE_EQ(4.0, grid3.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid3.origin().y);
    EXPECT_DOUBLE_EQ(6.0, grid3.origin().z);
    EXPECT_EQ(5u, grid3.dataSize().x);
    EXPECT_EQ(4u, grid3.dataSize().y);
    EXPECT_EQ(3u, grid3.dataSize().z);
    EXPECT_DOUBLE_EQ(4.5, grid3.dataOrigin().x);
    EXPECT_DOUBLE_EQ(6.0, grid3.dataOrigin().y);
    EXPECT_DOUBLE_EQ(7.5, grid3.dataOrigin().z);
    grid3.forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        EXPECT_DOUBLE_EQ(7.0, grid3(i, j, k).x);
        EXPECT_DOUBLE_EQ(8.0, grid3(i, j, k).y);
        EXPECT_DOUBLE_EQ(9.0, grid3(i, j, k).z);
    });
}

TEST(CellCenteredVectorGrid3, Swap) {
    CellCenteredVectorGrid3 grid1(
        5, 4, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    CellCenteredVectorGrid3 grid2(
        3, 8, 5, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0, 8.0, 1.0, 3.0);
    grid1.swap(&grid2);

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_EQ(5u, grid1.resolution().z);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().z);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid1.origin().y);
    EXPECT_DOUBLE_EQ(7.0, grid1.origin().z);
    EXPECT_EQ(3u, grid1.dataSize().x);
    EXPECT_EQ(8u, grid1.dataSize().y);
    EXPECT_EQ(5u, grid1.dataSize().z);
    EXPECT_DOUBLE_EQ(6.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.5, grid1.dataOrigin().y);
    EXPECT_DOUBLE_EQ(7.5, grid1.dataOrigin().z);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        EXPECT_DOUBLE_EQ(8.0, grid1(i, j, k).x);
        EXPECT_DOUBLE_EQ(1.0, grid1(i, j, k).y);
        EXPECT_DOUBLE_EQ(3.0, grid1(i, j, k).z);
    });

    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_EQ(3u, grid2.resolution().z);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.gridSpacing().z);
    EXPECT_DOUBLE_EQ(4.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(5.0, grid2.origin().y);
    EXPECT_DOUBLE_EQ(6.0, grid2.origin().z);
    EXPECT_EQ(5u, grid2.dataSize().x);
    EXPECT_EQ(4u, grid2.dataSize().y);
    EXPECT_EQ(3u, grid2.dataSize().z);
    EXPECT_DOUBLE_EQ(4.5, grid2.dataOrigin().x);
    EXPECT_DOUBLE_EQ(6.0, grid2.dataOrigin().y);
    EXPECT_DOUBLE_EQ(7.5, grid2.dataOrigin().z);
    grid2.forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        EXPECT_DOUBLE_EQ(7.0, grid2(i, j, k).x);
        EXPECT_DOUBLE_EQ(8.0, grid2(i, j, k).y);
        EXPECT_DOUBLE_EQ(9.0, grid2(i, j, k).z);
    });
}

TEST(CellCenteredVectorGrid3, Set) {
    CellCenteredVectorGrid3 grid1(
        5, 4, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    CellCenteredVectorGrid3 grid2(
        3, 8, 5, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0, 8.0, 1.0, 3.0);
    grid1.set(grid2);

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_EQ(5u, grid1.resolution().z);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().z);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid1.origin().y);
    EXPECT_DOUBLE_EQ(7.0, grid1.origin().z);
    EXPECT_EQ(3u, grid1.dataSize().x);
    EXPECT_EQ(8u, grid1.dataSize().y);
    EXPECT_EQ(5u, grid1.dataSize().z);
    EXPECT_DOUBLE_EQ(6.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.5, grid1.dataOrigin().y);
    EXPECT_DOUBLE_EQ(7.5, grid1.dataOrigin().z);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        EXPECT_DOUBLE_EQ(8.0, grid1(i, j, k).x);
        EXPECT_DOUBLE_EQ(1.0, grid1(i, j, k).y);
        EXPECT_DOUBLE_EQ(3.0, grid1(i, j, k).z);
    });
}

TEST(CellCenteredVectorGrid3, AssignmentOperator) {
    CellCenteredVectorGrid3 grid1(
        5, 4, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    CellCenteredVectorGrid3 grid2(
        3, 8, 5, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0, 8.0, 1.0, 3.0);
    grid1 = grid2;

    EXPECT_EQ(3u, grid1.resolution().x);
    EXPECT_EQ(8u, grid1.resolution().y);
    EXPECT_EQ(5u, grid1.resolution().z);
    EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().z);
    EXPECT_DOUBLE_EQ(5.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid1.origin().y);
    EXPECT_DOUBLE_EQ(7.0, grid1.origin().z);
    EXPECT_EQ(3u, grid1.dataSize().x);
    EXPECT_EQ(8u, grid1.dataSize().y);
    EXPECT_EQ(5u, grid1.dataSize().z);
    EXPECT_DOUBLE_EQ(6.0, grid1.dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.5, grid1.dataOrigin().y);
    EXPECT_DOUBLE_EQ(7.5, grid1.dataOrigin().z);
    grid1.forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        EXPECT_DOUBLE_EQ(8.0, grid1(i, j, k).x);
        EXPECT_DOUBLE_EQ(1.0, grid1(i, j, k).y);
        EXPECT_DOUBLE_EQ(3.0, grid1(i, j, k).z);
    });
}

TEST(CellCenteredVectorGrid3, Clone) {
    CellCenteredVectorGrid3 grid2(
        3, 8, 5, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0, 8.0, 1.0, 3.0);
    auto grid1 = grid2.clone();

    auto grid3 = std::dynamic_pointer_cast<CellCenteredVectorGrid3>(grid1);
    EXPECT_TRUE(grid3 != nullptr);

    EXPECT_EQ(3u, grid1->resolution().x);
    EXPECT_EQ(8u, grid1->resolution().y);
    EXPECT_EQ(5u, grid1->resolution().z);
    EXPECT_DOUBLE_EQ(2.0, grid1->gridSpacing().x);
    EXPECT_DOUBLE_EQ(3.0, grid1->gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid1->gridSpacing().z);
    EXPECT_DOUBLE_EQ(5.0, grid1->origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid1->origin().y);
    EXPECT_DOUBLE_EQ(7.0, grid1->origin().z);
    EXPECT_EQ(3u, grid3->dataSize().x);
    EXPECT_EQ(8u, grid3->dataSize().y);
    EXPECT_EQ(5u, grid3->dataSize().z);
    EXPECT_DOUBLE_EQ(6.0, grid3->dataOrigin().x);
    EXPECT_DOUBLE_EQ(5.5, grid3->dataOrigin().y);
    EXPECT_DOUBLE_EQ(7.5, grid3->dataOrigin().z);
    grid3->forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        EXPECT_DOUBLE_EQ(8.0, (*grid3)(i, j, k).x);
        EXPECT_DOUBLE_EQ(1.0, (*grid3)(i, j, k).y);
        EXPECT_DOUBLE_EQ(3.0, (*grid3)(i, j, k).z);
    });
}

TEST(CellCenteredVectorGrid3, Builder) {
    {
        auto grid1 = CellCenteredVectorGrid3::builder().build(
            {3, 8, 5}, {2.0, 3.0, 1.0}, {5.0, 4.0, 7.0}, {8.0, 1.0, 3.0});

        auto grid2 = std::dynamic_pointer_cast<CellCenteredVectorGrid3>(grid1);
        EXPECT_TRUE(grid2 != nullptr);

        EXPECT_EQ(3u, grid1->resolution().x);
        EXPECT_EQ(8u, grid1->resolution().y);
        EXPECT_EQ(5u, grid1->resolution().z);
        EXPECT_DOUBLE_EQ(2.0, grid1->gridSpacing().x);
        EXPECT_DOUBLE_EQ(3.0, grid1->gridSpacing().y);
        EXPECT_DOUBLE_EQ(1.0, grid1->gridSpacing().z);
        EXPECT_DOUBLE_EQ(5.0, grid1->origin().x);
        EXPECT_DOUBLE_EQ(4.0, grid1->origin().y);
        EXPECT_DOUBLE_EQ(7.0, grid1->origin().z);
        EXPECT_EQ(3u, grid2->dataSize().x);
        EXPECT_EQ(8u, grid2->dataSize().y);
        EXPECT_EQ(5u, grid2->dataSize().z);
        EXPECT_DOUBLE_EQ(6.0, grid2->dataOrigin().x);
        EXPECT_DOUBLE_EQ(5.5, grid2->dataOrigin().y);
        EXPECT_DOUBLE_EQ(7.5, grid2->dataOrigin().z);
        grid2->forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
            EXPECT_DOUBLE_EQ(8.0, (*grid2)(i, j, k).x);
            EXPECT_DOUBLE_EQ(1.0, (*grid2)(i, j, k).y);
            EXPECT_DOUBLE_EQ(3.0, (*grid2)(i, j, k).z);
        });
    }

    {
        auto grid1 = CellCenteredVectorGrid3::builder()
            .withResolution(3, 8, 5)
            .withGridSpacing(2.0, 3.0, 1.0)
            .withOrigin(5.0, 4.0, 7.0)
            .withInitialValue(8.0, 1.0, 3.0)
            .build();

        EXPECT_EQ(3u, grid1.resolution().x);
        EXPECT_EQ(8u, grid1.resolution().y);
        EXPECT_EQ(5u, grid1.resolution().z);
        EXPECT_DOUBLE_EQ(2.0, grid1.gridSpacing().x);
        EXPECT_DOUBLE_EQ(3.0, grid1.gridSpacing().y);
        EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().z);
        EXPECT_DOUBLE_EQ(5.0, grid1.origin().x);
        EXPECT_DOUBLE_EQ(4.0, grid1.origin().y);
        EXPECT_DOUBLE_EQ(7.0, grid1.origin().z);
        EXPECT_EQ(3u, grid1.dataSize().x);
        EXPECT_EQ(8u, grid1.dataSize().y);
        EXPECT_EQ(5u, grid1.dataSize().z);
        EXPECT_DOUBLE_EQ(6.0, grid1.dataOrigin().x);
        EXPECT_DOUBLE_EQ(5.5, grid1.dataOrigin().y);
        EXPECT_DOUBLE_EQ(7.5, grid1.dataOrigin().z);
        grid1.forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
            EXPECT_DOUBLE_EQ(8.0, grid1(i, j, k).x);
            EXPECT_DOUBLE_EQ(1.0, grid1(i, j, k).y);
            EXPECT_DOUBLE_EQ(3.0, grid1(i, j, k).z);
        });
    }
}

TEST(CellCenteredVectorGrid3, Fill) {
    CellCenteredVectorGrid3 grid(
        5, 4, 6, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    grid.fill(Vector3D(42.0, 27.0, 31.0));

    for (size_t k = 0; k < grid.dataSize().z; ++k) {
        for (size_t j = 0; j < grid.dataSize().y; ++j) {
            for (size_t i = 0; i < grid.dataSize().x; ++i) {
                EXPECT_DOUBLE_EQ(42.0, grid(i, j, k).x);
                EXPECT_DOUBLE_EQ(27.0, grid(i, j, k).y);
                EXPECT_DOUBLE_EQ(31.0, grid(i, j, k).z);
            }
        }
    }

    auto func = [](const Vector3D& x) {
        if (x.x < 3.0) {
            return Vector3D(2.0, 3.0, 1.0);
        } else {
            return Vector3D(5.0, 7.0, 9.0);
        }
    };
    grid.fill(func);

    for (size_t k = 0; k < grid.dataSize().z; ++k) {
        for (size_t j = 0; j < grid.dataSize().y; ++j) {
            for (size_t i = 0; i < grid.dataSize().x; ++i) {
                if (i < 3) {
                    EXPECT_DOUBLE_EQ(2.0, grid(i, j, k).x);
                    EXPECT_DOUBLE_EQ(3.0, grid(i, j, k).y);
                    EXPECT_DOUBLE_EQ(1.0, grid(i, j, k).z);
                } else {
                    EXPECT_DOUBLE_EQ(5.0, grid(i, j, k).x);
                    EXPECT_DOUBLE_EQ(7.0, grid(i, j, k).y);
                    EXPECT_DOUBLE_EQ(9.0, grid(i, j, k).z);
                }
            }
        }
    }
}

TEST(CellCenteredVectorGrid3, DivergenceAtDataPoint) {
    CellCenteredVectorGrid3 grid(5, 8, 6);

    grid.fill(Vector3D(1.0, -2.0, 3.0));

    for (size_t k = 0; k < grid.resolution().z; ++k) {
        for (size_t j = 0; j < grid.resolution().y; ++j) {
            for (size_t i = 0; i < grid.resolution().x; ++i) {
                EXPECT_DOUBLE_EQ(0.0, grid.divergenceAtDataPoint(i, j, k));
            }
        }
    }

    grid.fill([](const Vector3D& x) { return x; });

    for (size_t k = 1; k < grid.resolution().z - 1; ++k) {
        for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
            for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
                EXPECT_DOUBLE_EQ(3.0, grid.divergenceAtDataPoint(i, j, k));
            }
        }
    }
}

TEST(CellCenteredVectorGrid3, CurlAtDataPoint) {
    CellCenteredVectorGrid3 grid(5, 8, 6, 2.0, 3.0, 1.5);

    grid.fill(Vector3D(1.0, -2.0, 3.0));

    for (size_t k = 0; k < grid.resolution().z; ++k) {
        for (size_t j = 0; j < grid.resolution().y; ++j) {
            for (size_t i = 0; i < grid.resolution().x; ++i) {
                Vector3D curl = grid.curlAtDataPoint(i, j, k);
                EXPECT_DOUBLE_EQ(0.0, curl.x);
                EXPECT_DOUBLE_EQ(0.0, curl.y);
                EXPECT_DOUBLE_EQ(0.0, curl.z);
            }
        }
    }

    grid.fill([](const Vector3D& x) { return Vector3D(x.y, x.z, x.x); });

    for (size_t k = 1; k < grid.resolution().z - 1; ++k) {
        for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
            for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
                Vector3D curl = grid.curlAtDataPoint(i, j, k);
                EXPECT_DOUBLE_EQ(-1.0, curl.x);
                EXPECT_DOUBLE_EQ(-1.0, curl.y);
                EXPECT_DOUBLE_EQ(-1.0, curl.z);
            }
        }
    }
}

TEST(CellCenteredVectorGrid3, Serialization) {
    CellCenteredVectorGrid3 grid1(5, 4, 3, 1.0, 2.0, 3.0, -5.0, 3.0, 1.0);
    grid1.fill([&] (const Vector3D& pt) {
        return Vector3D(pt.x, pt.y, pt.z);
    });

    // Serialize to in-memoery stream
    std::vector<uint8_t> buffer1;
    grid1.serialize(&buffer1);

    // Deserialize to non-zero array
    CellCenteredVectorGrid3 grid2(1, 2, 4, 0.5, 1.0, 2.0, 0.5, 2.0, -3.0);
    grid2.deserialize(buffer1);
    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_EQ(3u, grid2.resolution().z);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.gridSpacing().z);
    EXPECT_DOUBLE_EQ(-5.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(3.0, grid2.origin().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.origin().z);
    EXPECT_DOUBLE_EQ(-5.0, grid2.boundingBox().lowerCorner.x);
    EXPECT_DOUBLE_EQ(3.0, grid2.boundingBox().lowerCorner.y);
    EXPECT_DOUBLE_EQ(1.0, grid2.boundingBox().lowerCorner.z);
    EXPECT_DOUBLE_EQ(0.0, grid2.boundingBox().upperCorner.x);
    EXPECT_DOUBLE_EQ(11.0, grid2.boundingBox().upperCorner.y);
    EXPECT_DOUBLE_EQ(10.0, grid2.boundingBox().upperCorner.z);

    grid1.forEachDataPointIndex([&] (size_t i, size_t j, size_t k) {
        EXPECT_EQ(grid1(i, j, k), grid2(i, j, k));
    });

    // Serialize zero-sized array
    CellCenteredVectorGrid3 grid3;
    std::vector<uint8_t> buffer2;
    grid3.serialize(&buffer2);

    // Deserialize to non-zero array
    grid2.deserialize(buffer2);
    EXPECT_EQ(0u, grid2.resolution().x);
    EXPECT_EQ(0u, grid2.resolution().y);
    EXPECT_EQ(0u, grid2.resolution().z);
    EXPECT_DOUBLE_EQ(0.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(0.0, grid2.origin().y);
    EXPECT_DOUBLE_EQ(0.0, grid2.origin().z);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().z);
}
