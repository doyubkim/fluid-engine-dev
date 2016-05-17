// Copyright (c) 2016 Doyub Kim

#include <jet/cell_centered_vector_grid2.h>
#include <jet/cell_centered_vector_grid3.h>
#include <gtest/gtest.h>

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
    std::stringstream strm1;
    grid1.serialize(&strm1);

    // Deserialize to non-zero array
    CellCenteredVectorGrid2 grid2(1, 2, 0.5, 1.0, 0.5, 2.0);
    grid2.deserialize(&strm1);
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
    std::stringstream strm2;
    grid3.serialize(&strm2);

    // Deserialize to non-zero array
    grid2.deserialize(&strm2);
    EXPECT_EQ(0u, grid2.resolution().x);
    EXPECT_EQ(0u, grid2.resolution().y);
    EXPECT_DOUBLE_EQ(0.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(0.0, grid2.origin().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().y);
}


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
    std::stringstream strm1;
    grid1.serialize(&strm1);

    // Deserialize to non-zero array
    CellCenteredVectorGrid3 grid2(1, 2, 4, 0.5, 1.0, 2.0, 0.5, 2.0, -3.0);
    grid2.deserialize(&strm1);
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
    std::stringstream strm2;
    grid3.serialize(&strm2);

    // Deserialize to non-zero array
    grid2.deserialize(&strm2);
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
