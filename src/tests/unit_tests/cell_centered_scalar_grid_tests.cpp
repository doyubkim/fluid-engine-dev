// Copyright (c) 2016 Doyub Kim

#include <jet/cell_centered_scalar_grid2.h>
#include <jet/cell_centered_scalar_grid3.h>
#include <gtest/gtest.h>
#include <sstream>

using namespace jet;

TEST(CellCenteredScalarGrid2, Constructors) {
    // Default constructors
    CellCenteredScalarGrid2 grid1;
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
    CellCenteredScalarGrid2 grid2(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0);
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
        EXPECT_DOUBLE_EQ(5.0, grid2(i, j));
    });

    // Copy constructor
    CellCenteredScalarGrid2 grid3(grid2);
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
        EXPECT_DOUBLE_EQ(5.0, grid3(i, j));
    });
}

TEST(CellCenteredScalarGrid2, Fill) {
    CellCenteredScalarGrid2 grid(5, 4, 1.0, 1.0, 0.0, 0.0, 0.0);
    grid.fill(42.0);

    for (size_t j = 0; j < grid.dataSize().y; ++j) {
        for (size_t i = 0; i < grid.dataSize().x; ++i) {
            EXPECT_DOUBLE_EQ(42.0, grid(i, j));
        }
    }

    auto func = [](const Vector2D& x) { return x.sum(); };
    grid.fill(func);

    for (size_t j = 0; j < 4; ++j) {
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_DOUBLE_EQ(static_cast<double>(i + j) + 1.0, grid(i, j));
        }
    }
}

TEST(CellCenteredScalarGrid2, GradientAtAtDataPoint) {
    CellCenteredScalarGrid2 grid(5, 8, 2.0, 3.0);

    grid.fill(1.0);

    for (size_t j = 0; j < grid.resolution().y; ++j) {
        for (size_t i = 0; i < grid.resolution().x; ++i) {
            Vector2D grad = grid.gradientAtDataPoint(i, j);
            EXPECT_DOUBLE_EQ(0.0, grad.x);
            EXPECT_DOUBLE_EQ(0.0, grad.y);
        }
    }

    grid.fill([](const Vector2D& x) { return x.x + 2.0 * x.y; });

    for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
        for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
            Vector2D grad = grid.gradientAtDataPoint(i, j);
            EXPECT_NEAR(1.0, grad.x, 1e-6);
            EXPECT_NEAR(2.0, grad.y, 1e-6);
        }
    }
}

TEST(CellCenteredScalarGrid2, LaplacianAtAtDataPoint) {
    CellCenteredScalarGrid2 grid(5, 8, 2.0, 3.0);

    grid.fill(1.0);

    for (size_t j = 0; j < grid.resolution().y; ++j) {
        for (size_t i = 0; i < grid.resolution().x; ++i) {
            EXPECT_DOUBLE_EQ(0.0, grid.laplacianAtDataPoint(i, j));
        }
    }

    grid.fill([](const Vector2D& x) {
        return square(x.x) + 2.0 * square(x.y);
    });

    for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
        for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
            EXPECT_NEAR(6.0, grid.laplacianAtDataPoint(i, j), 1e-6);
        }
    }
}

TEST(CellCenteredScalarGrid2, Serialization) {
    CellCenteredScalarGrid2 grid1(5, 4, 1.0, 2.0, -5.0, 3.0);
    grid1.fill([&] (const Vector2D& pt) {
        return pt.x + pt.y;
    });

    // Serialize to in-memoery stream
    std::stringstream strm1;
    grid1.serialize(&strm1);

    // Deserialize to non-zero array
    CellCenteredScalarGrid2 grid2(1, 2, 0.5, 1.0, 0.5, 2.0);
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
        EXPECT_DOUBLE_EQ(grid1(i, j), grid2(i, j));
    });

    // Serialize zero-sized array
    CellCenteredScalarGrid2 grid3;
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


TEST(CellCenteredScalarGrid3, Constructors) {
    // Default constructors
    CellCenteredScalarGrid3 grid1;
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
    CellCenteredScalarGrid3 grid2(5, 4, 3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
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
        EXPECT_DOUBLE_EQ(7.0, grid2(i, j, k));
    });

    // Copy constructor
    CellCenteredScalarGrid3 grid3(grid2);
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
        EXPECT_DOUBLE_EQ(7.0, grid3(i, j, k));
    });
}

TEST(CellCenteredScalarGrid3, Fill) {
    CellCenteredScalarGrid3 grid(5, 4, 6, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    grid.fill(42.0);

    for (size_t k = 0; k < grid.dataSize().z; ++k) {
        for (size_t j = 0; j < grid.dataSize().y; ++j) {
            for (size_t i = 0; i < grid.dataSize().x; ++i) {
                EXPECT_DOUBLE_EQ(42.0, grid(i, j, k));
            }
        }
    }

    auto func = [](const Vector3D& x) { return x.sum(); };
    grid.fill(func);

    for (size_t k = 0; k < grid.dataSize().z; ++k) {
        for (size_t j = 0; j < grid.dataSize().y; ++j) {
            for (size_t i = 0; i < grid.dataSize().x; ++i) {
                EXPECT_DOUBLE_EQ(
                    static_cast<double>(i + j + k) + 1.5, grid(i, j, k));
            }
        }
    }
}

TEST(CellCenteredScalarGrid3, GradientAtDataPoint) {
    CellCenteredScalarGrid3 grid(5, 8, 6, 2.0, 3.0, 1.5);

    grid.fill(1.0);

    for (size_t k = 0; k < grid.resolution().z; ++k) {
        for (size_t j = 0; j < grid.resolution().y; ++j) {
            for (size_t i = 0; i < grid.resolution().x; ++i) {
                Vector3D grad = grid.gradientAtDataPoint(i, j, k);
                EXPECT_DOUBLE_EQ(0.0, grad.x);
                EXPECT_DOUBLE_EQ(0.0, grad.y);
                EXPECT_DOUBLE_EQ(0.0, grad.z);
            }
        }
    }

    grid.fill([](const Vector3D& x) { return x.x + 2.0 * x.y - 3.0 * x.z; });

    for (size_t k = 1; k < grid.resolution().z - 1; ++k) {
        for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
            for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
                Vector3D grad = grid.gradientAtDataPoint(i, j, k);
                EXPECT_DOUBLE_EQ(1.0, grad.x);
                EXPECT_DOUBLE_EQ(2.0, grad.y);
                EXPECT_DOUBLE_EQ(-3.0, grad.z);
            }
        }
    }
}

TEST(CellCenteredScalarGrid3, LaplacianAtAtDataPoint) {
    CellCenteredScalarGrid3 grid(5, 8, 6, 2.0, 3.0, 1.5);

    grid.fill(1.0);

    for (size_t k = 0; k < grid.resolution().z; ++k) {
        for (size_t j = 0; j < grid.resolution().y; ++j) {
            for (size_t i = 0; i < grid.resolution().x; ++i) {
                EXPECT_DOUBLE_EQ(0.0, grid.laplacianAtDataPoint(i, j, k));
            }
        }
    }

    grid.fill([](const Vector3D& x) {
        return square(x.x) + 2.0 * square(x.y) - 4.0 * square(x.z);
    });

    for (size_t k = 1; k < grid.resolution().z - 1; ++k) {
        for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
            for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
                EXPECT_DOUBLE_EQ(-2.0, grid.laplacianAtDataPoint(i, j, k));
            }
        }
    }
}

TEST(CellCenteredScalarGrid3, Serialization) {
    CellCenteredScalarGrid3 grid1(5, 4, 3, 1.0, 2.0, 3.0, -5.0, 3.0, 1.0);
    grid1.fill([&] (const Vector3D& pt) {
        return pt.x + pt.y + pt.z;
    });

    // Serialize to in-memoery stream
    std::stringstream strm1;
    grid1.serialize(&strm1);

    // Deserialize to non-zero array
    CellCenteredScalarGrid3 grid2(1, 2, 4, 0.5, 1.0, 2.0, 0.5, 2.0, -3.0);
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
        EXPECT_DOUBLE_EQ(grid1(i, j, k), grid2(i, j, k));
    });

    // Serialize zero-sized array
    CellCenteredScalarGrid3 grid3;
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
