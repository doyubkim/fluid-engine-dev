// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/face_centered_grid2.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(FaceCenteredGrid2, Constructors) {
    // Default constructors
    FaceCenteredGrid2 grid1;
    EXPECT_EQ(0u, grid1.resolution().x);
    EXPECT_EQ(0u, grid1.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().x);
    EXPECT_DOUBLE_EQ(1.0, grid1.gridSpacing().y);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().x);
    EXPECT_DOUBLE_EQ(0.0, grid1.origin().y);
    EXPECT_EQ(0u, grid1.uSize().x);
    EXPECT_EQ(0u, grid1.uSize().y);
    EXPECT_EQ(0u, grid1.vSize().x);
    EXPECT_EQ(0u, grid1.vSize().y);
    EXPECT_DOUBLE_EQ(0.0, grid1.uOrigin().x);
    EXPECT_DOUBLE_EQ(0.5, grid1.uOrigin().y);
    EXPECT_DOUBLE_EQ(0.5, grid1.vOrigin().x);
    EXPECT_DOUBLE_EQ(0.0, grid1.vOrigin().y);

    // Constructor with params
    FaceCenteredGrid2 grid2(5, 4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    EXPECT_EQ(5u, grid2.resolution().x);
    EXPECT_EQ(4u, grid2.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid2.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid2.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.origin().y);
    EXPECT_EQ(6u, grid2.uSize().x);
    EXPECT_EQ(4u, grid2.uSize().y);
    EXPECT_EQ(5u, grid2.vSize().x);
    EXPECT_EQ(5u, grid2.vSize().y);
    EXPECT_DOUBLE_EQ(3.0, grid2.uOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid2.uOrigin().y);
    EXPECT_DOUBLE_EQ(3.5, grid2.vOrigin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.vOrigin().y);
    grid2.forEachUIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(5.0, grid2.u(i, j));
    });
    grid2.forEachVIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(6.0, grid2.v(i, j));
    });

    // Copy constructor
    FaceCenteredGrid2 grid3(grid2);
    EXPECT_EQ(5u, grid3.resolution().x);
    EXPECT_EQ(4u, grid3.resolution().y);
    EXPECT_DOUBLE_EQ(1.0, grid3.gridSpacing().x);
    EXPECT_DOUBLE_EQ(2.0, grid3.gridSpacing().y);
    EXPECT_DOUBLE_EQ(3.0, grid3.origin().x);
    EXPECT_DOUBLE_EQ(4.0, grid3.origin().y);
    EXPECT_EQ(6u, grid3.uSize().x);
    EXPECT_EQ(4u, grid3.uSize().y);
    EXPECT_EQ(5u, grid3.vSize().x);
    EXPECT_EQ(5u, grid3.vSize().y);
    EXPECT_DOUBLE_EQ(3.0, grid3.uOrigin().x);
    EXPECT_DOUBLE_EQ(5.0, grid3.uOrigin().y);
    EXPECT_DOUBLE_EQ(3.5, grid3.vOrigin().x);
    EXPECT_DOUBLE_EQ(4.0, grid3.vOrigin().y);
    grid3.forEachUIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(5.0, grid3.u(i, j));
    });
    grid3.forEachVIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(6.0, grid3.v(i, j));
    });
}

TEST(FaceCenteredGrid2, Fill) {
    FaceCenteredGrid2 grid(5, 4, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    grid.fill(Vector2D(42.0, 27.0));

    for (size_t j = 0; j < grid.uSize().y; ++j) {
        for (size_t i = 0; i < grid.uSize().x; ++i) {
            EXPECT_DOUBLE_EQ(42.0, grid.u(i, j));
        }
    }

    for (size_t j = 0; j < grid.vSize().y; ++j) {
        for (size_t i = 0; i < grid.vSize().x; ++i) {
            EXPECT_DOUBLE_EQ(27.0, grid.v(i, j));
        }
    }

    auto func = [](const Vector2D& x) { return x; };
    grid.fill(func);

    for (size_t j = 0; j < grid.uSize().y; ++j) {
        for (size_t i = 0; i < grid.uSize().x; ++i) {
            EXPECT_DOUBLE_EQ(static_cast<double>(i), grid.u(i, j));
        }
    }

    for (size_t j = 0; j < grid.vSize().y; ++j) {
        for (size_t i = 0; i < grid.vSize().x; ++i) {
            EXPECT_DOUBLE_EQ(static_cast<double>(j), grid.v(i, j));
        }
    }
}

TEST(FaceCenteredGrid2, DivergenceAtCellCenter) {
    FaceCenteredGrid2 grid(5, 8, 2.0, 3.0);

    grid.fill(Vector2D(1.0, -2.0));

    for (size_t j = 0; j < grid.resolution().y; ++j) {
        for (size_t i = 0; i < grid.resolution().x; ++i) {
            EXPECT_DOUBLE_EQ(0.0, grid.divergenceAtCellCenter(i, j));
        }
    }

    grid.fill([](const Vector2D& x) { return x; });

    for (size_t j = 0; j < grid.resolution().y; ++j) {
        for (size_t i = 0; i < grid.resolution().x; ++i) {
            EXPECT_NEAR(2.0, grid.divergenceAtCellCenter(i, j), 1e-6);
        }
    }
}

TEST(FaceCenteredGrid2, CurlAtCellCenter) {
    FaceCenteredGrid2 grid(5, 8, 2.0, 3.0);

    grid.fill(Vector2D(1.0, -2.0));

    for (size_t j = 0; j < grid.resolution().y; ++j) {
        for (size_t i = 0; i < grid.resolution().x; ++i) {
            EXPECT_DOUBLE_EQ(0.0, grid.curlAtCellCenter(i, j));
        }
    }

    grid.fill([](const Vector2D& x) { return Vector2D(-x.y, x.x); });

    for (size_t j = 1; j < grid.resolution().y - 1; ++j) {
        for (size_t i = 1; i < grid.resolution().x - 1; ++i) {
            EXPECT_NEAR(2.0, grid.curlAtCellCenter(i, j), 1e-6);
        }
    }
}

TEST(FaceCenteredGrid2, ValueAtCellCenter) {
    FaceCenteredGrid2 grid(5, 8, 2.0, 3.0);
    grid.fill([&](const Vector2D& x) {
        return Vector2D(3.0 * x.y + 1.0, 5.0 * x.x + 7.0);
    });

    auto pos = grid.cellCenterPosition();
    grid.forEachCellIndex([&](size_t i, size_t j) {
        Vector2D val = grid.valueAtCellCenter(i, j);
        Vector2D x = pos(i, j);
        Vector2D expected = Vector2D(3.0 * x.y + 1.0, 5.0 * x.x + 7.0);
        EXPECT_NEAR(expected.x, val.x, 1e-6);
        EXPECT_NEAR(expected.y, val.y, 1e-6);
    });
}

TEST(FaceCenteredGrid2, Sample) {
    FaceCenteredGrid2 grid(5, 8, 2.0, 3.0);
    grid.fill([&](const Vector2D& x) {
        return Vector2D(3.0 * x.y + 1.0, 5.0 * x.x + 7.0);
    });

    auto pos = grid.cellCenterPosition();
    grid.forEachCellIndex([&](size_t i, size_t j) {
        Vector2D x = pos(i, j);
        Vector2D val = grid.sample(x);
        Vector2D expected = Vector2D(3.0 * x.y + 1.0, 5.0 * x.x + 7.0);
        EXPECT_NEAR(expected.x, val.x, 1e-6);
        EXPECT_NEAR(expected.y, val.y, 1e-6);
    });
}

TEST(FaceCenteredGrid2, Builder) {
    {
        auto builder = FaceCenteredGrid2::builder();

        auto grid = builder.build(
            Size2(5, 2),
            Vector2D(2.0, 4.0),
            Vector2D(-1.0, 2.0),
            Vector2D(3.0, 5.0));
        EXPECT_EQ(Size2(5, 2), grid->resolution());
        EXPECT_EQ(Vector2D(2.0, 4.0), grid->gridSpacing());
        EXPECT_EQ(Vector2D(-1.0, 2.0), grid->origin());

        auto faceCenteredGrid
            = std::dynamic_pointer_cast<FaceCenteredGrid2>(grid);
        EXPECT_TRUE(faceCenteredGrid != nullptr);

        faceCenteredGrid->forEachUIndex(
            [&faceCenteredGrid](size_t i, size_t j) {
                EXPECT_DOUBLE_EQ(3.0, faceCenteredGrid->u(i, j));
            });
        faceCenteredGrid->forEachVIndex(
            [&faceCenteredGrid](size_t i, size_t j) {
                EXPECT_DOUBLE_EQ(5.0, faceCenteredGrid->v(i, j));
            });
    }

    {
        auto grid = FaceCenteredGrid2::builder()
            .withResolution(5, 2)
            .withGridSpacing(2, 4)
            .withOrigin(-1, 2)
            .withInitialValue(3, 5)
            .build();

        EXPECT_EQ(Size2(5, 2), grid.resolution());
        EXPECT_EQ(Vector2D(2.0, 4.0), grid.gridSpacing());
        EXPECT_EQ(Vector2D(-1.0, 2.0), grid.origin());

        grid.forEachUIndex(
            [&](size_t i, size_t j) {
                EXPECT_DOUBLE_EQ(3.0, grid.u(i, j));
            });
        grid.forEachVIndex(
            [&](size_t i, size_t j) {
                EXPECT_DOUBLE_EQ(5.0, grid.v(i, j));
            });
    }
}

TEST(FaceCenteredGrid2, Serialization) {
    FaceCenteredGrid2 grid1(5, 4, 1.0, 2.0, -5.0, 3.0);
    grid1.fill([&] (const Vector2D& pt) {
        return Vector2D(pt.x, pt.y);
    });

    // Serialize to in-memoery stream
    std::vector<uint8_t> buffer1;
    grid1.serialize(&buffer1);

    // Deserialize to non-zero array
    FaceCenteredGrid2 grid2(1, 2, 0.5, 1.0, 0.5, 2.0);
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
    EXPECT_EQ(6u, grid2.uSize().x);
    EXPECT_EQ(4u, grid2.uSize().y);
    EXPECT_EQ(5u, grid2.vSize().x);
    EXPECT_EQ(5u, grid2.vSize().y);
    EXPECT_DOUBLE_EQ(-5.0, grid2.uOrigin().x);
    EXPECT_DOUBLE_EQ(4.0, grid2.uOrigin().y);
    EXPECT_DOUBLE_EQ(-4.5, grid2.vOrigin().x);
    EXPECT_DOUBLE_EQ(3.0, grid2.vOrigin().y);

    grid1.forEachUIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(grid1.u(i, j), grid2.u(i, j));
    });

    grid1.forEachVIndex([&] (size_t i, size_t j) {
        EXPECT_DOUBLE_EQ(grid1.v(i, j), grid2.v(i, j));
    });

    // Serialize zero-sized array
    FaceCenteredGrid2 grid3;
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
    EXPECT_EQ(0u, grid2.uSize().x);
    EXPECT_EQ(0u, grid2.uSize().y);
    EXPECT_EQ(0u, grid2.vSize().x);
    EXPECT_EQ(0u, grid2.vSize().y);
    EXPECT_DOUBLE_EQ(0.0, grid2.uOrigin().x);
    EXPECT_DOUBLE_EQ(0.5, grid2.uOrigin().y);
    EXPECT_DOUBLE_EQ(0.5, grid2.vOrigin().x);
    EXPECT_DOUBLE_EQ(0.0, grid2.vOrigin().y);
}
