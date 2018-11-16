// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <gtest/gtest.h>
#include <jet/array.h>
#include <jet/bounding_box.h>
#include <jet/point_hash_grid_searcher.h>
#include <jet/point_hash_grid_utils.h>
#include <jet/point_parallel_hash_grid_searcher.h>
#include <jet/triangle_point_generator.h>

using namespace jet;

TEST(PointHashGridSearcher2, ForEachNearbyPoint) {
    Array1<Vector2D> points = {Vector2D(1, 3), Vector2D(2, 5), Vector2D(-1, 3)};

    PointHashGridSearcher2 searcher({4, 4}, 2.0 * std::sqrt(10));
    searcher.build(points);

    searcher.forEachNearbyPoint(Vector2D(0, 0), std::sqrt(10.0),
                                [&points](size_t i, const Vector2D& pt) {
                                    EXPECT_TRUE(i == 0 || i == 2);

                                    if (i == 0) {
                                        EXPECT_EQ(points[0], pt);
                                    } else if (i == 2) {
                                        EXPECT_EQ(points[2], pt);
                                    }
                                });
}

TEST(PointHashGridSearcher2, ForEachNearbyPointEmpty) {
    Array1<Vector2D> points;

    PointHashGridSearcher2 searcher({4, 4}, 2.0 * std::sqrt(10));
    searcher.build(points);

    searcher.forEachNearbyPoint(Vector2D(0, 0), std::sqrt(10.0),
                                [](size_t, const Vector2D&) {});
}

TEST(PointParallelHashGridSearcher2, Build) {
    Array1<Vector2D> points;
    TrianglePointGenerator pointsGenerator;
    BoundingBox2D bbox(Vector2D(0, 0), Vector2D(1, 1));
    double spacing = 0.1;

    pointsGenerator.generate(bbox, spacing, &points);

    PointHashGridSearcher2 pointSearcher({4, 4}, 0.18);
    pointSearcher.build(points);

    Array2<size_t> grid(4, 4);

    for (size_t j = 0; j < grid.size().y; ++j) {
        for (size_t i = 0; i < grid.size().x; ++i) {
            size_t key = PointHashGridUtils2::getHashKeyFromBucketIndex(
                Vector2Z(static_cast<ssize_t>(i), static_cast<ssize_t>(j)),
                Vector2Z(4, 4));
            size_t value = pointSearcher.buckets()[key].length();
            grid(i, j) = value;
        }
    }

    PointParallelHashGridSearcher2 parallelSearcher({4, 4}, 0.18);
    parallelSearcher.build(points);

    for (size_t j = 0; j < grid.size().y; ++j) {
        for (size_t i = 0; i < grid.size().x; ++i) {
            size_t key = PointHashGridUtils2::getHashKeyFromBucketIndex(
                Vector2Z(static_cast<ssize_t>(i), static_cast<ssize_t>(j)),
                Vector2Z(4, 4));
            size_t start = parallelSearcher.startIndexTable()[key];
            size_t end = parallelSearcher.endIndexTable()[key];
            size_t value = end - start;
            EXPECT_EQ(grid(i, j), value);
        }
    }
}
