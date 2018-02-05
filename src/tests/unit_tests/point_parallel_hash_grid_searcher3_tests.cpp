// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array1.h>
#include <jet/point_parallel_hash_grid_searcher3.h>
#include <gtest/gtest.h>
#include <vector>

using namespace jet;

TEST(PointParallelHashGridSearcher3, ForEachNearbyPoint) {
    Array1<Vector3D> points = {
        Vector3D(0, 1, 3),
        Vector3D(2, 5, 4),
        Vector3D(-1, 3, 0)
    };

    PointParallelHashGridSearcher3 searcher(4, 4, 4, std::sqrt(10));
    searcher.build(points.accessor());

    int cnt = 0;
    searcher.forEachNearbyPoint(
        Vector3D(0, 0, 0),
        std::sqrt(10.0),
        [&](size_t i, const Vector3D& pt) {
            EXPECT_TRUE(i == 0 || i == 2);

            if (i == 0) {
                EXPECT_EQ(points[0], pt);
            } else if (i == 2) {
                EXPECT_EQ(points[2], pt);
            }

            ++cnt;
        });
    EXPECT_EQ(2, cnt);
}

TEST(PointParallelHashGridSearcher3, ForEachNearbyPointEmpty) {
    Array1<Vector3D> points;

    PointParallelHashGridSearcher3 searcher(4, 4, 4, std::sqrt(10));
    searcher.build(points.accessor());

    searcher.forEachNearbyPoint(
        Vector3D(0, 0, 0),
        std::sqrt(10.0),
        [](size_t, const Vector3D&) {
        });
}

TEST(PointParallelHashGridSearcher3, CopyConstructor) {
    Array1<Vector3D> points = {
        Vector3D(0, 1, 3),
        Vector3D(2, 5, 4),
        Vector3D(-1, 3, 0)
    };

    PointParallelHashGridSearcher3 searcher(4, 4, 4, std::sqrt(10));
    searcher.build(points.accessor());

    PointParallelHashGridSearcher3 searcher2(searcher);
    int cnt = 0;
    searcher2.forEachNearbyPoint(
        Vector3D(0, 0, 0),
        std::sqrt(10.0),
        [&](size_t i, const Vector3D& pt) {
            EXPECT_TRUE(i == 0 || i == 2);

            if (i == 0) {
                EXPECT_EQ(points[0], pt);
            } else if (i == 2) {
                EXPECT_EQ(points[2], pt);
            }

            ++cnt;
        });
    EXPECT_EQ(2, cnt);
}

TEST(PointParallelHashGridSearcher3, Serialization) {
    Array1<Vector3D> points = {
        Vector3D(0, 1, 3),
        Vector3D(2, 5, 4),
        Vector3D(-1, 3, 0)
    };

    PointParallelHashGridSearcher3 searcher(4, 4, 4, std::sqrt(10));
    searcher.build(points.accessor());

    std::vector<uint8_t> buffer;
    searcher.serialize(&buffer);

    PointParallelHashGridSearcher3 searcher2(1, 1, 1, 1.0);
    searcher2.deserialize(buffer);

    int cnt = 0;
    searcher2.forEachNearbyPoint(
        Vector3D(0, 0, 0),
        std::sqrt(10.0),
        [&](size_t i, const Vector3D& pt) {
            EXPECT_TRUE(i == 0 || i == 2);

            if (i == 0) {
                EXPECT_EQ(points[0], pt);
            } else if (i == 2) {
                EXPECT_EQ(points[2], pt);
            }

            ++cnt;
        });
    EXPECT_EQ(2, cnt);
}
