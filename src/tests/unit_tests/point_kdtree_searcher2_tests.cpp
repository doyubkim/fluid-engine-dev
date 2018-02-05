// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array1.h>
#include <jet/array2.h>
#include <jet/bounding_box2.h>
#include <jet/point_kdtree_searcher2.h>
#include <jet/triangle_point_generator.h>

#include <gtest/gtest.h>

using namespace jet;

TEST(PointKdTreeSearcher2, ForEachNearbyPoint) {
    Array1<Vector2D> points = {Vector2D(1, 3), Vector2D(2, 5), Vector2D(-1, 3)};

    PointKdTreeSearcher2 searcher;
    searcher.build(points.accessor());

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

TEST(PointKdTreeSearcher2, ForEachNearbyPointEmpty) {
    Array1<Vector2D> points;

    PointKdTreeSearcher2 searcher;
    searcher.build(points.accessor());

    searcher.forEachNearbyPoint(Vector2D(0, 0), std::sqrt(10.0),
                                [](size_t, const Vector2D&) {});
}

TEST(PointKdTreeSearcher2, Serialization) {
    Array1<Vector2D> points = {Vector2D(1, 3), Vector2D(2, 5), Vector2D(-1, 3)};

    PointKdTreeSearcher2 searcher;
    searcher.build(points);

    std::vector<uint8_t> buffer;
    searcher.serialize(&buffer);

    EXPECT_GT(buffer.size(), 0U);

    PointKdTreeSearcher2 searcher2;
    searcher2.deserialize(buffer);

    std::vector<uint8_t> buffer2;
    searcher2.serialize(&buffer2);

    ASSERT_EQ(buffer.size(), buffer2.size());

    for (size_t i = 0; i < buffer.size(); ++i) {
        EXPECT_EQ(buffer[i], buffer2[i]);
    }
}
