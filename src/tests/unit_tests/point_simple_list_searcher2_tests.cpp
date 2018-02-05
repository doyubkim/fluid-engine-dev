// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/array1.h>
#include <jet/point_simple_list_searcher2.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PointSimpleListSearcher2, ForEachNearbyPoint) {
    Array1<Vector2D> points = {
        Vector2D(1, 3),
        Vector2D(2, 5),
        Vector2D(-1, 3)
    };

    PointSimpleListSearcher2 searcher;
    searcher.build(points.accessor());

    searcher.forEachNearbyPoint(
        Vector2D(0, 0),
        std::sqrt(10.0),
        [&](size_t i, const Vector2D& pt) {
            EXPECT_TRUE(i == 0 || i == 2);

            if (i == 0) {
                EXPECT_EQ(points[0], pt);
            } else if (i == 2) {
                EXPECT_EQ(points[2], pt);
            }
        });
}
