// Copyright (c) 2016 Doyub Kim

#include <jet/array1.h>
#include <jet/point_simple_list_searcher3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(PointSimpleListSearcher3, ForEachNearbyPoint) {
    Array1<Vector3D> points = {
        Vector3D(0, 1, 3),
        Vector3D(2, 5, 4),
        Vector3D(-1, 3, 0)
    };

    PointSimpleListSearcher3 searcher;
    searcher.build(points.accessor());

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
        });
}
