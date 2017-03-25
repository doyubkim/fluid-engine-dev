// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/triangle3.h>
#include <gtest/gtest.h>

using namespace jet;

TEST(Triangle3, Constructors) {
    Triangle3 tri1;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(0.0, tri1.points[i][j]);
            EXPECT_DOUBLE_EQ(0.0, tri1.normals[i][j]);
        }
        for (int j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(0.0, tri1.uvs[i][j]);
        }
    }

    std::array<Vector3D, 3> points = {{
        Vector3D(1, 2, 3),
        Vector3D(4, 5, 6),
        Vector3D(7, 8, 9)
    }};
    std::array<Vector3D, 3> normals = {{
        Vector3D(1, 0, 0),
        Vector3D(0, 1, 0),
        Vector3D(0, 0, 1)
    }};
    std::array<Vector2D, 3> uvs = {{
        Vector2D(1, 0),
        Vector2D(0, 1),
        Vector2D(0.5, 0.5)
    }};

    Triangle3 tri2(points, normals, uvs);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(points[i][j], tri2.points[i][j]);
            EXPECT_DOUBLE_EQ(normals[i][j], tri2.normals[i][j]);
        }
        for (int j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(uvs[i][j], tri2.uvs[i][j]);
        }
    }

    Triangle3 tri3(tri2);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(points[i][j], tri3.points[i][j]);
            EXPECT_DOUBLE_EQ(normals[i][j], tri3.normals[i][j]);
        }
        for (int j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(uvs[i][j], tri3.uvs[i][j]);
        }
    }
}

TEST(Triangle3, Builder) {
    std::array<Vector3D, 3> points = {{
        Vector3D(1, 2, 3),
        Vector3D(4, 5, 6),
        Vector3D(7, 8, 9)
    }};
    std::array<Vector3D, 3> normals = {{
        Vector3D(1, 0, 0),
        Vector3D(0, 1, 0),
        Vector3D(0, 0, 1)
    }};
    std::array<Vector2D, 3> uvs = {{
        Vector2D(1, 0),
        Vector2D(0, 1),
        Vector2D(0.5, 0.5)
    }};

    Triangle3 tri = Triangle3::builder()
        .withPoints(points)
        .withNormals(normals)
        .withUvs(uvs)
        .build();

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(points[i][j], tri.points[i][j]);
            EXPECT_DOUBLE_EQ(normals[i][j], tri.normals[i][j]);
        }
        for (int j = 0; j < 2; ++j) {
            EXPECT_DOUBLE_EQ(uvs[i][j], tri.uvs[i][j]);
        }
    }
}
