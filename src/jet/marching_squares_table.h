// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_JET_MARCHING_SQUARES_TABLE_H_
#define SRC_JET_MARCHING_SQUARES_TABLE_H_

namespace jet {

// VertexOffset lists the positions, relative to vertex0, of each of the 4
// vertices of a Rect
static const float vertexOffset2D[4][2] = {
    {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}
};

// EdgeConnection lists the index of the endpoint vertices for each of the 4
// edges of the Rect
static const int edgeConnection2D[4][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0}
};

// EdgeDirection lists the direction unit vector for each edge in the Rect
static const float edgeDirection2D[4][2] = {
    {1.0f, 0.0f}, {0.0f, 1.0f}, {-1.0f, 0.0f}, {0.0f, -1.0f}
};

// For any edge, if one vertex is inside of the surface and the other is outside
// of the surface then the edge intersects the surface
// For each of the 4 vertices of the Rect can be two possible states : either
// inside or outside of the surface
// For any cube there are 2^4=16 possible sets of vertex states
// This table lists the edges intersected by the surface for all 16 possible
// vertex states
// There are 4 edges.  For each entry in the table, if edge #n is intersected,
// then bit #n is set to 1
static const int squareEdgeFlags[16] = {
    0x000, 0x009, 0x003, 0x00a, 0x006, 0x00f, 0x005, 0x00c,
    0x00c, 0x005, 0x00f, 0x006, 0x00a, 0x003, 0x009, 0x000
};

// For each of the possible vertex states listed in RectEdgeFlags there is a
// specific triangulation of the edge intersection points.
// TriangleConnectionTable lists all of them in the form of 0-4 edge triples
// with the list terminated by the invalid value -1.
// For example: TriangleConnectionTable[3] list the 2 triangles formed when
// corner[0] and corner[1] are inside of the surface, but the rest of the cube
// is not.
// The notation of vertex is as follow.
//       6
// 3-----------2
// |           |
// |7          |5
// |           |
// |           |
// 0-----------1
//       4
// vertices at 0~1 nodes : 0~1
// vertices on 0~1 edges : 4~7
// Three verices compose a triangle.

static const int triangleConnectionTable2D[16][13] = {
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  1,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  5,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  7,  2,  6,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  1,  6,  1,  2,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  7,  7,  1,  6,  1,  2,  6, -1, -1, -1, -1 },
    {  7,  6,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  6,  0,  6,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  7,  6,  6,  7,  4,  6,  4,  5,  1,  5,  4, -1 },
    {  0,  6,  3,  0,  5,  6,  0,  1,  5, -1, -1, -1, -1 },
    {  7,  5,  3,  5,  2,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  0,  4,  3,  4,  5,  3,  5,  2, -1, -1, -1, -1 },
    {  2,  3,  7,  2,  7,  4,  2,  4,  1, -1, -1, -1, -1 },
    {  0,  1,  3,  1,  2,  3, -1, -1, -1, -1, -1, -1, -1 },
};

}  // namespace jet

#endif  // SRC_JET_MARCHING_SQUARES_TABLE_H_
