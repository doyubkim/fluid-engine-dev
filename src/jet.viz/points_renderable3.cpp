// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/points_renderable3.h>
#include <jet.viz/renderer.h>

using namespace jet;
using namespace viz;

PointsRenderable3::PointsRenderable3(Renderer* renderer) {
    _renderer = renderer;
    _shader = renderer->createPresetShader("point_sprite");
}

PointsRenderable3::~PointsRenderable3() {}

void PointsRenderable3::render(Renderer* renderer) {
    if (_vertexBuffer != nullptr && _shader != nullptr) {
        renderer->bindShader(_shader);
        renderer->bindVertexBuffer(_vertexBuffer);
        renderer->setPrimitiveType(PrimitiveType::Points);
        renderer->draw(_vertexBuffer->numberOfVertices());
        renderer->unbindVertexBuffer(_vertexBuffer);
        renderer->unbindShader(_shader);
    }
}

void PointsRenderable3::setPositions(const Vector3F* positions,
                                     size_t numberOfVertices) {
    std::vector<VertexPosition3Color4> vertices(numberOfVertices);
    for (size_t i = 0; i < numberOfVertices; ++i) {
        VertexPosition3Color4& vertex = vertices[i];
        vertex.x = positions[i].x;
        vertex.y = positions[i].y;
        vertex.z = positions[i].z;
        vertex.r = 1.f;
        vertex.g = 1.f;
        vertex.b = 1.f;
        vertex.a = 1.f;
    }

    setPositionsAndColors(vertices.data(), vertices.size());
}

void PointsRenderable3::setPositions(ConstArrayView1<Vector3F> positions) {
    setPositions(positions.data(), positions.size());
}

void PointsRenderable3::setPositionsAndColors(const Vector3F* positions,
                                              const Color* colors,
                                              size_t numberOfVertices) {
    std::vector<VertexPosition3Color4> vertices(numberOfVertices);
    for (size_t i = 0; i < numberOfVertices; ++i) {
        VertexPosition3Color4& vertex = vertices[i];
        vertex.x = positions[i].x;
        vertex.y = positions[i].y;
        vertex.z = positions[i].z;
        vertex.r = colors[i].r;
        vertex.g = colors[i].g;
        vertex.b = colors[i].b;
        vertex.a = colors[i].a;
    }

    setPositionsAndColors(vertices.data(), vertices.size());
}

void PointsRenderable3::setPositionsAndColors(
    ConstArrayView1<Vector3F> positions, ConstArrayView1<Color> colors) {
    setPositionsAndColors(positions.data(), colors.data(), positions.size());
}

void PointsRenderable3::setPositionsAndColors(
    const VertexPosition3Color4* vertices, size_t numberOfVertices) {
    if (_vertexBuffer == nullptr) {
        _vertexBuffer = _renderer->createVertexBuffer(
            _shader, (const float*)vertices, numberOfVertices);
    } else {
        _vertexBuffer->resize(_shader, (const float*)vertices,
                              numberOfVertices);
    }
}

void PointsRenderable3::setPositionsAndColors(
    ConstArrayView1<VertexPosition3Color4> vertices) {
    setPositionsAndColors(vertices.data(), vertices.size());
}

VertexBuffer* PointsRenderable3::vertexBuffer() const {
    return _vertexBuffer.get();
}

float PointsRenderable3::radius() const { return _radius; }

void PointsRenderable3::setRadius(float radius) {
    _radius = radius;
    _shader->setUserRenderParameter("Radius", radius);
}
