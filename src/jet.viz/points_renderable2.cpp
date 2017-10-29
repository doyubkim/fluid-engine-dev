// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/points_renderable2.h>
#include <jet.viz/renderer.h>

using namespace jet;
using namespace viz;

PointsRenderable2::PointsRenderable2(Renderer* renderer) {
    _renderer = renderer;
    _shader = renderer->createPresetShader("point_sprite");
}

PointsRenderable2::~PointsRenderable2() {}

void PointsRenderable2::render(Renderer* renderer) {
    if (_vertexBuffer != nullptr && _shader != nullptr) {
        renderer->bindShader(_shader);
        renderer->bindVertexBuffer(_vertexBuffer);
        renderer->setPrimitiveType(PrimitiveType::Points);
        renderer->draw(_vertexBuffer->numberOfVertices());
        renderer->unbindVertexBuffer(_vertexBuffer);
        renderer->unbindShader(_shader);
    }
}

void PointsRenderable2::setPositions(const Vector2F* positions,
                                     size_t numberOfParticles) {
    std::vector<VertexPosition3Color4> vertices(numberOfParticles);
    for (size_t i = 0; i < numberOfParticles; ++i) {
        VertexPosition3Color4& vertex = vertices[i];
        vertex.x = positions[i].x;
        vertex.y = positions[i].y;
        vertex.z = 0.f;
        vertex.r = 1.f;
        vertex.g = 1.f;
        vertex.b = 1.f;
        vertex.a = 1.f;
    }

    updateVertexBuffer(vertices);
}

void PointsRenderable2::setPositionsAndColors(const Vector2F* positions,
                                              const Color* colors,
                                              size_t numberOfParticles) {
    std::vector<VertexPosition3Color4> vertices(numberOfParticles);
    for (size_t i = 0; i < numberOfParticles; ++i) {
        VertexPosition3Color4& vertex = vertices[i];
        vertex.x = positions[i].x;
        vertex.y = positions[i].y;
        vertex.z = 0.f;
        vertex.r = colors[i].r;
        vertex.g = colors[i].g;
        vertex.b = colors[i].b;
        vertex.a = colors[i].a;
    }

    updateVertexBuffer(vertices);
}

float PointsRenderable2::radius() const { return _radius; }

void PointsRenderable2::setRadius(float radius) {
    _radius = radius;
    _shader->setUserRenderParameter("Radius", radius);
}

void PointsRenderable2::updateVertexBuffer(
    const std::vector<VertexPosition3Color4>& vertices) {
    if (_vertexBuffer == nullptr) {
        _vertexBuffer = _renderer->createVertexBuffer(
            _shader, (const float*)vertices.data(), vertices.size());
    } else {
        _vertexBuffer->resize(_shader, (const float*)vertices.data(),
                              vertices.size());
    }
}
