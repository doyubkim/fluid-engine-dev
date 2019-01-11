// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet.gfx/points_renderable.h>
#include <jet.gfx/renderer.h>

namespace jet {
namespace gfx {

PointsRenderable::PointsRenderable(const ConstArrayView1<Vector3F>& positions,
                                   const ConstArrayView1<Vector4F>& colors,
                                   float radius) {
    update(positions, colors, radius);
}

void PointsRenderable::update(const ConstArrayView1<Vector3F>& positions,
                              const ConstArrayView1<Vector4F>& colors) {
    update(positions, colors, _radius);
}

void PointsRenderable::update(const ConstArrayView1<Vector3F>& positions,
                              const ConstArrayView1<Vector4F>& colors,
                              float radius) {
    // This function can be called from non-render thread.
    std::lock_guard<std::mutex> lock(_dataMutex);

    _vertices.resize(positions.length());
    for (size_t i = 0; i < positions.length(); ++i) {
        VertexPosition3Color4& vertex = _vertices[i];
        vertex.x = positions[i].x;
        vertex.y = positions[i].y;
        vertex.z = positions[i].z;
    }
    for (size_t i = 0; i < colors.length(); ++i) {
        VertexPosition3Color4& vertex = _vertices[i];
        vertex.r = colors[i].x;
        vertex.g = colors[i].y;
        vertex.b = colors[i].z;
        vertex.a = colors[i].w;
    }
    _radius = radius;

    invalidateGpuResources();
}

void PointsRenderable::onInitializeGpuResources(Renderer* renderer) {
    JET_ASSERT(renderer != nullptr);

    if (_vertices.isEmpty()) {
        _vertexBuffer = nullptr;
        return;
    }

    // This function is called from render thread.
    std::lock_guard<std::mutex> lock(_dataMutex);

    if (!_shader) {
        _shader = renderer->createPresetShader("points");
    }
    if (!_shader) {
        throw NotImplementedException(
            "This renderer does not have point_sprite shader as a preset.");
    }

    _shader->setUserRenderParameter("Radius", _radius);

    if (_vertexBuffer) {
        _vertexBuffer->resize(_shader, (const float*)_vertices.data(),
                              _vertices.length());
    } else {
        _vertexBuffer = renderer->createVertexBuffer(
            _shader, (const float*)_vertices.data(), _vertices.length());
    }
}

void PointsRenderable::onRender(Renderer* renderer) {
    if (_vertexBuffer) {
        renderer->bindShader(_shader);
        renderer->bindVertexBuffer(_vertexBuffer);
        renderer->setPrimitiveType(PrimitiveType::Points);
        renderer->draw(_vertexBuffer->numberOfVertices());
        renderer->unbindVertexBuffer(_vertexBuffer);
        renderer->unbindShader(_shader);
    }
}

}  // namespace gfx
}  // namespace jet
