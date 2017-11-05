// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/image_renderable.h>
#include <jet.viz/renderer.h>

using namespace jet;
using namespace viz;

ImageRenderable::ImageRenderable(Renderer* renderer) : _renderer(renderer) {
    _shader = renderer->createPresetShader("simple_texture2");

    std::array<VertexPosition3TexCoord2, 6> vertices;
    vertices[0].x = -0.5f;
    vertices[0].y = -0.5f;
    vertices[0].z = 0;
    vertices[0].u = 0;
    vertices[0].v = 0;
    vertices[1].x = -0.5f;
    vertices[1].y = 0.5f;
    vertices[1].z = 0;
    vertices[1].u = 0;
    vertices[1].v = 1;
    vertices[2].x = 0.5f;
    vertices[2].y = 0.5f;
    vertices[2].z = 0;
    vertices[2].u = 1;
    vertices[2].v = 1;

    vertices[3].x = -0.5f;
    vertices[3].y = -0.5f;
    vertices[3].z = 0;
    vertices[3].u = 0;
    vertices[3].v = 0;
    vertices[4].x = 0.5f;
    vertices[4].y = 0.5f;
    vertices[4].z = 0;
    vertices[4].u = 1;
    vertices[4].v = 1;
    vertices[5].x = 0.5f;
    vertices[5].y = -0.5f;
    vertices[5].z = 0;
    vertices[5].u = 1;
    vertices[5].v = 0;

    _vertexBuffer = renderer->createVertexBuffer(
        _shader, reinterpret_cast<const float*>(vertices.data()), 6);
}

void ImageRenderable::setImage(const ByteImage& image) {
    auto data = reinterpret_cast<const uint8_t*>(image.data());

    if (_texture != nullptr && image.size() == _texture->size()) {
        _texture->update(data);
    } else {
        _texture = _renderer->createTexture2(data, image.size());
    }
}

void ImageRenderable::setTextureSamplingMode(const TextureSamplingMode& mode) {
    _texture->setSamplingMode(mode);
}

void ImageRenderable::render(Renderer* renderer) {
    if (_shader != nullptr && _texture != nullptr &&
        _texture->size() != Size2()) {
        renderer->bindShader(_shader);
        renderer->bindVertexBuffer(_vertexBuffer);
        renderer->bindTexture(_texture, 0);
        renderer->setPrimitiveType(PrimitiveType::Triangles);

        renderer->draw(_vertexBuffer->numberOfVertices());

        renderer->unbindVertexBuffer(_vertexBuffer);
        renderer->unbindShader(_shader);
    }
}
