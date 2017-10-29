// Copyright (c) 2017 Doyub Kim
//
// This code is adopted from Ingemar Rask and Johannes Schmid's work:
// https://graphics.ethz.ch/teaching/former/imagesynthesis_06/miniprojects/p3/
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet.viz/renderer.h>
#include <jet.viz/simple_volume_renderable.h>

#include <algorithm>

using namespace jet;
using namespace viz;

// 8 vertices of a cube
static const float cubeVertices[8][3] = {
    {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, 0.5f},
    {-0.5f, -0.5f, 0.5f},  {-0.5f, 0.5f, -0.5f}, {0.5f, 0.5f, -0.5f},
    {0.5f, 0.5f, 0.5f},    {-0.5f, 0.5f, 0.5f}};

// 12 edges of the cube
static const int cubeEdges[12][2] = {{0, 1}, {1, 2}, {3, 2}, {0, 3},
                                     {4, 5}, {5, 6}, {7, 6}, {4, 7},
                                     {0, 4}, {1, 5}, {2, 6}, {3, 7}};

static void getSliceVertices(float a, float b, float c, float d,
                             Vector3F* outputVertices, int& numberOfVertices) {
    numberOfVertices = 0;

    for (int i = 0; i < 12; i++) {
        const float* v = cubeVertices[cubeEdges[i][0]];
        Vector3F rayOrigin(v[0], v[1], v[2]);
        v = cubeVertices[cubeEdges[i][1]];
        Vector3F rayDirection = Vector3F(v[0], v[1], v[2]) - rayOrigin;

        // ray-plane intersection
        float t =
            -(a * rayOrigin[0] + b * rayOrigin[1] + c * rayOrigin[2] + d) /
            (a * rayDirection[0] + b * rayDirection[1] + c * rayDirection[2]);

        if ((t > 0.f) && (t < 1.f)) {
            Vector3F p = rayOrigin + rayDirection * t;
            outputVertices[numberOfVertices++] = p;
        }
    }
}

SimpleVolumeRenderable::SimpleVolumeRenderable(Renderer* renderer)
    : _renderer(renderer) {
    _shader = renderer->createPresetShader("simple_texture3");
    _shader->setUserRenderParameter(
        "Multiplier",
        Vector4F(_brightness, _brightness, _brightness, _density));
}

SimpleVolumeRenderable::~SimpleVolumeRenderable() {}

void SimpleVolumeRenderable::render(Renderer* renderer) {
    if (_shader != nullptr && _texture != nullptr &&
        _texture->size() != Size3()) {
        const Vector3D& currentLookAt =
            renderer->camera()->basicCameraState().lookAt;
        const Vector3D& currentOrigin =
            renderer->camera()->basicCameraState().origin;

        if (!_prevCameraLookAtDir.isSimilar(currentLookAt) ||
            !_prevCameraOrigin.isSimilar(currentOrigin)) {
            updateVertexBuffer(renderer);
        }

        renderer->bindShader(_shader);
        renderer->bindVertexBuffer(_vertexBuffer);
        renderer->bindTexture(_texture, 0);
        renderer->setPrimitiveType(PrimitiveType::Triangles);
        for (auto& indexBuffer : _indexBuffers) {
            renderer->bindIndexBuffer(indexBuffer);
            renderer->drawIndexed(indexBuffer->numberOfIndices());
            renderer->unbindIndexBuffer(indexBuffer);
        }
        renderer->unbindVertexBuffer(_vertexBuffer);
        renderer->unbindShader(_shader);

        _prevCameraLookAtDir = currentLookAt;
        _prevCameraOrigin = currentOrigin;
    }
}

void SimpleVolumeRenderable::setVolume(const float* data, const Size3& size) {
    if (_texture != nullptr && size == _texture->size()) {
        _texture->update(data);
    } else {
        _texture = _renderer->createTexture3(data, size);
    }
}

float SimpleVolumeRenderable::brightness() const { return _brightness; }

void SimpleVolumeRenderable::setBrightness(float newBrightness) {
    _shader->setUserRenderParameter(
        "Multiplier",
        Vector4F(_brightness, _brightness, _brightness, _density));
    _brightness = newBrightness;
}

float SimpleVolumeRenderable::density() const { return _density; }

void SimpleVolumeRenderable::setDensity(float newDensity) {
    _shader->setUserRenderParameter(
        "Multiplier",
        Vector4F(_brightness, _brightness, _brightness, _density));
    _density = newDensity;
}

float SimpleVolumeRenderable::stepSize() const { return _stepSize; }

void SimpleVolumeRenderable::setStepSize(float newStepSize) {
    _stepSize = newStepSize;
}

void SimpleVolumeRenderable::updateVertexBuffer(Renderer* renderer) {
    Vector3F viewPos =
        renderer->camera()->basicCameraState().origin.castTo<float>();
    Vector3F viewDir =
        renderer->camera()->basicCameraState().lookAt.castTo<float>();

    // Get closest/farmost cube vertex
    Vector3F closest(cubeVertices[0][0], cubeVertices[0][1],
                     cubeVertices[0][2]);
    Vector3F farmost(cubeVertices[0][0], cubeVertices[0][1],
                     cubeVertices[0][2]);
    float minDistSquared = (closest - viewPos).lengthSquared();
    float maxDistSquared = minDistSquared;

    for (int i = 1; i < 8; i++) {
        Vector3F v(cubeVertices[i][0], cubeVertices[i][1], cubeVertices[i][2]);
        float d = (v - viewPos).lengthSquared();
        if (minDistSquared > d) {
            minDistSquared = d;
            closest = v;
        } else if (maxDistSquared < d) {
            maxDistSquared = d;
            farmost = v;
        }
    }

    // Reset index buffers
    _indexBuffers.clear();

    // Create new vertex buffers
    std::vector<VertexPosition3TexCoord3> vertices;
    float dist = sqrtf(maxDistSquared) - sqrtf(minDistSquared);
    for (float z = dist; z >= 0.f; z -= _stepSize) {
        Vector3F midpoint = closest + z * viewDir;

        // Get plane equation
        float a, b, c, d;
        a = viewDir.x;
        b = viewDir.y;
        c = viewDir.z;
        d = -(a * midpoint.x + b * midpoint.y + c * midpoint.z);

        // Get intersection verts
        Vector3F intersectingPoints[6];
        int numberOfIntersectingPoints;
        getSliceVertices(a, b, c, d, intersectingPoints,
                         numberOfIntersectingPoints);

        // Sort vertices CW
        std::sort(intersectingPoints,
                  intersectingPoints + numberOfIntersectingPoints,
                  [&](const Vector3F& a, const Vector3F& b) {
                      Vector3F va = a - intersectingPoints[0];
                      Vector3F vb = b - intersectingPoints[0];
                      return viewDir.dot(va.cross(vb)) > 0;
                  });

        // Vertex buffer
        if (numberOfIntersectingPoints >= 3) {
            // Add vertices
            for (int i = 0; i < numberOfIntersectingPoints; ++i) {
                VertexPosition3TexCoord3 vertex;
                vertex.x = intersectingPoints[i].x;
                vertex.y = intersectingPoints[i].y;
                vertex.z = intersectingPoints[i].z;
                vertex.u = intersectingPoints[i].x + 0.5f;
                vertex.v = intersectingPoints[i].y + 0.5f;
                vertex.w = intersectingPoints[i].z + 0.5f;
                vertices.push_back(vertex);
            }
        }

        _vertexBuffer = renderer->createVertexBuffer(
            _shader, reinterpret_cast<const float*>(vertices.data()),
            vertices.size());

        // Index buffer
        if (numberOfIntersectingPoints >= 3) {
            // Build indices
            std::vector<uint32_t> indices;
            for (int i = 0; i < numberOfIntersectingPoints - 2; ++i) {
                uint32_t baseIndex =
                    static_cast<uint32_t>(vertices.size());
                indices.push_back(baseIndex);
                indices.push_back(baseIndex + i + 1);
                indices.push_back(baseIndex + i + 2);
            }

            _indexBuffers.push_back(renderer->createIndexBuffer(
                _vertexBuffer,
                reinterpret_cast<const uint32_t*>(indices.data()),
                indices.size()));
        }
    }
}
