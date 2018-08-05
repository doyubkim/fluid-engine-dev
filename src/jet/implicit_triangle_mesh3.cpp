// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/implicit_triangle_mesh3.h>
#include <jet/triangle_mesh_to_sdf.h>

using namespace jet;

ImplicitTriangleMesh3::ImplicitTriangleMesh3(const TriangleMesh3Ptr& mesh,
                                             size_t resolutionX, double margin,
                                             const Transform3& transform,
                                             bool isNormalFlipped)
    : ImplicitSurface3(transform, isNormalFlipped), _mesh(mesh) {
    if (mesh->numberOfTriangles() > 0 && mesh->numberOfPoints() > 0) {
        BoundingBox3D box = _mesh->boundingBox();
        Vector3D scale(box.width(), box.height(), box.depth());
        box.lowerCorner -= margin * scale;
        box.upperCorner += margin * scale;
        size_t resolutionY = static_cast<size_t>(
            std::ceil(resolutionX * box.height() / box.width()));
        size_t resolutionZ = static_cast<size_t>(
            std::ceil(resolutionX * box.depth() / box.width()));

        double dx = box.width() / resolutionX;

        _grid = std::make_shared<VertexCenteredScalarGrid3>();
        _grid->resize(resolutionX, resolutionY, resolutionZ, dx, dx, dx,
                      box.lowerCorner.x, box.lowerCorner.y, box.lowerCorner.z);

        triangleMeshToSdf(*_mesh, _grid.get());

        _customImplicitSurface =
            CustomImplicitSurface3::builder()
                .withSignedDistanceFunction([&](const Vector3D& pt) -> double {
                    return _grid->sample(pt);
                })
                .withDomain(_grid->boundingBox())
                .withResolution(dx)
                .makeShared();
    } else {
        // Empty mesh -- return big/uniform number
        _customImplicitSurface =
            CustomImplicitSurface3::builder()
                .withSignedDistanceFunction(
                    [&](const Vector3D&) -> double { return kMaxD; })
                .makeShared();
    }
}

ImplicitTriangleMesh3::~ImplicitTriangleMesh3() {}

Vector3D ImplicitTriangleMesh3::closestPointLocal(
    const Vector3D& otherPoint) const {
    return _customImplicitSurface->closestPoint(otherPoint);
}

double ImplicitTriangleMesh3::closestDistanceLocal(
    const Vector3D& otherPoint) const {
    return _customImplicitSurface->closestDistance(otherPoint);
}

bool ImplicitTriangleMesh3::intersectsLocal(const Ray3D& ray) const {
    return _customImplicitSurface->intersects(ray);
}

BoundingBox3D ImplicitTriangleMesh3::boundingBoxLocal() const {
    return _mesh->boundingBox();
}

Vector3D ImplicitTriangleMesh3::closestNormalLocal(
    const Vector3D& otherPoint) const {
    return _customImplicitSurface->closestNormal(otherPoint);
}

double ImplicitTriangleMesh3::signedDistanceLocal(
    const Vector3D& otherPoint) const {
    return _customImplicitSurface->signedDistance(otherPoint);
}

SurfaceRayIntersection3 ImplicitTriangleMesh3::closestIntersectionLocal(
    const Ray3D& ray) const {
    return _customImplicitSurface->closestIntersection(ray);
}

ImplicitTriangleMesh3::Builder ImplicitTriangleMesh3::builder() {
    return ImplicitTriangleMesh3::Builder();
}

const VertexCenteredScalarGrid3Ptr& ImplicitTriangleMesh3::grid() const {
    return _grid;
}

ImplicitTriangleMesh3::Builder&
ImplicitTriangleMesh3::Builder::withTriangleMesh(const TriangleMesh3Ptr& mesh) {
    _mesh = mesh;
    return *this;
}

ImplicitTriangleMesh3::Builder& ImplicitTriangleMesh3::Builder::withResolutionX(
    size_t resolutionX) {
    _resolutionX = resolutionX;
    return *this;
}

ImplicitTriangleMesh3::Builder& ImplicitTriangleMesh3::Builder::withMargin(
    double margin) {
    _margin = margin;
    return *this;
}

ImplicitTriangleMesh3 ImplicitTriangleMesh3::Builder::build() const {
    return ImplicitTriangleMesh3(_mesh, _resolutionX, _margin, _transform,
                                 _isNormalFlipped);
}

ImplicitTriangleMesh3Ptr ImplicitTriangleMesh3::Builder::makeShared() const {
    return std::shared_ptr<ImplicitTriangleMesh3>(
        new ImplicitTriangleMesh3(_mesh, _resolutionX, _margin, _transform,
                                  _isNormalFlipped),
        [](ImplicitTriangleMesh3* obj) { delete obj; });
}
