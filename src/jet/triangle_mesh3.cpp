// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/parallel.h>
#include <jet/triangle_mesh3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#define TINYOBJLOADER_USE_DOUBLE
#include <tiny_obj_loader.h>

#include <fstream>

using namespace jet;

#define SZ(x) static_cast<size_t>(x)

inline std::ostream& operator<<(std::ostream& strm, const Vector2D& v) {
    strm << v.x << ' ' << v.y;
    return strm;
}

inline std::ostream& operator<<(std::ostream& strm, const Vector3D& v) {
    strm << v.x << ' ' << v.y << ' ' << v.z;
    return strm;
}

TriangleMesh3::TriangleMesh3(const Transform3& transform_,
                             bool isNormalFlipped_)
    : Surface3(transform_, isNormalFlipped_) {}

TriangleMesh3::TriangleMesh3(const PointArray& points,
                             const NormalArray& normals, const UvArray& uvs,
                             const IndexArray& pointIndices,
                             const IndexArray& normalIndices,
                             const IndexArray& uvIndices,
                             const Transform3& transform_,
                             bool isNormalFlipped_)
    : Surface3(transform_, isNormalFlipped_),
      _points(points),
      _normals(normals),
      _uvs(uvs),
      _pointIndices(pointIndices),
      _normalIndices(normalIndices),
      _uvIndices(uvIndices) {}

TriangleMesh3::TriangleMesh3(const TriangleMesh3& other) : Surface3(other) {
    set(other);
}

void TriangleMesh3::updateQueryEngine() { buildBvh(); }

Vector3D TriangleMesh3::closestPointLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [this](const size_t& triIdx, const Vector3D& pt) {
        Triangle3 tri = triangle(triIdx);
        return tri.closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return triangle(*queryResult.item).closestPoint(otherPoint);
}

Vector3D TriangleMesh3::closestNormalLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [this](const size_t& triIdx, const Vector3D& pt) {
        Triangle3 tri = triangle(triIdx);
        return tri.closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    //    printf("%zu\n", *queryResult.item);
    return triangle(*queryResult.item).closestNormal(otherPoint);
}

SurfaceRayIntersection3 TriangleMesh3::closestIntersectionLocal(
    const Ray3D& ray) const {
    buildBvh();

    const auto testFunc = [this](const size_t& triIdx, const Ray3D& ray) {
        Triangle3 tri = triangle(triIdx);
        SurfaceRayIntersection3 result = tri.closestIntersection(ray);
        return result.distance;
    };

    const auto queryResult = _bvh.closestIntersection(ray, testFunc);
    SurfaceRayIntersection3 result;
    result.distance = queryResult.distance;
    result.isIntersecting = queryResult.item != nullptr;
    if (queryResult.item != nullptr) {
        result.point = ray.pointAt(queryResult.distance);
        result.normal = triangle(*queryResult.item).closestNormal(result.point);
    }
    return result;
}

BoundingBox3D TriangleMesh3::boundingBoxLocal() const {
    buildBvh();

    return _bvh.boundingBox();
}

bool TriangleMesh3::intersectsLocal(const Ray3D& ray) const {
    buildBvh();

    const auto testFunc = [this](const size_t& triIdx, const Ray3D& ray) {
        Triangle3 tri = triangle(triIdx);
        return tri.intersects(ray);
    };

    return _bvh.intersects(ray, testFunc);
}

double TriangleMesh3::closestDistanceLocal(const Vector3D& otherPoint) const {
    buildBvh();

    const auto distanceFunc = [this](const size_t& triIdx, const Vector3D& pt) {
        Triangle3 tri = triangle(triIdx);
        return tri.closestDistance(pt);
    };

    const auto queryResult = _bvh.nearest(otherPoint, distanceFunc);
    return queryResult.distance;
}

void TriangleMesh3::clear() {
    _points.clear();
    _normals.clear();
    _uvs.clear();
    _pointIndices.clear();
    _normalIndices.clear();
    _uvIndices.clear();

    invalidateBvh();
}

void TriangleMesh3::set(const TriangleMesh3& other) {
    _points.copyFrom(other._points);
    _normals.copyFrom(other._normals);
    _uvs.copyFrom(other._uvs);
    _pointIndices.copyFrom(other._pointIndices);
    _normalIndices.copyFrom(other._normalIndices);
    _uvIndices.copyFrom(other._uvIndices);

    invalidateBvh();
}

void TriangleMesh3::swap(TriangleMesh3& other) {
    _points.swap(other._points);
    _normals.swap(other._normals);
    _uvs.swap(other._uvs);
    _pointIndices.swap(other._pointIndices);
    _normalIndices.swap(other._normalIndices);
    _uvIndices.swap(other._uvIndices);
}

double TriangleMesh3::area() const {
    double a = 0;
    for (size_t i = 0; i < numberOfTriangles(); ++i) {
        Triangle3 tri = triangle(i);
        a += tri.area();
    }
    return a;
}

double TriangleMesh3::volume() const {
    double vol = 0;
    for (size_t i = 0; i < numberOfTriangles(); ++i) {
        Triangle3 tri = triangle(i);
        vol += tri.points[0].dot(tri.points[1].cross(tri.points[2])) / 6.f;
    }
    return vol;
}

const Vector3D& TriangleMesh3::point(size_t i) const { return _points[i]; }

Vector3D& TriangleMesh3::point(size_t i) {
    invalidateBvh();
    return _points[i];
}

const Vector3D& TriangleMesh3::normal(size_t i) const { return _normals[i]; }

Vector3D& TriangleMesh3::normal(size_t i) { return _normals[i]; }

const Vector2D& TriangleMesh3::uv(size_t i) const { return _uvs[i]; }

Vector2D& TriangleMesh3::uv(size_t i) { return _uvs[i]; }

const Vector3UZ& TriangleMesh3::pointIndex(size_t i) const {
    return _pointIndices[i];
}

Vector3UZ& TriangleMesh3::pointIndex(size_t i) { return _pointIndices[i]; }

const Vector3UZ& TriangleMesh3::normalIndex(size_t i) const {
    return _normalIndices[i];
}

Vector3UZ& TriangleMesh3::normalIndex(size_t i) { return _normalIndices[i]; }

const Vector3UZ& TriangleMesh3::uvIndex(size_t i) const { return _uvIndices[i]; }

Vector3UZ& TriangleMesh3::uvIndex(size_t i) { return _uvIndices[i]; }

Triangle3 TriangleMesh3::triangle(size_t i) const {
    Triangle3 tri;
    for (int j = 0; j < 3; j++) {
        tri.points[j] = _points[_pointIndices[i][j]];
        if (hasUvs()) {
            tri.uvs[j] = _uvs[_uvIndices[i][j]];
        }
    }

    Vector3D n = tri.faceNormal();

    for (int j = 0; j < 3; j++) {
        if (hasNormals()) {
            tri.normals[j] = _normals[_normalIndices[i][j]];
        } else {
            tri.normals[j] = n;
        }
    }

    return tri;
}

size_t TriangleMesh3::numberOfPoints() const { return _points.length(); }

size_t TriangleMesh3::numberOfNormals() const { return _normals.length(); }

size_t TriangleMesh3::numberOfUvs() const { return _uvs.length(); }

size_t TriangleMesh3::numberOfTriangles() const { return _pointIndices.length(); }

bool TriangleMesh3::hasNormals() const { return _normals.length() > 0; }

bool TriangleMesh3::hasUvs() const { return _uvs.length() > 0; }

void TriangleMesh3::addPoint(const Vector3D& pt) { _points.append(pt); }

void TriangleMesh3::addNormal(const Vector3D& n) { _normals.append(n); }

void TriangleMesh3::addUv(const Vector2D& t) { _uvs.append(t); }

void TriangleMesh3::addPointTriangle(const Vector3UZ& newPointIndices) {
    _pointIndices.append(newPointIndices);
    invalidateBvh();
}

void TriangleMesh3::addNormalTriangle(const Point3UI& newNormalIndices) {
    _normalIndices.append(newNormalIndices);

    invalidateBvh();
}

void TriangleMesh3::addUvTriangle(const Point3UI& newUvIndices) {
    _uvIndices.append(newUvIndices);

    invalidateBvh();
}

void TriangleMesh3::addPointNormalTriangle(const Point3UI& newPointIndices,
                                           const Point3UI& newNormalIndices) {
    _pointIndices.append(newPointIndices);
    _normalIndices.append(newNormalIndices);

    invalidateBvh();
}

void TriangleMesh3::addPointUvNormalTriangle(const Point3UI& newPointIndices,
                                             const Point3UI& newUvIndices,
                                             const Point3UI& newNormalIndices) {
    _pointIndices.append(newPointIndices);
    _normalIndices.append(newNormalIndices);
    _uvIndices.append(newUvIndices);

    invalidateBvh();
}

void TriangleMesh3::addPointUvTriangle(const Point3UI& newPointIndices,
                                       const Point3UI& newUvIndices) {
    _pointIndices.append(newPointIndices);
    _uvIndices.append(newUvIndices);

    invalidateBvh();
}

void TriangleMesh3::addTriangle(const Triangle3& tri) {
    size_t vStart = _points.length();
    size_t nStart = _normals.length();
    size_t tStart = _uvs.length();
    Vector3UZ newPointIndices;
    Vector3UZ newNormalIndices;
    Vector3UZ newUvIndices;
    for (size_t i = 0; i < 3; i++) {
        _points.append(tri.points[i]);
        _normals.append(tri.normals[i]);
        _uvs.append(tri.uvs[i]);
        newPointIndices[i] = vStart + i;
        newNormalIndices[i] = nStart + i;
        newUvIndices[i] = tStart + i;
    }
    _pointIndices.append(newPointIndices);
    _normalIndices.append(newNormalIndices);
    _uvIndices.append(newUvIndices);

    invalidateBvh();
}

void TriangleMesh3::setFaceNormal() {
    _normals.resize(_points.length());
    _normalIndices.copyFrom(_pointIndices);

    for (size_t i = 0; i < numberOfTriangles(); ++i) {
        Triangle3 tri = triangle(i);
        Vector3D n = tri.faceNormal();
        Vector3UZ f = _pointIndices[i];
        _normals[f.x] = n;
        _normals[f.y] = n;
        _normals[f.z] = n;
    }
}

void TriangleMesh3::setAngleWeightedVertexNormal() {
    _normals.clear();
    _normalIndices.clear();

    Array1<double> angleWeights(_points.length());
    Vector3DArray pseudoNormals(_points.length());

    for (size_t i = 0; i < _points.length(); ++i) {
        angleWeights[i] = 0;
        pseudoNormals[i] = Vector3D();
    }

    for (size_t i = 0; i < numberOfTriangles(); ++i) {
        Vector3D pts[3];
        Vector3D normal, e0, e1;
        double cosangle, angle;
        size_t idx[3];

        // Quick references
        for (int j = 0; j < 3; j++) {
            idx[j] = _pointIndices[i][j];
            pts[j] = _points[idx[j]];
        }

        // Angle for point 0
        e0 = pts[1] - pts[0];
        e1 = pts[2] - pts[0];
        e0.normalize();
        e1.normalize();
        normal = e0.cross(e1);
        normal.normalize();
        cosangle = clamp(e0.dot(e1), -1.0, 1.0);
        angle = std::acos(cosangle);
        angleWeights[idx[0]] += angle;
        pseudoNormals[idx[0]] += angle * normal;

        // Angle for point 1
        e0 = pts[2] - pts[1];
        e1 = pts[0] - pts[1];
        e0.normalize();
        e1.normalize();
        normal = e0.cross(e1);
        normal.normalize();
        cosangle = clamp(e0.dot(e1), -1.0, 1.0);
        angle = std::acos(cosangle);
        angleWeights[idx[1]] += angle;
        pseudoNormals[idx[1]] += angle * normal;

        // Angle for point 2
        e0 = pts[0] - pts[2];
        e1 = pts[1] - pts[2];
        e0.normalize();
        e1.normalize();
        normal = e0.cross(e1);
        normal.normalize();
        cosangle = clamp(e0.dot(e1), -1.0, 1.0);
        angle = std::acos(cosangle);
        angleWeights[idx[2]] += angle;
        pseudoNormals[idx[2]] += angle * normal;
    }

    for (size_t i = 0; i < _points.length(); ++i) {
        if (angleWeights[i] > 0) {
            pseudoNormals[i] /= angleWeights[i];
        }
    }

    std::swap(pseudoNormals, _normals);
    _normalIndices.copyFrom(_pointIndices);
}

void TriangleMesh3::scale(double factor) {
    parallelFor(kZeroSize, numberOfPoints(),
                [this, factor](size_t i) { _points[i] *= factor; });
    invalidateBvh();
}

void TriangleMesh3::translate(const Vector3D& t) {
    parallelFor(kZeroSize, numberOfPoints(),
                [this, t](size_t i) { _points[i] += t; });
    invalidateBvh();
}

void TriangleMesh3::rotate(const Quaternion<double>& q) {
    parallelFor(kZeroSize, numberOfPoints(),
                [this, q](size_t i) { _points[i] = q * _points[i]; });

    parallelFor(kZeroSize, numberOfNormals(),
                [this, q](size_t i) { _normals[i] = q * _normals[i]; });

    invalidateBvh();
}

void TriangleMesh3::writeObj(std::ostream* strm) const {
    // vertex
    for (const auto& pt : _points) {
        (*strm) << "v " << pt << std::endl;
    }

    // uv coords
    for (const auto& uv : _uvs) {
        (*strm) << "vt " << uv << std::endl;
    }

    // normals
    for (const auto& n : _normals) {
        (*strm) << "vn " << n << std::endl;
    }

    // faces
    bool hasUvs_ = hasUvs();
    bool hasNormals_ = hasNormals();
    for (size_t i = 0; i < numberOfTriangles(); ++i) {
        (*strm) << "f ";
        for (int j = 0; j < 3; ++j) {
            (*strm) << _pointIndices[i][j] + 1;
            if (hasNormals_ || hasUvs_) {
                (*strm) << '/';
            }
            if (hasUvs_) {
                (*strm) << _uvIndices[i][j] + 1;
            }
            if (hasNormals_) {
                (*strm) << '/' << _normalIndices[i][j] + 1;
            }
            (*strm) << ' ';
        }
        (*strm) << std::endl;
    }
}

bool TriangleMesh3::writeObj(const std::string& filename) const {
    std::ofstream file(filename.c_str());
    if (file) {
        writeObj(&file);
        file.close();

        return true;
    } else {
        return false;
    }
}

bool TriangleMesh3::readObj(std::istream* strm) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    const bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, strm);

    // `err` may contain warning message.
    if (!err.empty()) {
        JET_ERROR << err;
        return false;
    }

    // Failed to load obj.
    if (!ret) {
        return false;
    }

    invalidateBvh();

    // Read vertices
    for (size_t idx = 0; idx < attrib.vertices.size() / 3; ++idx) {
        // Access to vertex
        tinyobj::real_t vx = attrib.vertices[3 * idx + 0];
        tinyobj::real_t vy = attrib.vertices[3 * idx + 1];
        tinyobj::real_t vz = attrib.vertices[3 * idx + 2];

        addPoint({vx, vy, vz});
    }

    // Read normals
    for (size_t idx = 0; idx < attrib.normals.size() / 3; ++idx) {
        // Access to normal
        tinyobj::real_t vx = attrib.normals[3 * idx + 0];
        tinyobj::real_t vy = attrib.normals[3 * idx + 1];
        tinyobj::real_t vz = attrib.normals[3 * idx + 2];

        addNormal({vx, vy, vz});
    }

    // Read UVs
    for (size_t idx = 0; idx < attrib.texcoords.size() / 2; ++idx) {
        // Access to UV
        tinyobj::real_t tu = attrib.texcoords[2 * idx + 0];
        tinyobj::real_t tv = attrib.texcoords[2 * idx + 1];

        addUv({tu, tv});
    }

    // Read faces
    for (auto& shape : shapes) {
        size_t idx = 0;

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const size_t fv = shape.mesh.num_face_vertices[f];

            if (fv == 3) {
                if (!attrib.vertices.empty()) {
                    addPointTriangle(
                        {shape.mesh.indices[idx].vertex_index,
                         shape.mesh.indices[idx + 1].vertex_index,
                         shape.mesh.indices[idx + 2].vertex_index});
                }

                if (!attrib.normals.empty()) {
                    addNormalTriangle(
                        {shape.mesh.indices[idx].normal_index,
                         shape.mesh.indices[idx + 1].normal_index,
                         shape.mesh.indices[idx + 2].normal_index});
                }

                if (!attrib.texcoords.empty()) {
                    addUvTriangle({shape.mesh.indices[idx].texcoord_index,
                                   shape.mesh.indices[idx + 1].texcoord_index,
                                   shape.mesh.indices[idx + 2].texcoord_index});
                }
            }

            idx += fv;
        }
    }

    return true;
}

bool TriangleMesh3::readObj(const std::string& filename) {
    std::ifstream file(filename.c_str());
    if (file) {
        bool result = readObj(&file);
        file.close();

        return result;
    } else {
        return false;
    }
}

TriangleMesh3& TriangleMesh3::operator=(const TriangleMesh3& other) {
    set(other);
    return *this;
}

TriangleMesh3::Builder TriangleMesh3::builder() { return Builder(); }

void TriangleMesh3::invalidateBvh() { _bvhInvalidated = true; }

void TriangleMesh3::buildBvh() const {
    if (_bvhInvalidated) {
        size_t nTris = numberOfTriangles();
        std::vector<size_t> ids(nTris);
        std::vector<BoundingBox3D> bounds(nTris);
        for (size_t i = 0; i < nTris; ++i) {
            ids[i] = i;
            bounds[i] = triangle(i).boundingBox();
        }
        _bvh.build(ids, bounds);
        _bvhInvalidated = false;
    }
}

//

TriangleMesh3::Builder& TriangleMesh3::Builder::withPoints(
    const PointArray& points) {
    _points = points;
    return *this;
}

TriangleMesh3::Builder& TriangleMesh3::Builder::withNormals(
    const NormalArray& normals) {
    _normals = normals;
    return *this;
}

TriangleMesh3::Builder& TriangleMesh3::Builder::withUvs(const UvArray& uvs) {
    _uvs = uvs;
    return *this;
}

TriangleMesh3::Builder& TriangleMesh3::Builder::withPointIndices(
    const IndexArray& pointIndices) {
    _pointIndices = pointIndices;
    return *this;
}

TriangleMesh3::Builder& TriangleMesh3::Builder::withNormalIndices(
    const IndexArray& normalIndices) {
    _normalIndices = normalIndices;
    return *this;
}

TriangleMesh3::Builder& TriangleMesh3::Builder::withUvIndices(
    const IndexArray& uvIndices) {
    _uvIndices = uvIndices;
    return *this;
}

TriangleMesh3 TriangleMesh3::Builder::build() const {
    return TriangleMesh3(_points, _normals, _uvs, _pointIndices, _normalIndices,
                         _uvIndices, _transform, _isNormalFlipped);
}

TriangleMesh3Ptr TriangleMesh3::Builder::makeShared() const {
    return std::shared_ptr<TriangleMesh3>(
        new TriangleMesh3(_points, _normals, _uvs, _pointIndices,
                          _normalIndices, _uvIndices, _transform,
                          _isNormalFlipped),
        [](TriangleMesh3* obj) { delete obj; });
}
