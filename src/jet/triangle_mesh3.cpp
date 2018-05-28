// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>

#include <jet/parallel.h>
#include <jet/triangle_mesh3.h>

#include <obj/obj_parser.hpp>

#include <fstream>

using namespace jet;

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
    _points.set(other._points);
    _normals.set(other._normals);
    _uvs.set(other._uvs);
    _pointIndices.set(other._pointIndices);
    _normalIndices.set(other._normalIndices);
    _uvIndices.set(other._uvIndices);

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

const Size3& TriangleMesh3::pointIndex(size_t i) const {
    return _pointIndices[i];
}

Size3& TriangleMesh3::pointIndex(size_t i) { return _pointIndices[i]; }

const Size3& TriangleMesh3::normalIndex(size_t i) const {
    return _normalIndices[i];
}

Size3& TriangleMesh3::normalIndex(size_t i) { return _normalIndices[i]; }

const Size3& TriangleMesh3::uvIndex(size_t i) const { return _uvIndices[i]; }

Size3& TriangleMesh3::uvIndex(size_t i) { return _uvIndices[i]; }

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

size_t TriangleMesh3::numberOfPoints() const { return _points.size(); }

size_t TriangleMesh3::numberOfNormals() const { return _normals.size(); }

size_t TriangleMesh3::numberOfUvs() const { return _uvs.size(); }

size_t TriangleMesh3::numberOfTriangles() const { return _pointIndices.size(); }

bool TriangleMesh3::hasNormals() const { return _normals.size() > 0; }

bool TriangleMesh3::hasUvs() const { return _uvs.size() > 0; }

void TriangleMesh3::addPoint(const Vector3D& pt) { _points.append(pt); }

void TriangleMesh3::addNormal(const Vector3D& n) { _normals.append(n); }

void TriangleMesh3::addUv(const Vector2D& t) { _uvs.append(t); }

void TriangleMesh3::addPointTriangle(const Size3& newPointIndices) {
    _pointIndices.append(newPointIndices);
    invalidateBvh();
}

void TriangleMesh3::addPointNormalTriangle(const Size3& newPointIndices,
                                           const Size3& newNormalIndices) {
    // Number of normal indicies must match with number of point indices once
    // you decided to add normal indicies. Same for the uvs as well.
    JET_ASSERT(_pointIndices.size() == _normalIndices.size());

    _pointIndices.append(newPointIndices);
    _normalIndices.append(newNormalIndices);

    invalidateBvh();
}

void TriangleMesh3::addPointUvNormalTriangle(const Size3& newPointIndices,
                                             const Size3& newUvIndices,
                                             const Size3& newNormalIndices) {
    // Number of normal indicies must match with number of point indices once
    // you decided to add normal indicies. Same for the uvs as well.
    JET_ASSERT(_pointIndices.size() == _normalIndices.size());
    JET_ASSERT(_pointIndices.size() == _uvIndices.size());
    _pointIndices.append(newPointIndices);
    _normalIndices.append(newNormalIndices);
    _uvIndices.append(newUvIndices);

    invalidateBvh();
}

void TriangleMesh3::addPointUvTriangle(const Size3& newPointIndices,
                                       const Size3& newUvIndices) {
    // Number of normal indicies must match with number of point indices once
    // you decided to add normal indicies. Same for the uvs as well.
    JET_ASSERT(_pointIndices.size() == _uvs.size());
    _pointIndices.append(newPointIndices);
    _uvIndices.append(newUvIndices);

    invalidateBvh();
}

void TriangleMesh3::addTriangle(const Triangle3& tri) {
    size_t vStart = _points.size();
    size_t nStart = _normals.size();
    size_t tStart = _uvs.size();
    Size3 newPointIndices;
    Size3 newNormalIndices;
    Size3 newUvIndices;
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
    _normals.resize(_points.size());
    _normalIndices.set(_pointIndices);

    for (size_t i = 0; i < numberOfTriangles(); ++i) {
        Triangle3 tri = triangle(i);
        Vector3D n = tri.faceNormal();
        Size3 f = _pointIndices[i];
        _normals[f.x] = n;
        _normals[f.y] = n;
        _normals[f.z] = n;
    }
}

void TriangleMesh3::setAngleWeightedVertexNormal() {
    _normals.clear();
    _normalIndices.clear();

    Array1<double> angleWeights(_points.size());
    Vector3DArray pseudoNormals(_points.size());

    for (size_t i = 0; i < _points.size(); ++i) {
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

    for (size_t i = 0; i < _points.size(); ++i) {
        if (angleWeights[i] > 0) {
            pseudoNormals[i] /= angleWeights[i];
        }
    }

    std::swap(pseudoNormals, _normals);
    _normalIndices.set(_pointIndices);
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
    obj::obj_parser parser(obj::obj_parser::triangulate_faces |
                           obj::obj_parser::translate_negative_indices);

    parser.info_callback([](size_t lineNumber, const std::string& message) {
        std::cout << lineNumber << " " << message << std::endl;
    });
    parser.warning_callback([](size_t lineNumber, const std::string& message) {
        std::cerr << lineNumber << " " << message << std::endl;
    });
    parser.error_callback([](size_t lineNumber, const std::string& message) {
        std::cerr << lineNumber << " " << message << std::endl;
    });

    parser.geometric_vertex_callback(
        [this](obj::float_type x, obj::float_type y, obj::float_type z) {
            addPoint({x, y, z});
        });

    parser.texture_vertex_callback(
        [this](obj::float_type u, obj::float_type v) {
            addUv({u, v});
        });

    parser.vertex_normal_callback(
        [this](obj::float_type nx, obj::float_type ny, obj::float_type nz) {
            addNormal({nx, ny, nz});
        });

    parser.face_callbacks(
        // triangular_face_geometric_vertices_callback_type
        [this](obj::index_type v0, obj::index_type v1, obj::index_type v2) {
            addPointTriangle({v0 - 1, v1 - 1, v2 - 1});
        },
        // triangular_face_geometric_vertices_texture_vertices_callback_type
        [this](const obj::index_2_tuple_type& v0_vt0,
               const obj::index_2_tuple_type& v1_vt1,
               const obj::index_2_tuple_type& v2_vt2) {
            addPointUvTriangle(
                {std::get<0>(v0_vt0) - 1, std::get<0>(v1_vt1) - 1,
                 std::get<0>(v2_vt2) - 1},
                {std::get<1>(v0_vt0) - 1, std::get<1>(v1_vt1) - 1,
                 std::get<1>(v2_vt2) - 1});
        },
        // triangular_face_geometric_vertices_vertex_normals_callback_type
        [this](const obj::index_2_tuple_type& v0_vn0,
               const obj::index_2_tuple_type& v1_vn1,
               const obj::index_2_tuple_type& v2_vn2) {
            addPointNormalTriangle(
                {std::get<0>(v0_vn0) - 1, std::get<0>(v1_vn1) - 1,
                 std::get<0>(v2_vn2) - 1},
                {std::get<1>(v0_vn0) - 1, std::get<1>(v1_vn1) - 1,
                 std::get<1>(v2_vn2) - 1});
        },
        // triangular_face_geometric_vertices_texture_vertices_vertex_normals...
        [this](const obj::index_3_tuple_type& v0_vt0_vn0,
               const obj::index_3_tuple_type& v1_vt1_vn1,
               const obj::index_3_tuple_type& v2_vt2_vn2) {
            addPointUvNormalTriangle(
                {std::get<0>(v0_vt0_vn0) - 1, std::get<0>(v1_vt1_vn1) - 1,
                 std::get<0>(v2_vt2_vn2) - 1},
                {std::get<1>(v0_vt0_vn0) - 1, std::get<1>(v1_vt1_vn1) - 1,
                 std::get<1>(v2_vt2_vn2) - 1},
                {std::get<2>(v0_vt0_vn0) - 1, std::get<2>(v1_vt1_vn1) - 1,
                 std::get<2>(v2_vt2_vn2) - 1});
        },
        // quadrilateral_face_geometric_vertices_callback_type
        [](obj::index_type, obj::index_type, obj::index_type, obj::index_type) {
        },
        // quadrilateral_face_geometric_vertices_texture_vertices_callback_type
        [](const obj::index_2_tuple_type&, const obj::index_2_tuple_type&,
           const obj::index_2_tuple_type&, const obj::index_2_tuple_type&) {},
        // quadrilateral_face_geometric_vertices_vertex_normals_callback_type
        [](const obj::index_2_tuple_type&, const obj::index_2_tuple_type&,
           const obj::index_2_tuple_type&, const obj::index_2_tuple_type&) {},
        // quadrilateral_face_geometric_vertices_texture_vertices_vertex_norm...
        [](const obj::index_3_tuple_type&, const obj::index_3_tuple_type&,
           const obj::index_3_tuple_type&, const obj::index_3_tuple_type&) {},
        // polygonal_face_geometric_vertices_begin_callback_type
        [](obj::index_type, obj::index_type, obj::index_type) {},
        // polygonal_face_geometric_vertices_vertex_callback_type
        [](obj::index_type) {},
        // polygonal_face_geometric_vertices_end_callback_type
        []() {},
        // polygonal_face_geometric_vertices_texture_vertices_begin_callback_...
        [](const obj::index_2_tuple_type&, const obj::index_2_tuple_type&,
           const obj::index_2_tuple_type&) {},
        // polygonal_face_geometric_vertices_texture_vertices_vertex_callback...
        [](const obj::index_2_tuple_type&) {},
        // polygonal_face_geometric_vertices_texture_vertices_end_callback_type
        []() {},
        // polygonal_face_geometric_vertices_vertex_normals_begin_callback_type
        [](const obj::index_2_tuple_type&, const obj::index_2_tuple_type&,
           const obj::index_2_tuple_type&) {},
        // polygonal_face_geometric_vertices_vertex_normals_vertex_callback_type
        [](const obj::index_2_tuple_type&) {},
        // polygonal_face_geometric_vertices_vertex_normals_end_callback_type
        []() {},
        // polygonal_face_geometric_vertices_texture_vertices_vertex_normals_...
        [](const obj::index_3_tuple_type&, const obj::index_3_tuple_type&,
           const obj::index_3_tuple_type&) {},
        // polygonal_face_geometric_vertices_texture_vertices_vertex_normals_...
        [](const obj::index_3_tuple_type&) {},
        // polygonal_face_geometric_vertices_texture_vertices_vertex_normals_...
        []() {});
    parser.group_name_callback([](const std::string&) {});
    parser.smoothing_group_callback([](obj::size_type) {});
    parser.object_name_callback([](const std::string&) {});
    parser.material_library_callback([](const std::string&) {});
    parser.material_name_callback([](const std::string&) {});
    parser.comment_callback([](const std::string&) {});

    invalidateBvh();

    return parser.parse(*strm);
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
