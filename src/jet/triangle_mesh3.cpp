// Copyright (c) 2019 Doyub Kim
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

namespace {

constexpr double kDefaultFastWindingNumberAccuracy = 2.0;

struct WindingNumberGatherData {
    double areaSums = 0;
    Vector3D areaWeightedNormalSums;
    Vector3D areaWeightedPositionSums;

    WindingNumberGatherData operator+(
        const WindingNumberGatherData& other) const {
        WindingNumberGatherData sum;
        sum.areaSums = areaSums + other.areaSums;
        sum.areaWeightedNormalSums =
            areaWeightedNormalSums + other.areaWeightedNormalSums;
        sum.areaWeightedPositionSums =
            areaWeightedPositionSums + other.areaWeightedPositionSums;

        return sum;
    }
};

template <typename GatherFunc, typename LeafGatherFunc>
WindingNumberGatherData postOrderTraversal(
    const Bvh3<size_t>& bvh, size_t nodeIndex, const GatherFunc& visitorFunc,
    const LeafGatherFunc& leafFunc,
    const WindingNumberGatherData& initGatherData) {
    WindingNumberGatherData data = initGatherData;

    if (bvh.isLeaf(nodeIndex)) {
        data = leafFunc(nodeIndex);
    } else {
        const auto children = bvh.children(nodeIndex);
        data = data + postOrderTraversal(bvh, children.first, visitorFunc,
                                         leafFunc, initGatherData);
        data = data + postOrderTraversal(bvh, children.second, visitorFunc,
                                         leafFunc, initGatherData);
    }
    visitorFunc(nodeIndex, data);

    return data;
}

inline std::ostream& operator<<(std::ostream& strm, const Vector2D& v) {
    strm << v.x << ' ' << v.y;
    return strm;
}

inline std::ostream& operator<<(std::ostream& strm, const Vector3D& v) {
    strm << v.x << ' ' << v.y << ' ' << v.z;
    return strm;
}

}  // namespace

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

void TriangleMesh3::updateQueryEngine() {
    buildBvh();
    buildWindingNumbers();
}

void TriangleMesh3::updateQueryEngine() const {
    buildBvh();
    buildWindingNumbers();
}

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

bool TriangleMesh3::isInsideLocal(const Vector3D& otherPoint) const {
    return fastWindingNumber(otherPoint, kDefaultFastWindingNumberAccuracy) >
           0.5;
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

    invalidateCache();
}

void TriangleMesh3::set(const TriangleMesh3& other) {
    _points.set(other._points);
    _normals.set(other._normals);
    _uvs.set(other._uvs);
    _pointIndices.set(other._pointIndices);
    _normalIndices.set(other._normalIndices);
    _uvIndices.set(other._uvIndices);

    invalidateCache();
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
    invalidateCache();
    return _points[i];
}

const Vector3D& TriangleMesh3::normal(size_t i) const { return _normals[i]; }

Vector3D& TriangleMesh3::normal(size_t i) { return _normals[i]; }

const Vector2D& TriangleMesh3::uv(size_t i) const { return _uvs[i]; }

Vector2D& TriangleMesh3::uv(size_t i) { return _uvs[i]; }

const Point3UI& TriangleMesh3::pointIndex(size_t i) const {
    return _pointIndices[i];
}

Point3UI& TriangleMesh3::pointIndex(size_t i) { return _pointIndices[i]; }

const Point3UI& TriangleMesh3::normalIndex(size_t i) const {
    return _normalIndices[i];
}

Point3UI& TriangleMesh3::normalIndex(size_t i) { return _normalIndices[i]; }

const Point3UI& TriangleMesh3::uvIndex(size_t i) const { return _uvIndices[i]; }

Point3UI& TriangleMesh3::uvIndex(size_t i) { return _uvIndices[i]; }

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

void TriangleMesh3::addPointTriangle(const Point3UI& newPointIndices) {
    _pointIndices.append(newPointIndices);
    invalidateCache();
}

void TriangleMesh3::addNormalTriangle(const Point3UI& newNormalIndices) {
    _normalIndices.append(newNormalIndices);

    invalidateCache();
}

void TriangleMesh3::addUvTriangle(const Point3UI& newUvIndices) {
    _uvIndices.append(newUvIndices);

    invalidateCache();
}

void TriangleMesh3::addPointNormalTriangle(const Point3UI& newPointIndices,
                                           const Point3UI& newNormalIndices) {
    _pointIndices.append(newPointIndices);
    _normalIndices.append(newNormalIndices);

    invalidateCache();
}

void TriangleMesh3::addPointUvNormalTriangle(const Point3UI& newPointIndices,
                                             const Point3UI& newUvIndices,
                                             const Point3UI& newNormalIndices) {
    _pointIndices.append(newPointIndices);
    _normalIndices.append(newNormalIndices);
    _uvIndices.append(newUvIndices);

    invalidateCache();
}

void TriangleMesh3::addPointUvTriangle(const Point3UI& newPointIndices,
                                       const Point3UI& newUvIndices) {
    _pointIndices.append(newPointIndices);
    _uvIndices.append(newUvIndices);

    invalidateCache();
}

void TriangleMesh3::addTriangle(const Triangle3& tri) {
    size_t vStart = _points.size();
    size_t nStart = _normals.size();
    size_t tStart = _uvs.size();
    Point3UI newPointIndices;
    Point3UI newNormalIndices;
    Point3UI newUvIndices;
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

    invalidateCache();
}

void TriangleMesh3::setFaceNormal() {
    _normals.resize(_points.size());
    _normalIndices.set(_pointIndices);

    for (size_t i = 0; i < numberOfTriangles(); ++i) {
        Triangle3 tri = triangle(i);
        Vector3D n = tri.faceNormal();
        Point3UI f = _pointIndices[i];
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
    invalidateCache();
}

void TriangleMesh3::translate(const Vector3D& t) {
    parallelFor(kZeroSize, numberOfPoints(),
                [this, t](size_t i) { _points[i] += t; });
    invalidateCache();
}

void TriangleMesh3::rotate(const Quaternion<double>& q) {
    parallelFor(kZeroSize, numberOfPoints(),
                [this, q](size_t i) { _points[i] = q * _points[i]; });

    parallelFor(kZeroSize, numberOfNormals(),
                [this, q](size_t i) { _normals[i] = q * _normals[i]; });

    invalidateCache();
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

    invalidateCache();

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

void TriangleMesh3::invalidateCache() {
    _bvhInvalidated = true;
    _wnInvalidated = true;
}

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

void TriangleMesh3::buildWindingNumbers() const {
    // Barill et al., Fast Winding Numbers for Soups and Clouds, ACM SIGGRAPH
    // 2018
    if (_wnInvalidated) {
        buildBvh();

        size_t nNodes = _bvh.numberOfNodes();
        _wnAreaWeightedNormalSums.resize(nNodes);
        _wnAreaWeightedAvgPositions.resize(nNodes);

        const auto visitorFunc = [&](size_t nodeIndex,
                                     const WindingNumberGatherData& data) {
            _wnAreaWeightedNormalSums[nodeIndex] = data.areaWeightedNormalSums;
            _wnAreaWeightedAvgPositions[nodeIndex] =
                data.areaWeightedPositionSums / data.areaSums;

        };
        const auto leafFunc = [&](size_t nodeIndex) -> WindingNumberGatherData {
            WindingNumberGatherData result;

            auto iter = _bvh.itemOfNode(nodeIndex);
            JET_ASSERT(iter != _bvh.end());

            Triangle3 tri = triangle(*iter);
            double area = tri.area();
            result.areaSums = area;
            result.areaWeightedNormalSums = area * tri.faceNormal();
            result.areaWeightedPositionSums =
                area * (tri.points[0] + tri.points[1] + tri.points[2]) / 3.0;

            return result;
        };

        postOrderTraversal(_bvh, 0, visitorFunc, leafFunc,
                           WindingNumberGatherData());

        _wnInvalidated = false;
    }
}

double TriangleMesh3::windingNumber(const Vector3D& queryPoint,
                                    size_t triIndex) const {
    // Jacobson et al., Robust Inside-Outside Segmentation using Generalized
    // Winding Numbers, ACM SIGGRAPH 2013.
    const Vector3D& vi = _points[_pointIndices[triIndex][0]];
    const Vector3D& vj = _points[_pointIndices[triIndex][1]];
    const Vector3D& vk = _points[_pointIndices[triIndex][2]];
    const Vector3D va = vi - queryPoint;
    const Vector3D vb = vj - queryPoint;
    const Vector3D vc = vk - queryPoint;
    const double a = va.length();
    const double b = vb.length();
    const double c = vc.length();

    const Matrix3x3D mat(va.x, vb.x, vc.x, va.y, vb.y, vc.y, va.z, vb.z, vc.z);
    const double det = mat.determinant();
    const double denom =
        a * b * c + va.dot(vb) * c + vb.dot(vc) * a + vc.dot(va) * b;

    const double solidAngle = 2.0 * std::atan2(det, denom);

    return solidAngle;
}

double TriangleMesh3::fastWindingNumber(const Vector3D& queryPoint,
                                        double accuracy) const {
    buildWindingNumbers();

    return fastWindingNumber(queryPoint, 0, accuracy);
}

double TriangleMesh3::fastWindingNumber(const Vector3D& q, size_t rootNodeIndex,
                                        double accuracy) const {
    // Barill et al., Fast Winding Numbers for Soups and Clouds, ACM SIGGRAPH
    // 2018.
    const Vector3D& treeP = _wnAreaWeightedAvgPositions[rootNodeIndex];
    const double qToP2 = q.distanceSquaredTo(treeP);

    const Vector3D& treeN = _wnAreaWeightedNormalSums[rootNodeIndex];
    const BoundingBox3D& treeBound = _bvh.nodeBound(rootNodeIndex);
    const Vector3D treeRVec =
        jet::max(treeP - treeBound.lowerCorner, treeBound.upperCorner - treeP);
    const double treeR = treeRVec.length();

    if (qToP2 > square(accuracy * treeR)) {
        // Case: q is sufficiently far from all elements in tree
        // TODO: This is zero-th order approximation. Higher-order approximation
        // from Section 3.2.1 could be implemented for better accuracy in the
        // future.
        return (treeP - q).dot(treeN) / (kFourPiD * cubic(std::sqrt(qToP2)));
    } else {
        if (_bvh.isLeaf(rootNodeIndex)) {
            // Case: q is nearby; use direct sum for tree’s elements
            auto iter = _bvh.itemOfNode(rootNodeIndex);
            return windingNumber(q, *iter) * kInvFourPiD;
        } else {
            // Case: Recursive call
            const auto children = _bvh.children(rootNodeIndex);
            double wn = 0.0;
            wn += fastWindingNumber(q, children.first, accuracy);
            wn += fastWindingNumber(q, children.second, accuracy);
            return wn;
        }
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
