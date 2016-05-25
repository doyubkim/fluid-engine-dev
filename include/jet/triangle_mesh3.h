// Copyright (c) 2016 Doyub Kim

#ifndef INCLUDE_JET_TRIANGLE_MESH3_H_
#define INCLUDE_JET_TRIANGLE_MESH3_H_

#include <jet/array1.h>
#include <jet/point3.h>
#include <jet/quaternion.h>
#include <jet/surface3.h>
#include <jet/triangle3.h>
#include <iostream>
#include <utility>  // just make cpplint happy..

namespace jet {


class TriangleMesh3 final : public Surface3 {
 public:
    typedef Array1<Vector2D> Vector2DArray;
    typedef Array1<Vector3D> Vector3DArray;
    typedef Array1<Point3UI> IndexArray;

    //! Default constructor.
    TriangleMesh3();

    //! Copy constructor.
    TriangleMesh3(const TriangleMesh3& other);

    Vector3D closestPoint(const Vector3D& otherPoint) const override;

    Vector3D actualClosestNormal(const Vector3D& otherPoint) const override;

    void getClosestIntersection(
        const Ray3D& ray,
        SurfaceRayIntersection3* intersection) const override;

    BoundingBox3D boundingBox() const override;

    bool intersects(const Ray3D& ray) const override;

    double closestDistance(const Vector3D& otherPoint) const override;

    //! Clears all content.
    void clear();

    void set(const TriangleMesh3& other);

    void swap(TriangleMesh3& other);

    //! Returns area of this mesh.
    double area() const;

    //! Returns volume of this mesh.
    double volume() const;

    //! Returns random sample on the surface.
    void sample(
        double u1,
        double u2,
        double u3,
        Vector3D* x,
        Vector3D* n) const;

    const Vector3D& point(size_t i) const;

    Vector3D& point(size_t i);

    const Vector3D& normal(size_t i) const;

    Vector3D& normal(size_t i);

    const Vector2D& uv(size_t i) const;

    Vector2D& uv(size_t i);

    const Point3UI& pointIndex(size_t i) const;

    Point3UI& pointIndex(size_t i);

    const Point3UI& normalIndex(size_t i) const;

    Point3UI& normalIndex(size_t i);

    const Point3UI& uvIndex(size_t i) const;

    Point3UI& uvIndex(size_t i);

    //! Returns i-th triangle.
    Triangle3 triangle(size_t i) const;

    size_t numberOfPoints() const;

    size_t numberOfNormals() const;

    size_t numberOfUvs() const;

    size_t numberOfFaces() const;

    bool hasNormals() const;

    bool hasUvs() const;

    void addPoint(const Vector3D& pt);

    void addNormal(const Vector3D& n);

    void addUv(const Vector2D& t);

    void addPointFace(const Point3UI& newPointIndices);

    void addPointNormalFace(
        const Point3UI& newPointIndices,
        const Point3UI& newNormalIndices);

    void addPointNormalUvFace(
        const Point3UI& newPointIndices,
        const Point3UI& newNormalIndices,
        const Point3UI& newUvIndices);

    void addPointUvFace(
        const Point3UI& newPointIndices,
        const Point3UI& newUvIndices);

    //! Add a triangle.
    void addTriangle(const Triangle3& tri);

    void setFaceNormal();

    void setAngleWeightedVertexNormal();

    void clearAreaCache();

    void computeAreaCache();

    void scale(double factor);

    void translate(const Vector3D& t);

    void rotate(const QuaternionD& q);

    void writeObj(std::ostream* strm) const;

    bool readObj(std::istream* strm);

    TriangleMesh3& operator=(const TriangleMesh3& other);

 private:
    Vector3DArray _points;
    Vector3DArray _normals;
    Vector2DArray _uvs;
    IndexArray _pointIndices;
    IndexArray _normalIndices;
    IndexArray _uvIndices;
    mutable Array1<double> _areaCache;

    void computeAreaCacheP() const;
};

typedef std::shared_ptr<TriangleMesh3> TriangleMesh3Ptr;

}  // namespace jet

#endif  // INCLUDE_JET_TRIANGLE_MESH3_H_
