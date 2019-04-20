// Copyright (c) 2019 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_TRIANGLE_MESH3_H_
#define INCLUDE_JET_TRIANGLE_MESH3_H_

#include <jet/array1.h>
#include <jet/bvh3.h>
#include <jet/point3.h>
#include <jet/quaternion.h>
#include <jet/surface3.h>
#include <jet/triangle3.h>

#include <iostream>

namespace jet {

//!
//! \brief 3-D triangle mesh geometry.
//!
//! This class represents 3-D triangle mesh geometry which extends Surface3 by
//! overriding surface-related queries. The mesh structure stores point,
//! normals, and UV coordinates.
//!
class TriangleMesh3 final : public Surface3 {
 public:
    class Builder;

    typedef Array1<Vector2D> Vector2DArray;
    typedef Array1<Vector3D> Vector3DArray;
    typedef Array1<Point3UI> IndexArray;

    typedef Vector3DArray PointArray;
    typedef Vector3DArray NormalArray;
    typedef Vector2DArray UvArray;

    //! Default constructor.
    TriangleMesh3(const Transform3& transform = Transform3(),
                  bool isNormalFlipped = false);

    //! Constructs mesh with points, normals, uvs, and their indices.
    TriangleMesh3(const PointArray& points, const NormalArray& normals,
                  const UvArray& uvs, const IndexArray& pointIndices,
                  const IndexArray& normalIndices, const IndexArray& uvIndices,
                  const Transform3& transform_ = Transform3(),
                  bool isNormalFlipped = false);

    //! Copy constructor.
    TriangleMesh3(const TriangleMesh3& other);

    //! Updates internal spatial query engine.
    void updateQueryEngine() override;

    //! Updates internal spatial query engine.
    void updateQueryEngine() const;

    //! Clears all content.
    void clear();

    //! Copies the contents from \p other mesh.
    void set(const TriangleMesh3& other);

    //! Swaps the contents with \p other mesh.
    void swap(TriangleMesh3& other);

    //! Returns area of this mesh.
    double area() const;

    //! Returns volume of this mesh.
    double volume() const;

    //! Returns constant reference to the i-th point.
    const Vector3D& point(size_t i) const;

    //! Returns reference to the i-th point.
    Vector3D& point(size_t i);

    //! Returns constant reference to the i-th normal.
    const Vector3D& normal(size_t i) const;

    //! Returns reference to the i-th normal.
    Vector3D& normal(size_t i);

    //! Returns constant reference to the i-th UV coordinates.
    const Vector2D& uv(size_t i) const;

    //! Returns reference to the i-th UV coordinates.
    Vector2D& uv(size_t i);

    //! Returns constant reference to the point indices of i-th triangle.
    const Point3UI& pointIndex(size_t i) const;

    //! Returns reference to the point indices of i-th triangle.
    Point3UI& pointIndex(size_t i);

    //! Returns constant reference to the normal indices of i-th triangle.
    const Point3UI& normalIndex(size_t i) const;

    //! Returns reference to the normal indices of i-th triangle.
    Point3UI& normalIndex(size_t i);

    //! Returns constant reference to the UV indices of i-th triangle.
    const Point3UI& uvIndex(size_t i) const;

    //! Returns reference to the UV indices of i-th triangle.
    Point3UI& uvIndex(size_t i);

    //! Returns i-th triangle.
    Triangle3 triangle(size_t i) const;

    //! Returns number of points.
    size_t numberOfPoints() const;

    //! Returns number of normals.
    size_t numberOfNormals() const;

    //! Returns number of UV coordinates.
    size_t numberOfUvs() const;

    //! Returns number of triangles.
    size_t numberOfTriangles() const;

    //! Returns true if the mesh has normals.
    bool hasNormals() const;

    //! Returns true if the mesh has UV coordinates.
    bool hasUvs() const;

    //! Adds a point.
    void addPoint(const Vector3D& pt);

    //! Adds a normal.
    void addNormal(const Vector3D& n);

    //! Adds a UV.
    void addUv(const Vector2D& t);

    //! Adds a triangle with points.
    void addPointTriangle(const Point3UI& newPointIndices);

    //! Adds a triangle with normal.
    void addNormalTriangle(const Point3UI& newNormalIndices);

    //! Adds a triangle with UV.
    void addUvTriangle(const Point3UI& newUvIndices);

    //! Adds a triangle with point and normal.
    void addPointNormalTriangle(const Point3UI& newPointIndices,
                                const Point3UI& newNormalIndices);

    //! Adds a triangle with point, normal, and UV.
    void addPointUvNormalTriangle(const Point3UI& newPointIndices,
                                  const Point3UI& newUvIndices,
                                  const Point3UI& newNormalIndices);

    //! Adds a triangle with point and UV.
    void addPointUvTriangle(const Point3UI& newPointIndices,
                            const Point3UI& newUvIndices);

    //! Add a triangle.
    void addTriangle(const Triangle3& tri);

    //! Sets entire normals to the face normals.
    void setFaceNormal();

    //! Sets angle weighted vertex normal.
    void setAngleWeightedVertexNormal();

    //! Scales the mesh by given factor.
    void scale(double factor);

    //! Translates the mesh.
    void translate(const Vector3D& t);

    //! Rotates the mesh.
    void rotate(const QuaternionD& q);

    //! Writes the mesh in obj format to the output stream.
    void writeObj(std::ostream* strm) const;

    //! Writes the mesh in obj format to the file.
    bool writeObj(const std::string& filename) const;

    //! Reads the mesh in obj format from the input stream.
    bool readObj(std::istream* strm);

    //! Reads the mesh in obj format from the file.
    bool readObj(const std::string& filename);

    //! Copies \p other mesh.
    TriangleMesh3& operator=(const TriangleMesh3& other);

    //! Returns builder fox TriangleMesh3.
    static Builder builder();

 protected:
    Vector3D closestPointLocal(const Vector3D& otherPoint) const override;

    double closestDistanceLocal(const Vector3D& otherPoint) const override;

    bool intersectsLocal(const Ray3D& ray) const override;

    BoundingBox3D boundingBoxLocal() const override;

    Vector3D closestNormalLocal(const Vector3D& otherPoint) const override;

    SurfaceRayIntersection3 closestIntersectionLocal(
        const Ray3D& ray) const override;

    bool isInsideLocal(const Vector3D& otherPoint) const override;

 private:
    PointArray _points;
    NormalArray _normals;
    UvArray _uvs;
    IndexArray _pointIndices;
    IndexArray _normalIndices;
    IndexArray _uvIndices;

    mutable Bvh3<size_t> _bvh;
    mutable bool _bvhInvalidated = true;

    mutable Array1<Vector3D> _wnAreaWeightedNormalSums;
    mutable Array1<Vector3D> _wnAreaWeightedAvgPositions;
    mutable bool _wnInvalidated = true;

    void invalidateCache();

    void buildBvh() const;

    void buildWindingNumbers() const;

    double windingNumber(const Vector3D& queryPoint, size_t triIndex) const;

    double fastWindingNumber(const Vector3D& queryPoint, double accuracy) const;

    double fastWindingNumber(const Vector3D& queryPoint, size_t rootNodeIndex,
                             double accuracy) const;
};

//! Shared pointer for the TriangleMesh3 type.
typedef std::shared_ptr<TriangleMesh3> TriangleMesh3Ptr;

//!
//! \brief Front-end to create TriangleMesh3 objects step by step.
//!
class TriangleMesh3::Builder final
    : public SurfaceBuilderBase3<TriangleMesh3::Builder> {
 public:
    //! Returns builder with points.
    Builder& withPoints(const PointArray& points);

    //! Returns builder with normals.
    Builder& withNormals(const NormalArray& normals);

    //! Returns builder with uvs.
    Builder& withUvs(const UvArray& uvs);

    //! Returns builder with point indices.
    Builder& withPointIndices(const IndexArray& pointIndices);

    //! Returns builder with normal indices.
    Builder& withNormalIndices(const IndexArray& normalIndices);

    //! Returns builder with uv indices.
    Builder& withUvIndices(const IndexArray& uvIndices);

    //! Builds TriangleMesh3.
    TriangleMesh3 build() const;

    //! Builds shared pointer of TriangleMesh3 instance.
    TriangleMesh3Ptr makeShared() const;

 private:
    PointArray _points;
    NormalArray _normals;
    UvArray _uvs;
    IndexArray _pointIndices;
    IndexArray _normalIndices;
    IndexArray _uvIndices;
};

}  // namespace jet

#endif  // INCLUDE_JET_TRIANGLE_MESH3_H_
