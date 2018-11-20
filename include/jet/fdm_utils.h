// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_FDM_UTILS_H_
#define INCLUDE_JET_FDM_UTILS_H_

#include <jet/array_view.h>
#include <jet/matrix.h>

#include <iostream>

namespace jet {

//! \brief Returns 2-D gradient vector from given 2-D scalar grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j).
Vector2D gradient2(const ConstArrayView2<double>& data,
                   const Vector2D& gridSpacing, size_t i, size_t j);

//! \brief Returns 2-D gradient vectors from given 2-D vector grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j).
std::array<Vector2D, 2> gradient2(const ConstArrayView2<Vector2D>& data,
                                  const Vector2D& gridSpacing, size_t i,
                                  size_t j);

//! \brief Returns 3-D gradient vector from given 3-D scalar grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j, \p k).
Vector3D gradient3(const ConstArrayView3<double>& data,
                   const Vector3D& gridSpacing, size_t i, size_t j, size_t k);

//! \brief Returns 3-D gradient vectors from given 3-D vector grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j, \p k).
std::array<Vector3D, 3> gradient3(const ConstArrayView3<Vector3D>& data,
                                  const Vector3D& gridSpacing, size_t i,
                                  size_t j, size_t k);

//! \brief Returns Laplacian value from given 2-D scalar grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j).
double laplacian2(const ConstArrayView2<double>& data,
                  const Vector2D& gridSpacing, size_t i, size_t j);

//! \brief Returns 2-D Laplacian vectors from given 2-D vector grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j).
Vector2D laplacian2(const ConstArrayView2<Vector2D>& data,
                    const Vector2D& gridSpacing, size_t i, size_t j);

//! \brief Returns Laplacian value from given 3-D scalar grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j, \p k).
double laplacian3(const ConstArrayView3<double>& data,
                  const Vector3D& gridSpacing, size_t i, size_t j, size_t k);

//! \brief Returns 3-D Laplacian vectors from given 3-D vector grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j, \p k).
Vector3D laplacian3(const ConstArrayView3<Vector3D>& data,
                    const Vector3D& gridSpacing, size_t i, size_t j, size_t k);

//! \brief Returns divergence value from given 2-D vector grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j).
double divergence2(const ConstArrayView2<Vector2D>& data,
                   const Vector2D& gridSpacing, size_t i, size_t j);

//! \brief Returns diverence value from given 3-D vector grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j, \p k).
double divergence3(const ConstArrayView3<Vector3D>& data,
                   const Vector3D& gridSpacing, size_t i, size_t j, size_t k);

//! \brief Returns curl value from given 2-D vector grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j).
double curl2(const ConstArrayView2<Vector2D>& data, const Vector2D& gridSpacing,
             size_t i, size_t j);

//! \brief Returns curl value from given 3-D vector grid-like array
//!        \p data, \p gridSpacing, and array index (\p i, \p j, \p k).
Vector3D curl3(const ConstArrayView3<Vector3D>& data,
               const Vector3D& gridSpacing, size_t i, size_t j, size_t k);

template <size_t N>
struct GetFdmUtils {};

template <>
struct GetFdmUtils<2> {
    static Vector2D gradient(const ConstArrayView2<double>& data,
                             const Vector2D& gridSpacing,
                             const Vector2UZ& idx) {
        return gradient2(data, gridSpacing, idx.x, idx.y);
    }

    static double laplacian(const ConstArrayView2<double>& data,
                            const Vector2D& gridSpacing, const Vector2UZ& idx) {
        return laplacian2(data, gridSpacing, idx.x, idx.y);
    }

    static double divergence(const ConstArrayView2<Vector2D>& data,
                             const Vector2D& gridSpacing,
                             const Vector2UZ& idx) {
        return divergence2(data, gridSpacing, idx.x, idx.y);
    }

    static double curl(const ConstArrayView2<Vector2D>& data,
                       const Vector2D& gridSpacing, const Vector2UZ& idx) {
        return curl2(data, gridSpacing, idx.x, idx.y);
    }
};

template <>
struct GetFdmUtils<3> {
    static Vector3D gradient(const ConstArrayView3<double>& data,
                             const Vector3D& gridSpacing,
                             const Vector3UZ& idx) {
        return gradient3(data, gridSpacing, idx.x, idx.y, idx.z);
    }

    static double laplacian(const ConstArrayView3<double>& data,
                            const Vector3D& gridSpacing, const Vector3UZ& idx) {
        return laplacian3(data, gridSpacing, idx.x, idx.y, idx.z);
    }

    static double divergence(const ConstArrayView3<Vector3D>& data,
                             const Vector3D& gridSpacing,
                             const Vector3UZ& idx) {
        return divergence3(data, gridSpacing, idx.x, idx.y, idx.z);
    }

    static Vector3D curl(const ConstArrayView3<Vector3D>& data,
                         const Vector3D& gridSpacing, const Vector3UZ& idx) {
        return curl3(data, gridSpacing, idx.x, idx.y, idx.z);
    }
};

}  // namespace jet

#endif  // INCLUDE_JET_FDM_UTILS_H_
