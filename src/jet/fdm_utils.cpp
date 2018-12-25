// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <common.h>

#include <jet/fdm_utils.h>

namespace jet {

Vector2D gradient2(const ConstArrayView2<double>& data,
                   const Vector2D& gridSpacing, size_t i, size_t j) {
    const Vector2UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    double left = data((i > 0) ? i - 1 : i, j);
    double right = data((i + 1 < ds.x) ? i + 1 : i, j);
    double down = data(i, (j > 0) ? j - 1 : j);
    double up = data(i, (j + 1 < ds.y) ? j + 1 : j);

    return 0.5 * elemDiv(Vector2D(right - left, up - down), gridSpacing);
}

std::array<Vector2D, 2> gradient2(const ConstArrayView2<Vector2D>& data,
                                  const Vector2D& gridSpacing, size_t i,
                                  size_t j) {
    const Vector2UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    Vector2D left = data((i > 0) ? i - 1 : i, j);
    Vector2D right = data((i + 1 < ds.x) ? i + 1 : i, j);
    Vector2D down = data(i, (j > 0) ? j - 1 : j);
    Vector2D up = data(i, (j + 1 < ds.y) ? j + 1 : j);

    std::array<Vector2D, 2> result;
    result[0] =
        0.5 * elemDiv(Vector2D(right.x - left.x, up.x - down.x), gridSpacing);
    result[1] =
        0.5 * elemDiv(Vector2D(right.y - left.y, up.y - down.y), gridSpacing);
    return result;
}

Vector3D gradient3(const ConstArrayView3<double>& data,
                   const Vector3D& gridSpacing, size_t i, size_t j, size_t k) {
    const Vector3UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    double left = data((i > 0) ? i - 1 : i, j, k);
    double right = data((i + 1 < ds.x) ? i + 1 : i, j, k);
    double down = data(i, (j > 0) ? j - 1 : j, k);
    double up = data(i, (j + 1 < ds.y) ? j + 1 : j, k);
    double back = data(i, j, (k > 0) ? k - 1 : k);
    double front = data(i, j, (k + 1 < ds.z) ? k + 1 : k);

    return 0.5 * elemDiv(Vector3D(right - left, up - down, front - back),
                         gridSpacing);
}

std::array<Vector3D, 3> gradient3(const ConstArrayView3<Vector3D>& data,
                                  const Vector3D& gridSpacing, size_t i,
                                  size_t j, size_t k) {
    const Vector3UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    Vector3D left = data((i > 0) ? i - 1 : i, j, k);
    Vector3D right = data((i + 1 < ds.x) ? i + 1 : i, j, k);
    Vector3D down = data(i, (j > 0) ? j - 1 : j, k);
    Vector3D up = data(i, (j + 1 < ds.y) ? j + 1 : j, k);
    Vector3D back = data(i, j, (k > 0) ? k - 1 : k);
    Vector3D front = data(i, j, (k + 1 < ds.z) ? k + 1 : k);

    std::array<Vector3D, 3> result;
    result[0] = 0.5 * elemDiv(Vector3D(right.x - left.x, up.x - down.x,
                                       front.x - back.x),
                              gridSpacing);
    result[1] = 0.5 * elemDiv(Vector3D(right.y - left.y, up.y - down.y,
                                       front.y - back.y),
                              gridSpacing);
    result[2] = 0.5 * elemDiv(Vector3D(right.z - left.z, up.z - down.z,
                                       front.z - back.z),
                              gridSpacing);
    return result;
}

double laplacian2(const ConstArrayView2<double>& data,
                  const Vector2D& gridSpacing, size_t i, size_t j) {
    const double center = data(i, j);
    const Vector2UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    double dleft = 0.0;
    double dright = 0.0;
    double ddown = 0.0;
    double dup = 0.0;

    if (i > 0) {
        dleft = center - data(i - 1, j);
    }
    if (i + 1 < ds.x) {
        dright = data(i + 1, j) - center;
    }

    if (j > 0) {
        ddown = center - data(i, j - 1);
    }
    if (j + 1 < ds.y) {
        dup = data(i, j + 1) - center;
    }

    return (dright - dleft) / square(gridSpacing.x) +
           (dup - ddown) / square(gridSpacing.y);
}

Vector2D laplacian2(const ConstArrayView2<Vector2D>& data,
                    const Vector2D& gridSpacing, size_t i, size_t j) {
    const Vector2D center = data(i, j);
    const Vector2UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    Vector2D dleft;
    Vector2D dright;
    Vector2D ddown;
    Vector2D dup;

    if (i > 0) {
        dleft = center - data(i - 1, j);
    }
    if (i + 1 < ds.x) {
        dright = data(i + 1, j) - center;
    }

    if (j > 0) {
        ddown = center - data(i, j - 1);
    }
    if (j + 1 < ds.y) {
        dup = data(i, j + 1) - center;
    }

    return (dright - dleft) / square(gridSpacing.x) +
           (dup - ddown) / square(gridSpacing.y);
}

double laplacian3(const ConstArrayView3<double>& data,
                  const Vector3D& gridSpacing, size_t i, size_t j, size_t k) {
    const double center = data(i, j, k);
    const Vector3UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    double dleft = 0.0;
    double dright = 0.0;
    double ddown = 0.0;
    double dup = 0.0;
    double dback = 0.0;
    double dfront = 0.0;

    if (i > 0) {
        dleft = center - data(i - 1, j, k);
    }
    if (i + 1 < ds.x) {
        dright = data(i + 1, j, k) - center;
    }

    if (j > 0) {
        ddown = center - data(i, j - 1, k);
    }
    if (j + 1 < ds.y) {
        dup = data(i, j + 1, k) - center;
    }

    if (k > 0) {
        dback = center - data(i, j, k - 1);
    }
    if (k + 1 < ds.z) {
        dfront = data(i, j, k + 1) - center;
    }

    return (dright - dleft) / square(gridSpacing.x) +
           (dup - ddown) / square(gridSpacing.y) +
           (dfront - dback) / square(gridSpacing.z);
}

Vector3D laplacian3(const ConstArrayView3<Vector3D>& data,
                    const Vector3D& gridSpacing, size_t i, size_t j, size_t k) {
    const Vector3D center = data(i, j, k);
    const Vector3UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    Vector3D dleft;
    Vector3D dright;
    Vector3D ddown;
    Vector3D dup;
    Vector3D dback;
    Vector3D dfront;

    if (i > 0) {
        dleft = center - data(i - 1, j, k);
    }
    if (i + 1 < ds.x) {
        dright = data(i + 1, j, k) - center;
    }

    if (j > 0) {
        ddown = center - data(i, j - 1, k);
    }
    if (j + 1 < ds.y) {
        dup = data(i, j + 1, k) - center;
    }

    if (k > 0) {
        dback = center - data(i, j, k - 1);
    }
    if (k + 1 < ds.z) {
        dfront = data(i, j, k + 1) - center;
    }

    return (dright - dleft) / square(gridSpacing.x) +
           (dup - ddown) / square(gridSpacing.y) +
           (dfront - dback) / square(gridSpacing.z);
}

double divergence2(const ConstArrayView2<Vector2D>& data,
                   const Vector2D& gridSpacing, size_t i, size_t j) {
    const Vector2UZ& ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    double left = data((i > 0) ? i - 1 : i, j).x;
    double right = data((i + 1 < ds.x) ? i + 1 : i, j).x;
    double down = data(i, (j > 0) ? j - 1 : j).y;
    double up = data(i, (j + 1 < ds.y) ? j + 1 : j).y;

    return 0.5 * (right - left) / gridSpacing.x +
           0.5 * (up - down) / gridSpacing.y;
}

double divergence3(const ConstArrayView3<Vector3D>& data,
                   const Vector3D& gridSpacing, size_t i, size_t j, size_t k) {
    const Vector3UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    double left = data((i > 0) ? i - 1 : i, j, k).x;
    double right = data((i + 1 < ds.x) ? i + 1 : i, j, k).x;
    double down = data(i, (j > 0) ? j - 1 : j, k).y;
    double up = data(i, (j + 1 < ds.y) ? j + 1 : j, k).y;
    double back = data(i, j, (k > 0) ? k - 1 : k).z;
    double front = data(i, j, (k + 1 < ds.z) ? k + 1 : k).z;

    return 0.5 * (right - left) / gridSpacing.x +
           0.5 * (up - down) / gridSpacing.y +
           0.5 * (front - back) / gridSpacing.z;
}

double curl2(const ConstArrayView2<Vector2D>& data, const Vector2D& gridSpacing,
             size_t i, size_t j) {
    const Vector2UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    Vector2D left = data((i > 0) ? i - 1 : i, j);
    Vector2D right = data((i + 1 < ds.x) ? i + 1 : i, j);
    Vector2D bottom = data(i, (j > 0) ? j - 1 : j);
    Vector2D top = data(i, (j + 1 < ds.y) ? j + 1 : j);

    double Fx_ym = bottom.x;
    double Fx_yp = top.x;

    double Fy_xm = left.y;
    double Fy_xp = right.y;

    return 0.5 * (Fy_xp - Fy_xm) / gridSpacing.x -
           0.5 * (Fx_yp - Fx_ym) / gridSpacing.y;
}

Vector3D curl3(const ConstArrayView3<Vector3D>& data,
               const Vector3D& gridSpacing, size_t i, size_t j, size_t k) {
    const Vector3UZ ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    Vector3D left = data((i > 0) ? i - 1 : i, j, k);
    Vector3D right = data((i + 1 < ds.x) ? i + 1 : i, j, k);
    Vector3D down = data(i, (j > 0) ? j - 1 : j, k);
    Vector3D up = data(i, (j + 1 < ds.y) ? j + 1 : j, k);
    Vector3D back = data(i, j, (k > 0) ? k - 1 : k);
    Vector3D front = data(i, j, (k + 1 < ds.z) ? k + 1 : k);

    double Fx_ym = down.x;
    double Fx_yp = up.x;
    double Fx_zm = back.x;
    double Fx_zp = front.x;

    double Fy_xm = left.y;
    double Fy_xp = right.y;
    double Fy_zm = back.y;
    double Fy_zp = front.y;

    double Fz_xm = left.z;
    double Fz_xp = right.z;
    double Fz_ym = down.z;
    double Fz_yp = up.z;

    return Vector3D(0.5 * (Fz_yp - Fz_ym) / gridSpacing.y -
                        0.5 * (Fy_zp - Fy_zm) / gridSpacing.z,
                    0.5 * (Fx_zp - Fx_zm) / gridSpacing.z -
                        0.5 * (Fz_xp - Fz_xm) / gridSpacing.x,
                    0.5 * (Fy_xp - Fy_xm) / gridSpacing.x -
                        0.5 * (Fx_yp - Fx_ym) / gridSpacing.y);
}

}  // namespace jet
