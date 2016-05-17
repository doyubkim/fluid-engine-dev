// Copyright (c) 2016 Doyub Kim

#include <pch.h>
#include <jet/fdm_utils.h>
#include <jet/parallel.h>

namespace jet {

Vector2D gradient2(
    const ConstArrayAccessor2<double>& data,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j) {
    const Size2 ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    double left = data((i > 0) ? i - 1 : i, j);
    double right = data((i + 1 < ds.x) ? i + 1 : i, j);
    double down = data(i, (j > 0) ? j - 1 : j);
    double up = data(i, (j + 1 < ds.y) ? j + 1 : j);

    return 0.5 * Vector2D(right - left, up - down) / gridSpacing;
}

std::array<Vector2D, 2> gradient2(
    const ConstArrayAccessor2<Vector2D>& data,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j) {
    const Size2 ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y);

    Vector2D left = data((i > 0) ? i - 1 : i, j);
    Vector2D right = data((i + 1 < ds.x) ? i + 1 : i, j);
    Vector2D down = data(i, (j > 0) ? j - 1 : j);
    Vector2D up = data(i, (j + 1 < ds.y) ? j + 1 : j);

    std::array<Vector2D, 2> result;
    result[0] = 0.5 * Vector2D(right.x - left.x, up.x - down.x) / gridSpacing;
    result[1] = 0.5 * Vector2D(right.y - left.y, up.y - down.y) / gridSpacing;
    return result;
}

Vector3D gradient3(
    const ConstArrayAccessor3<double>& data,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k) {
    const Size3 ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    double left = data((i > 0) ? i - 1 : i, j, k);
    double right = data((i + 1 < ds.x) ? i + 1 : i, j, k);
    double down = data(i, (j > 0) ? j - 1 : j, k);
    double up = data(i, (j + 1 < ds.y) ? j + 1 : j, k);
    double back = data(i, j, (k > 0) ? k - 1 : k);
    double front = data(i, j, (k + 1 < ds.z) ? k + 1 : k);

    return 0.5 * Vector3D(right - left, up - down, front - back) / gridSpacing;
}

std::array<Vector3D, 3> gradient3(
    const ConstArrayAccessor3<Vector3D>& data,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k) {
    const Size3 ds = data.size();

    JET_ASSERT(i < ds.x && j < ds.y && k < ds.z);

    Vector3D left = data((i > 0) ? i - 1 : i, j, k);
    Vector3D right = data((i + 1 < ds.x) ? i + 1 : i, j, k);
    Vector3D down = data(i, (j > 0) ? j - 1 : j, k);
    Vector3D up = data(i, (j + 1 < ds.y) ? j + 1 : j, k);
    Vector3D back = data(i, j, (k > 0) ? k - 1 : k);
    Vector3D front = data(i, j, (k + 1 < ds.z) ? k + 1 : k);

    std::array<Vector3D, 3> result;
    result[0] = 0.5 * Vector3D(
        right.x - left.x, up.x - down.x, front.x - back.x) / gridSpacing;
    result[1] = 0.5 * Vector3D(
        right.y - left.y, up.y - down.y, front.y - back.y) / gridSpacing;
    result[2] = 0.5 * Vector3D(
        right.z - left.z, up.z - down.z, front.z - back.z) / gridSpacing;
    return result;
}

double laplacian2(
    const ConstArrayAccessor2<double>& data,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j) {
    const double center = data(i, j);
    const Size2 ds = data.size();

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

    return (dright - dleft) / square(gridSpacing.x)
        + (dup - ddown) / square(gridSpacing.y);
}

Vector2D laplacian2(
    const ConstArrayAccessor2<Vector2D>& data,
    const Vector2D& gridSpacing,
    size_t i,
    size_t j) {
    const Vector2D center = data(i, j);
    const Size2 ds = data.size();

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

    return (dright - dleft) / square(gridSpacing.x)
        + (dup - ddown) / square(gridSpacing.y);
}

double laplacian3(
    const ConstArrayAccessor3<double>& data,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k) {
    const double center = data(i, j, k);
    const Size3 ds = data.size();

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

    return (dright - dleft) / square(gridSpacing.x)
        + (dup - ddown) / square(gridSpacing.y)
        + (dfront - dback) / square(gridSpacing.z);
}

Vector3D laplacian3(
    const ConstArrayAccessor3<Vector3D>& data,
    const Vector3D& gridSpacing,
    size_t i,
    size_t j,
    size_t k) {
    const Vector3D center = data(i, j, k);
    const Size3 ds = data.size();

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

    return (dright - dleft) / square(gridSpacing.x)
        + (dup - ddown) / square(gridSpacing.y)
        + (dfront - dback) / square(gridSpacing.z);
}

void projectVectorFieldToSdf(
    const ScalarField2& sdf,
    CollocatedVectorGrid2* data) {
    auto pos = data->dataPosition();
    auto size = data->resolution();

    parallelFor(
        kZeroSize, size.x, kZeroSize, size.y,
        [&data, &sdf, &pos](size_t i, size_t j) {
            Vector2D pt = pos(i, j);
            Vector2D gradSdf = sdf.gradient(pt);
            if (gradSdf.lengthSquared() > 0.0) {
                Vector2D normal = gradSdf.normalized();

                Vector2D& v = (*data)(i, j);
                v -= v.dot(normal) * normal;
            }
        });
}

void projectVectorFieldToSdf(
    const ScalarField3& sdf,
    CollocatedVectorGrid3* data) {
    auto pos = data->dataPosition();
    auto size = data->resolution();

    parallelFor(
        kZeroSize, size.x, kZeroSize, size.y, kZeroSize, size.z,
        [&sdf, &data, &pos](size_t i, size_t j, size_t k) {
            Vector3D pt = pos(i, j, k);
            Vector3D gradSdf = sdf.gradient(pt);
            if (gradSdf.lengthSquared() > 0.0) {
                Vector3D normal = gradSdf.normalized();

                Vector3D& v = (*data)(i, j, k);
                v -= v.dot(normal) * normal;
            }
        });
}

void projectVectorFieldToSdf(
    const ScalarField2& sdf,
    FaceCenteredGrid2* data) {
    auto u = data->uAccessor();
    auto uPos = data->uPosition();
    auto uSize = data->uSize();

    parallelFor(
        kZeroSize, uSize.x, kZeroSize, uSize.y,
        [&sdf, &u, &uPos](size_t i, size_t j) {
            Vector2D pt = uPos(i, j);
            Vector2D gradSdf = sdf.gradient(pt);
            if (gradSdf.lengthSquared() > 0.0) {
                Vector2D normal = gradSdf.normalized();

                u(i, j) -= u(i, j) * normal.x;
            }
        });

    auto v = data->vAccessor();
    auto vPos = data->vPosition();
    auto vSize = data->vSize();

    parallelFor(
        kZeroSize, vSize.x, kZeroSize, vSize.y,
        [&sdf, &v, &vPos](size_t i, size_t j) {
            Vector2D pt = vPos(i, j);
            Vector2D gradSdf = sdf.gradient(pt);
            if (gradSdf.lengthSquared() > 0.0) {
                Vector2D normal = gradSdf.normalized();

                v(i, j) -= v(i, j) * normal.y;
            }
        });
}

void projectVectorFieldToSdf(const ScalarField3& sdf, FaceCenteredGrid3* data) {
    auto u = data->uAccessor();
    auto uPos = data->uPosition();
    auto uSize = data->uSize();

    parallelFor(kZeroSize, uSize.x, kZeroSize, uSize.y, kZeroSize, uSize.z,
        [&sdf, &u, &uPos](size_t i, size_t j, size_t k) {
            Vector3D pt = uPos(i, j, k);
            Vector3D gradSdf = sdf.gradient(pt);
            if (gradSdf.lengthSquared() > 0.0) {
                Vector3D normal = gradSdf.normalized();

                u(i, j, k) -= u(i, j, k) * normal.x;
            }
        });

    auto v = data->vAccessor();
    auto vPos = data->vPosition();
    auto vSize = data->vSize();

    parallelFor(kZeroSize, vSize.x, kZeroSize, vSize.y, kZeroSize, vSize.z,
        [&sdf, &v, &vPos](size_t i, size_t j, size_t k) {
            Vector3D pt = vPos(i, j, k);
            Vector3D gradSdf = sdf.gradient(pt);
            if (gradSdf.lengthSquared() > 0.0) {
                Vector3D normal = gradSdf.normalized();

                v(i, j, k) -= v(i, j, k) * normal.y;
            }
        });

    auto w = data->wAccessor();
    auto wPos = data->wPosition();
    auto wSize = data->wSize();

    parallelFor(kZeroSize, wSize.x, kZeroSize, wSize.y, kZeroSize, wSize.z,
        [&sdf, &w, &wPos](size_t i, size_t j, size_t k) {
            Vector3D pt = wPos(i, j, k);
            Vector3D gradSdf = sdf.gradient(pt);
            if (gradSdf.lengthSquared() > 0.0) {
                Vector3D normal = gradSdf.normalized();

                w(i, j, k) -= w(i, j, k) * normal.z;
            }
        });
}

}  // namespace jet
