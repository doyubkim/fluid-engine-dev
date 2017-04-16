// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <pch.h>
#include <jet/fdm_utils.h>
#include <jet/fmm_level_set_solver3.h>
#include <jet/level_set_utils.h>

#include <algorithm>
#include <vector>
#include <queue>

using namespace jet;

static const char kUnknown = 0;
static const char kKnown = 1;
static const char kTrial = 2;

// Find geometric solution near the boundary
inline double solveQuadNearBoundary(
    const Array3<char>& markers,
    ArrayAccessor3<double> output,
    const Vector3D& gridSpacing,
    const Vector3D& invGridSpacingSqr,
    double sign,
    size_t i,
    size_t j,
    size_t k) {
    UNUSED_VARIABLE(markers);
    UNUSED_VARIABLE(invGridSpacingSqr);

    Size3 size = output.size();

    bool hasX = false;
    double phiX = kMaxD;

    if (i > 0) {
        if (isInsideSdf(sign * output(i - 1, j, k))) {
            hasX = true;
            phiX = std::min(phiX, sign * output(i - 1, j, k));
        }
    }

    if (i + 1 < size.x) {
        if (isInsideSdf(sign * output(i + 1, j, k))) {
            hasX = true;
            phiX = std::min(phiX, sign * output(i + 1, j, k));
        }
    }

    bool hasY = false;
    double phiY = kMaxD;

    if (j > 0) {
        if (isInsideSdf(sign * output(i, j - 1, k))) {
            hasY = true;
            phiY = std::min(phiY, sign * output(i, j - 1, k));
        }
    }

    if (j + 1 < size.y) {
        if (isInsideSdf(sign * output(i, j + 1, k))) {
            hasY = true;
            phiY = std::min(phiY, sign * output(i, j + 1, k));
        }
    }

    bool hasZ = false;
    double phiZ = kMaxD;

    if (k > 0) {
        if (isInsideSdf(sign * output(i, j, k - 1))) {
            hasZ = true;
            phiZ = std::min(phiZ, sign * output(i, j, k - 1));
        }
    }

    if (k + 1 < size.z) {
        if (isInsideSdf(sign * output(i, j, k + 1))) {
            hasZ = true;
            phiZ = std::min(phiZ, sign * output(i, j, k + 1));
        }
    }

    JET_ASSERT(hasX || hasY || hasZ);

    double distToBndX
        = gridSpacing.x * std::abs(output(i, j, k))
        / (std::abs(output(i, j, k)) + std::abs(phiX));

    double distToBndY
        = gridSpacing.y * std::abs(output(i, j, k))
        / (std::abs(output(i, j, k)) + std::abs(phiY));

    double distToBndZ
        = gridSpacing.z * std::abs(output(i, j, k))
        / (std::abs(output(i, j, k)) + std::abs(phiZ));

    double solution;
    double denomSqr = 0.0;

    if (hasX) {
        denomSqr += 1.0 / square(distToBndX);
    }
    if (hasY) {
        denomSqr += 1.0 / square(distToBndY);
    }
    if (hasZ) {
        denomSqr += 1.0 / square(distToBndZ);
    }

    solution = 1.0 / std::sqrt(denomSqr);

    return sign * solution;
}

inline double solveQuad(
    const Array3<char>& markers,
    ArrayAccessor3<double> output,
    const Vector3D& gridSpacing,
    const Vector3D& invGridSpacingSqr,
    size_t i,
    size_t j,
    size_t k) {
    Size3 size = output.size();

    bool hasX = false;
    double phiX = kMaxD;

    if (i > 0) {
        if (markers(i - 1, j, k) == kKnown) {
            hasX = true;
            phiX = std::min(phiX, output(i - 1, j, k));
        }
    }

    if (i + 1 < size.x) {
        if (markers(i + 1, j, k) == kKnown) {
            hasX = true;
            phiX = std::min(phiX, output(i + 1, j, k));
        }
    }

    bool hasY = false;
    double phiY = kMaxD;

    if (j > 0) {
        if (markers(i, j - 1, k) == kKnown) {
            hasY = true;
            phiY = std::min(phiY, output(i, j - 1, k));
        }
    }

    if (j + 1 < size.y) {
        if (markers(i, j + 1, k) == kKnown) {
            hasY = true;
            phiY = std::min(phiY, output(i, j + 1, k));
        }
    }

    bool hasZ = false;
    double phiZ = kMaxD;

    if (k > 0) {
        if (markers(i, j, k - 1) == kKnown) {
            hasZ = true;
            phiZ = std::min(phiZ, output(i, j, k - 1));
        }
    }

    if (k + 1 < size.z) {
        if (markers(i, j, k + 1) == kKnown) {
            hasZ = true;
            phiZ = std::min(phiZ, output(i, j, k + 1));
        }
    }

    JET_ASSERT(hasX || hasY || hasZ);

    double solution = 0.0;

    // Initial guess
    if (hasX) {
        solution = std::max(solution, phiX + gridSpacing.x);
    }
    if (hasY) {
        solution = std::max(solution, phiY + gridSpacing.y);
    }
    if (hasZ) {
        solution = std::max(solution, phiZ + gridSpacing.z);
    }

    // Solve quad
    double a = 0.0;
    double b = 0.0;
    double c = -1.0;

    if (hasX) {
        a += invGridSpacingSqr.x;
        b -= phiX * invGridSpacingSqr.x;
        c += square(phiX) * invGridSpacingSqr.x;
    }
    if (hasY) {
        a += invGridSpacingSqr.y;
        b -= phiY * invGridSpacingSqr.y;
        c += square(phiY) * invGridSpacingSqr.y;
    }
    if (hasZ) {
        a += invGridSpacingSqr.z;
        b -= phiZ * invGridSpacingSqr.z;
        c += square(phiZ) * invGridSpacingSqr.z;
    }

    double det = b * b - a * c;

    if (det > 0.0) {
        solution = (-b + std::sqrt(det)) / a;
    }

    return solution;
}

FmmLevelSetSolver3::FmmLevelSetSolver3() {
}

void FmmLevelSetSolver3::reinitialize(
    const ScalarGrid3& inputSdf,
    double maxDistance,
    ScalarGrid3* outputSdf) {
    JET_THROW_INVALID_ARG_IF(!inputSdf.hasSameShape(*outputSdf));

    Size3 size = inputSdf.dataSize();
    Vector3D gridSpacing = inputSdf.gridSpacing();
    Vector3D invGridSpacing = 1.0 / gridSpacing;
    Vector3D invGridSpacingSqr = invGridSpacing * invGridSpacing;
    Array3<char> markers(size);

    auto output = outputSdf->dataAccessor();

    markers.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        output(i, j, k) = inputSdf(i, j, k);
    });

    // Solve geometrically near the boundary
    markers.forEachIndex([&](size_t i, size_t j, size_t k) {
        if (!isInsideSdf(output(i, j, k))
            && ((i > 0 && isInsideSdf(output(i - 1, j, k)))
             || (i + 1 < size.x && isInsideSdf(output(i + 1, j, k)))
             || (j > 0 && isInsideSdf(output(i, j - 1, k)))
             || (j + 1 < size.y && isInsideSdf(output(i, j + 1, k)))
             || (k > 0 && isInsideSdf(output(i, j, k - 1)))
             || (k + 1 < size.z && isInsideSdf(output(i, j, k + 1))))) {
            output(i, j, k) = solveQuadNearBoundary(
                markers, output, gridSpacing, invGridSpacingSqr, 1.0, i, j, k);
        } else if (isInsideSdf(output(i, j, k))
            && ((i > 0 && !isInsideSdf(output(i - 1, j, k)))
             || (i + 1 < size.x && !isInsideSdf(output(i + 1, j, k)))
             || (j > 0 && !isInsideSdf(output(i, j - 1, k)))
             || (j + 1 < size.y && !isInsideSdf(output(i, j + 1, k)))
             || (k > 0 && !isInsideSdf(output(i, j, k - 1)))
             || (k + 1 < size.z && !isInsideSdf(output(i, j, k + 1))))) {
            output(i, j, k) = solveQuadNearBoundary(
                markers, output, gridSpacing, invGridSpacingSqr, -1.0, i, j, k);
        }
    });

    for (int sign = 0; sign < 2; ++sign) {
        // Build markers
        markers.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
            if (isInsideSdf(output(i, j, k))) {
                markers(i, j, k) = kKnown;
            } else {
                markers(i, j, k) = kUnknown;
            }
        });

        auto compare = [&](const Point3UI& a, const Point3UI& b) {
            return output(a.x, a.y, a.z) > output(b.x, b.y, b.z);
        };

        // Enqueue initial candidates
        std::priority_queue<
            Point3UI, std::vector<Point3UI>, decltype(compare)> trial(compare);
        markers.forEachIndex([&](size_t i, size_t j, size_t k) {
            if (markers(i, j, k) != kKnown
                && ((i > 0 && markers(i - 1, j, k) == kKnown)
                 || (i + 1 < size.x && markers(i + 1, j, k) == kKnown)
                 || (j > 0 && markers(i, j - 1, k) == kKnown)
                 || (j + 1 < size.y && markers(i, j + 1, k) == kKnown)
                 || (k > 0 && markers(i, j, k - 1) == kKnown)
                 || (k + 1 < size.z && markers(i, j, k + 1) == kKnown))) {
                trial.push(Point3UI(i, j, k));
                markers(i, j, k) = kTrial;
            }
        });

        // Propagate
        while (!trial.empty()) {
            Point3UI idx = trial.top();
            trial.pop();

            size_t i = idx.x;
            size_t j = idx.y;
            size_t k = idx.z;

            markers(i, j, k) = kKnown;
            output(i, j, k) = solveQuad(
                markers, output, gridSpacing, invGridSpacingSqr, i, j, k);

            if (output(i, j, k) > maxDistance) {
                break;
            }

            if (i > 0) {
                if (markers(i - 1, j, k) == kUnknown) {
                    markers(i - 1, j, k) = kTrial;
                    output(i - 1, j, k) = solveQuad(
                        markers,
                        output,
                        gridSpacing,
                        invGridSpacingSqr,
                        i - 1,
                        j,
                        k);
                    trial.push(Point3UI(i - 1, j, k));
                }
            }

            if (i + 1 < size.x) {
                if (markers(i + 1, j, k) == kUnknown) {
                    markers(i + 1, j, k) = kTrial;
                    output(i + 1, j, k) = solveQuad(
                        markers,
                        output,
                        gridSpacing,
                        invGridSpacingSqr,
                        i + 1,
                        j,
                        k);
                    trial.push(Point3UI(i + 1, j, k));
                }
            }

            if (j > 0) {
                if (markers(i, j - 1, k) == kUnknown) {
                    markers(i, j - 1, k) = kTrial;
                    output(i, j - 1, k) = solveQuad(
                        markers,
                        output,
                        gridSpacing,
                        invGridSpacingSqr,
                        i,
                        j - 1,
                        k);
                    trial.push(Point3UI(i, j - 1, k));
                }
            }

            if (j + 1 < size.y) {
                if (markers(i, j + 1, k) == kUnknown) {
                    markers(i, j + 1, k) = kTrial;
                    output(i, j + 1, k) = solveQuad(
                        markers,
                        output,
                        gridSpacing,
                        invGridSpacingSqr,
                        i,
                        j + 1,
                        k);
                    trial.push(Point3UI(i, j + 1, k));
                }
            }

            if (k > 0) {
                if (markers(i, j, k - 1) == kUnknown) {
                    markers(i, j, k - 1) = kTrial;
                    output(i, j, k - 1) = solveQuad(
                        markers,
                        output,
                        gridSpacing,
                        invGridSpacingSqr,
                        i,
                        j,
                        k - 1);
                    trial.push(Point3UI(i, j, k - 1));
                }
            }

            if (k + 1 < size.z) {
                if (markers(i, j, k + 1) == kUnknown) {
                    markers(i, j, k + 1) = kTrial;
                    output(i, j, k + 1) = solveQuad(
                        markers,
                        output,
                        gridSpacing,
                        invGridSpacingSqr,
                        i,
                        j,
                        k + 1);
                    trial.push(Point3UI(i, j, k + 1));
                }
            }
        }

        // Flip the sign
        markers.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
            output(i, j, k) = -output(i, j, k);
        });
    }
}

void FmmLevelSetSolver3::extrapolate(
    const ScalarGrid3& input,
    const ScalarField3& sdf,
    double maxDistance,
    ScalarGrid3* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    Array3<double> sdfGrid(input.dataSize());
    auto pos = input.dataPosition();
    sdfGrid.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        sdfGrid(i, j, k) = sdf.sample(pos(i, j, k));
    });

    extrapolate(
        input.constDataAccessor(),
        sdfGrid.constAccessor(),
        input.gridSpacing(),
        maxDistance,
        output->dataAccessor());
}

void FmmLevelSetSolver3::extrapolate(
    const CollocatedVectorGrid3& input,
    const ScalarField3& sdf,
    double maxDistance,
    CollocatedVectorGrid3* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    Array3<double> sdfGrid(input.dataSize());
    auto pos = input.dataPosition();
    sdfGrid.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        sdfGrid(i, j, k) = sdf.sample(pos(i, j, k));
    });

    const Vector3D gridSpacing = input.gridSpacing();

    Array3<double> u(input.dataSize());
    Array3<double> u0(input.dataSize());
    Array3<double> v(input.dataSize());
    Array3<double> v0(input.dataSize());
    Array3<double> w(input.dataSize());
    Array3<double> w0(input.dataSize());

    input.parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        u(i, j, k) = input(i, j, k).x;
        v(i, j, k) = input(i, j, k).y;
        w(i, j, k) = input(i, j, k).z;
    });

    extrapolate(
        u,
        sdfGrid.constAccessor(),
        gridSpacing,
        maxDistance,
        u0);

    extrapolate(
        v,
        sdfGrid.constAccessor(),
        gridSpacing,
        maxDistance,
        v0);

    extrapolate(
        w,
        sdfGrid.constAccessor(),
        gridSpacing,
        maxDistance,
        w0);

    output->parallelForEachDataPointIndex([&](size_t i, size_t j, size_t k) {
        (*output)(i, j, k).x = u(i, j, k);
        (*output)(i, j, k).y = v(i, j, k);
        (*output)(i, j, k).z = w(i, j, k);
    });
}

void FmmLevelSetSolver3::extrapolate(
    const FaceCenteredGrid3& input,
    const ScalarField3& sdf,
    double maxDistance,
    FaceCenteredGrid3* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    const Vector3D gridSpacing = input.gridSpacing();

    auto u = input.uConstAccessor();
    auto uPos = input.uPosition();
    Array3<double> sdfAtU(u.size());
    input.parallelForEachUIndex([&](size_t i, size_t j, size_t k) {
        sdfAtU(i, j, k) = sdf.sample(uPos(i, j, k));
    });

    extrapolate(
        u,
        sdfAtU,
        gridSpacing,
        maxDistance,
        output->uAccessor());

    auto v = input.vConstAccessor();
    auto vPos = input.vPosition();
    Array3<double> sdfAtV(v.size());
    input.parallelForEachVIndex([&](size_t i, size_t j, size_t k) {
        sdfAtV(i, j, k) = sdf.sample(vPos(i, j, k));
    });

    extrapolate(
        v,
        sdfAtV,
        gridSpacing,
        maxDistance,
        output->vAccessor());

    auto w = input.wConstAccessor();
    auto wPos = input.wPosition();
    Array3<double> sdfAtW(w.size());
    input.parallelForEachWIndex([&](size_t i, size_t j, size_t k) {
        sdfAtW(i, j, k) = sdf.sample(wPos(i, j, k));
    });

    extrapolate(
        w,
        sdfAtW,
        gridSpacing,
        maxDistance,
        output->wAccessor());
}

void FmmLevelSetSolver3::extrapolate(
    const ConstArrayAccessor3<double>& input,
    const ConstArrayAccessor3<double>& sdf,
    const Vector3D& gridSpacing,
    double maxDistance,
    ArrayAccessor3<double> output) {
    Size3 size = input.size();
    Vector3D invGridSpacing = 1.0 / gridSpacing;

    // Build markers
    Array3<char> markers(size, kUnknown);
    markers.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
        if (isInsideSdf(sdf(i, j, k))) {
            markers(i, j, k) = kKnown;
        }
        output(i, j, k) = input(i, j, k);
    });

    auto compare = [&](const Point3UI& a, const Point3UI& b) {
        return sdf(a.x, a.y, a.z) > sdf(b.x, b.y, b.z);
    };

    // Enqueue initial candidates
    std::priority_queue<
        Point3UI, std::vector<Point3UI>, decltype(compare)> trial(compare);
    markers.forEachIndex([&](size_t i, size_t j, size_t k) {
        if (markers(i, j, k) == kKnown) {
            return;
        }

        if (i > 0 && markers(i - 1, j, k) == kKnown) {
            trial.push(Point3UI(i, j, k));
            markers(i, j, k) = kTrial;
            return;
        }

        if (i + 1 < size.x && markers(i + 1, j, k) == kKnown) {
            trial.push(Point3UI(i, j, k));
            markers(i, j, k) = kTrial;
            return;
        }

        if (j > 0 && markers(i, j - 1, k) == kKnown) {
            trial.push(Point3UI(i, j, k));
            markers(i, j, k) = kTrial;
            return;
        }

        if (j + 1 < size.y && markers(i, j + 1, k) == kKnown) {
            trial.push(Point3UI(i, j, k));
            markers(i, j, k) = kTrial;
            return;
        }

        if (k > 0 && markers(i, j, k - 1) == kKnown) {
            trial.push(Point3UI(i, j, k));
            markers(i, j, k) = kTrial;
            return;
        }

        if (k + 1 < size.z && markers(i, j, k + 1) == kKnown) {
            trial.push(Point3UI(i, j, k));
            markers(i, j, k) = kTrial;
            return;
        }
    });

    // Propagate
    while (!trial.empty()) {
        Point3UI idx = trial.top();
        trial.pop();

        size_t i = idx.x;
        size_t j = idx.y;
        size_t k = idx.z;

        if (sdf(i, j, k) > maxDistance) {
            break;
        }

        Vector3D grad = gradient3(sdf, gridSpacing, i, j, k).normalized();

        double sum = 0.0;
        double count = 0.0;

        if (i > 0) {
            if (markers(i - 1, j, k) == kKnown) {
                double weight = std::max(grad.x, 0.0) * invGridSpacing.x;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i - 1, j, k);
                count += weight;
            } else if (markers(i - 1, j, k) == kUnknown) {
                markers(i - 1, j, k) = kTrial;
                trial.push(Point3UI(i - 1, j, k));
            }
        }

        if (i + 1 < size.x) {
            if (markers(i + 1, j, k) == kKnown) {
                double weight = -std::min(grad.x, 0.0) * invGridSpacing.x;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i + 1, j, k);
                count += weight;
            } else if (markers(i + 1, j, k) == kUnknown) {
                markers(i + 1, j, k) = kTrial;
                trial.push(Point3UI(i + 1, j, k));
            }
        }

        if (j > 0) {
            if (markers(i, j - 1, k) == kKnown) {
                double weight = std::max(grad.y, 0.0) * invGridSpacing.y;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i, j - 1, k);
                count += weight;
            } else if (markers(i, j - 1, k) == kUnknown) {
                markers(i, j - 1, k) = kTrial;
                trial.push(Point3UI(i, j - 1, k));
            }
        }

        if (j + 1 < size.y) {
            if (markers(i, j + 1, k) == kKnown) {
                double weight = -std::min(grad.y, 0.0) * invGridSpacing.y;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i, j + 1, k);
                count += weight;
            } else if (markers(i, j + 1, k) == kUnknown) {
                markers(i, j + 1, k) = kTrial;
                trial.push(Point3UI(i, j + 1, k));
            }
        }

        if (k > 0) {
            if (markers(i, j, k - 1) == kKnown) {
                double weight = std::max(grad.z, 0.0) * invGridSpacing.z;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i, j, k - 1);
                count += weight;
            } else if (markers(i, j, k - 1) == kUnknown) {
                markers(i, j, k - 1) = kTrial;
                trial.push(Point3UI(i, j, k - 1));
            }
        }

        if (k + 1 < size.z) {
            if (markers(i, j, k + 1) == kKnown) {
                double weight = -std::min(grad.z, 0.0) * invGridSpacing.z;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i, j, k + 1);
                count += weight;
            } else if (markers(i, j, k + 1) == kUnknown) {
                markers(i, j, k + 1) = kTrial;
                trial.push(Point3UI(i, j, k + 1));
            }
        }

        JET_ASSERT(count > 0.0);

        output(i, j, k) = sum / count;
        markers(i, j, k) = kKnown;
    }
}
