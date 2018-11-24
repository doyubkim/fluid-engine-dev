// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/fdm_utils.h>
#include <jet/fmm_level_set_solver2.h>
#include <jet/level_set_utils.h>
#include <pch.h>

#include <algorithm>
#include <queue>
#include <vector>

using namespace jet;

static const char kUnknown = 0;
static const char kKnown = 1;
static const char kTrial = 2;

// Find geometric solution near the boundary
inline double solveQuadNearBoundary(const Array2<char>& markers,
                                    ArrayView2<double> output,
                                    const Vector2D& gridSpacing,
                                    const Vector2D& invGridSpacingSqr,
                                    double sign, size_t i, size_t j) {
    UNUSED_VARIABLE(markers);
    UNUSED_VARIABLE(invGridSpacingSqr);

    Vector2UZ size = output.size();

    bool hasX = false;
    double phiX = kMaxD;

    if (i > 0) {
        if (isInsideSdf(sign * output(i - 1, j))) {
            hasX = true;
            phiX = std::min(phiX, sign * output(i - 1, j));
        }
    }

    if (i + 1 < size.x) {
        if (isInsideSdf(sign * output(i + 1, j))) {
            hasX = true;
            phiX = std::min(phiX, sign * output(i + 1, j));
        }
    }

    bool hasY = false;
    double phiY = kMaxD;

    if (j > 0) {
        if (isInsideSdf(sign * output(i, j - 1))) {
            hasY = true;
            phiY = std::min(phiY, sign * output(i, j - 1));
        }
    }

    if (j + 1 < size.y) {
        if (isInsideSdf(sign * output(i, j + 1))) {
            hasY = true;
            phiY = std::min(phiY, sign * output(i, j + 1));
        }
    }

    JET_ASSERT(hasX || hasY);

    double distToBndX = gridSpacing.x * std::abs(output(i, j)) /
                        (std::abs(output(i, j)) + std::abs(phiX));

    double distToBndY = gridSpacing.y * std::abs(output(i, j)) /
                        (std::abs(output(i, j)) + std::abs(phiY));

    double solution;
    double denomSqr = 0.0;

    if (hasX) {
        denomSqr += 1.0 / square(distToBndX);
    }
    if (hasY) {
        denomSqr += 1.0 / square(distToBndY);
    }

    solution = 1.0 / std::sqrt(denomSqr);

    return sign * solution;
}

inline double solveQuad(const Array2<char>& markers, ArrayView2<double> output,
                        const Vector2D& gridSpacing,
                        const Vector2D& invGridSpacingSqr, size_t i, size_t j) {
    Vector2UZ size = output.size();

    bool hasX = false;
    double phiX = kMaxD;

    if (i > 0) {
        if (markers(i - 1, j) == kKnown) {
            hasX = true;
            phiX = std::min(phiX, output(i - 1, j));
        }
    }

    if (i + 1 < size.x) {
        if (markers(i + 1, j) == kKnown) {
            hasX = true;
            phiX = std::min(phiX, output(i + 1, j));
        }
    }

    bool hasY = false;
    double phiY = kMaxD;

    if (j > 0) {
        if (markers(i, j - 1) == kKnown) {
            hasY = true;
            phiY = std::min(phiY, output(i, j - 1));
        }
    }

    if (j + 1 < size.y) {
        if (markers(i, j + 1) == kKnown) {
            hasY = true;
            phiY = std::min(phiY, output(i, j + 1));
        }
    }

    JET_ASSERT(hasX || hasY);

    double solution = 0.0;

    // Initial guess
    if (hasX) {
        solution = phiX + gridSpacing.x;
    }
    if (hasY) {
        solution = std::max(solution, phiY + gridSpacing.y);
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

    double det = b * b - a * c;

    if (det > 0.0) {
        solution = (-b + std::sqrt(det)) / a;
    }

    return solution;
}

FmmLevelSetSolver2::FmmLevelSetSolver2() {}

void FmmLevelSetSolver2::reinitialize(const ScalarGrid2& inputSdf,
                                      double maxDistance,
                                      ScalarGrid2* outputSdf) {
    JET_THROW_INVALID_ARG_IF(!inputSdf.hasSameShape(*outputSdf));

    Vector2UZ size = inputSdf.dataSize();
    Vector2D gridSpacing = inputSdf.gridSpacing();
    Vector2D invGridSpacing = 1.0 / gridSpacing;
    Vector2D invGridSpacingSqr = elemMul(invGridSpacing, invGridSpacing);
    Array2<char> markers(size);

    auto output = outputSdf->dataView();

    parallelForEachIndex(markers.size(), [&](size_t i, size_t j) {
        output(i, j) = inputSdf(i, j);
    });

    // Solve geometrically near the boundary
    forEachIndex(markers.size(), [&](size_t i, size_t j) {
        if (!isInsideSdf(output(i, j)) &&
            ((i > 0 && isInsideSdf(output(i - 1, j))) ||
             (i + 1 < size.x && isInsideSdf(output(i + 1, j))) ||
             (j > 0 && isInsideSdf(output(i, j - 1))) ||
             (j + 1 < size.y && isInsideSdf(output(i, j + 1))))) {
            output(i, j) = solveQuadNearBoundary(markers, output, gridSpacing,
                                                 invGridSpacingSqr, 1.0, i, j);
        } else if (isInsideSdf(output(i, j)) &&
                   ((i > 0 && !isInsideSdf(output(i - 1, j))) ||
                    (i + 1 < size.x && !isInsideSdf(output(i + 1, j))) ||
                    (j > 0 && !isInsideSdf(output(i, j - 1))) ||
                    (j + 1 < size.y && !isInsideSdf(output(i, j + 1))))) {
            output(i, j) = solveQuadNearBoundary(markers, output, gridSpacing,
                                                 invGridSpacingSqr, -1.0, i, j);
        }
    });

    for (int sign = 0; sign < 2; ++sign) {
        // Build markers
        parallelForEachIndex(markers.size(), [&](size_t i, size_t j) {
            if (isInsideSdf(output(i, j))) {
                markers(i, j) = kKnown;
            } else {
                markers(i, j) = kUnknown;
            }
        });

        auto compare = [&](const Vector2UZ& a, const Vector2UZ& b) {
            return output(a.x, a.y) > output(b.x, b.y);
        };

        // Enqueue initial candidates
        std::priority_queue<Vector2UZ, std::vector<Vector2UZ>,
                            decltype(compare)>
            trial(compare);
        forEachIndex(markers.size(), [&](size_t i, size_t j) {
            if (markers(i, j) != kKnown &&
                ((i > 0 && markers(i - 1, j) == kKnown) ||
                 (i + 1 < size.x && markers(i + 1, j) == kKnown) ||
                 (j > 0 && markers(i, j - 1) == kKnown) ||
                 (j + 1 < size.y && markers(i, j + 1) == kKnown))) {
                trial.push(Vector2UZ(i, j));
                markers(i, j) = kTrial;
            }
        });

        // Propagate
        while (!trial.empty()) {
            Vector2UZ idx = trial.top();
            trial.pop();

            size_t i = idx.x;
            size_t j = idx.y;

            markers(i, j) = kKnown;
            output(i, j) = solveQuad(markers, output, gridSpacing,
                                     invGridSpacingSqr, i, j);

            if (output(i, j) > maxDistance) {
                break;
            }

            if (i > 0) {
                if (markers(i - 1, j) == kUnknown) {
                    markers(i - 1, j) = kTrial;
                    output(i - 1, j) = solveQuad(markers, output, gridSpacing,
                                                 invGridSpacingSqr, i - 1, j);
                    trial.push(Vector2UZ(i - 1, j));
                }
            }

            if (i + 1 < size.x) {
                if (markers(i + 1, j) == kUnknown) {
                    markers(i + 1, j) = kTrial;
                    output(i + 1, j) = solveQuad(markers, output, gridSpacing,
                                                 invGridSpacingSqr, i + 1, j);
                    trial.push(Vector2UZ(i + 1, j));
                }
            }

            if (j > 0) {
                if (markers(i, j - 1) == kUnknown) {
                    markers(i, j - 1) = kTrial;
                    output(i, j - 1) = solveQuad(markers, output, gridSpacing,
                                                 invGridSpacingSqr, i, j - 1);
                    trial.push(Vector2UZ(i, j - 1));
                }
            }

            if (j + 1 < size.y) {
                if (markers(i, j + 1) == kUnknown) {
                    markers(i, j + 1) = kTrial;
                    output(i, j + 1) = solveQuad(markers, output, gridSpacing,
                                                 invGridSpacingSqr, i, j + 1);
                    trial.push(Vector2UZ(i, j + 1));
                }
            }
        }

        // Flip the sign
        parallelForEachIndex(markers.size(), [&](size_t i, size_t j) {
            output(i, j) = -output(i, j);
        });
    }
}

void FmmLevelSetSolver2::extrapolate(const ScalarGrid2& input,
                                     const ScalarField2& sdf,
                                     double maxDistance, ScalarGrid2* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    Array2<double> sdfGrid(input.dataSize());
    auto pos = unroll2(input.dataPosition());
    parallelForEachIndex(sdfGrid.size(), [&](size_t i, size_t j) {
        sdfGrid(i, j) = sdf.sample(pos(i, j));
    });

    extrapolate(input.dataView(), sdfGrid, input.gridSpacing(), maxDistance,
                output->dataView());
}

void FmmLevelSetSolver2::extrapolate(const CollocatedVectorGrid2& input,
                                     const ScalarField2& sdf,
                                     double maxDistance,
                                     CollocatedVectorGrid2* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    Array2<double> sdfGrid(input.dataSize());
    auto pos = unroll2(input.dataPosition());
    parallelForEachIndex(sdfGrid.size(), [&](size_t i, size_t j) {
        sdfGrid(i, j) = sdf.sample(pos(i, j));
    });

    const Vector2D gridSpacing = input.gridSpacing();

    Array2<double> u(input.dataSize());
    Array2<double> u0(input.dataSize());
    Array2<double> v(input.dataSize());
    Array2<double> v0(input.dataSize());

    input.parallelForEachDataPointIndex([&](const Vector2UZ& idx) {
        u(idx) = input(idx).x;
        v(idx) = input(idx).y;
    });

    extrapolate(u, sdfGrid, gridSpacing, maxDistance, u0);

    extrapolate(v, sdfGrid, gridSpacing, maxDistance, v0);

    output->parallelForEachDataPointIndex([&](const Vector2UZ& idx) {
        (*output)(idx).x = u(idx);
        (*output)(idx).y = v(idx);
    });
}

void FmmLevelSetSolver2::extrapolate(const FaceCenteredGrid2& input,
                                     const ScalarField2& sdf,
                                     double maxDistance,
                                     FaceCenteredGrid2* output) {
    JET_THROW_INVALID_ARG_IF(!input.hasSameShape(*output));

    const Vector2D gridSpacing = input.gridSpacing();

    auto u = input.uView();
    auto uPos = input.uPosition();
    Array2<double> sdfAtU(u.size());
    input.parallelForEachUIndex(
        [&](const Vector2UZ& idx) { sdfAtU(idx) = sdf.sample(uPos(idx)); });

    extrapolate(u, sdfAtU, gridSpacing, maxDistance, output->uView());

    auto v = input.vView();
    auto vPos = input.vPosition();
    Array2<double> sdfAtV(v.size());
    input.parallelForEachVIndex(
        [&](const Vector2UZ& idx) { sdfAtV(idx) = sdf.sample(vPos(idx)); });

    extrapolate(v, sdfAtV, gridSpacing, maxDistance, output->vView());
}

void FmmLevelSetSolver2::extrapolate(const ConstArrayView2<double>& input,
                                     const ConstArrayView2<double>& sdf,
                                     const Vector2D& gridSpacing,
                                     double maxDistance,
                                     ArrayView2<double> output) {
    Vector2UZ size = input.size();
    Vector2D invGridSpacing = 1.0 / gridSpacing;

    // Build markers
    Array2<char> markers(size, kUnknown);
    parallelForEachIndex(markers.size(), [&](size_t i, size_t j) {
        if (isInsideSdf(sdf(i, j))) {
            markers(i, j) = kKnown;
        }
        output(i, j) = input(i, j);
    });

    auto compare = [&](const Vector2UZ& a, const Vector2UZ& b) {
        return sdf(a.x, a.y) > sdf(b.x, b.y);
    };

    // Enqueue initial candidates
    std::priority_queue<Vector2UZ, std::vector<Vector2UZ>, decltype(compare)>
        trial(compare);
    forEachIndex(markers.size(), [&](size_t i, size_t j) {
        if (markers(i, j) == kKnown) {
            return;
        }

        if (i > 0 && markers(i - 1, j) == kKnown) {
            trial.push(Vector2UZ(i, j));
            markers(i, j) = kTrial;
            return;
        }

        if (i + 1 < size.x && markers(i + 1, j) == kKnown) {
            trial.push(Vector2UZ(i, j));
            markers(i, j) = kTrial;
            return;
        }

        if (j > 0 && markers(i, j - 1) == kKnown) {
            trial.push(Vector2UZ(i, j));
            markers(i, j) = kTrial;
            return;
        }

        if (j + 1 < size.y && markers(i, j + 1) == kKnown) {
            trial.push(Vector2UZ(i, j));
            markers(i, j) = kTrial;
            return;
        }
    });

    // Propagate
    while (!trial.empty()) {
        Vector2UZ idx = trial.top();
        trial.pop();

        size_t i = idx.x;
        size_t j = idx.y;

        if (sdf(i, j) > maxDistance) {
            break;
        }

        Vector2D grad = gradient2(sdf, gridSpacing, i, j).normalized();

        double sum = 0.0;
        double count = 0.0;

        if (i > 0) {
            if (markers(i - 1, j) == kKnown) {
                double weight = std::max(grad.x, 0.0) * invGridSpacing.x;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i - 1, j);
                count += weight;
            } else if (markers(i - 1, j) == kUnknown) {
                markers(i - 1, j) = kTrial;
                trial.push(Vector2UZ(i - 1, j));
            }
        }

        if (i + 1 < size.x) {
            if (markers(i + 1, j) == kKnown) {
                double weight = -std::min(grad.x, 0.0) * invGridSpacing.x;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i + 1, j);
                count += weight;
            } else if (markers(i + 1, j) == kUnknown) {
                markers(i + 1, j) = kTrial;
                trial.push(Vector2UZ(i + 1, j));
            }
        }

        if (j > 0) {
            if (markers(i, j - 1) == kKnown) {
                double weight = std::max(grad.y, 0.0) * invGridSpacing.y;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i, j - 1);
                count += weight;
            } else if (markers(i, j - 1) == kUnknown) {
                markers(i, j - 1) = kTrial;
                trial.push(Vector2UZ(i, j - 1));
            }
        }

        if (j + 1 < size.y) {
            if (markers(i, j + 1) == kKnown) {
                double weight = -std::min(grad.y, 0.0) * invGridSpacing.y;

                // If gradient is zero, then just assign 1 to weight
                if (weight < kEpsilonD) {
                    weight = 1.0;
                }

                sum += weight * output(i, j + 1);
                count += weight;
            } else if (markers(i, j + 1) == kUnknown) {
                markers(i, j + 1) = kTrial;
                trial.push(Vector2UZ(i, j + 1));
            }
        }

        JET_ASSERT(count > 0.0);

        output(i, j) = sum / count;
        markers(i, j) = kKnown;
    }
}
