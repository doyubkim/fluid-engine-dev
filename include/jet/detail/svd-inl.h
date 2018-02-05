// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.
//
// This implementation is adopted from Numerical Recipes
// http://numerical.recipes/webnotes/nr3web2.pdf
//

#ifndef INCLUDE_JET_DETAIL_SVD_H_
#define INCLUDE_JET_DETAIL_SVD_H_

#include <jet/math_utils.h>

namespace jet {

namespace internal {

template <typename T>
inline T sign(T a, T b) {
    return b >= 0.0 ? std::fabs(a) : -std::fabs(a);
}

template <typename T>
inline T pythag(T a, T b) {
    T at = std::fabs(a);
    T bt = std::fabs(b);
    T ct;
    T result;

    if (at > bt) {
        ct = bt / at;
        result = at * std::sqrt(1 + ct * ct);
    } else if (bt > 0) {
        ct = at / bt;
        result = bt * std::sqrt(1 + ct * ct);
    } else {
        result = 0;
    }

    return result;
}

}  // namespace internal

template <typename T>
void svd(const MatrixMxN<T>& a, MatrixMxN<T>& u, VectorN<T>& w,
         MatrixMxN<T>& v) {
    const int m = (int)a.rows();
    const int n = (int)a.cols();

    int flag, i = 0, its = 0, j = 0, jj = 0, k = 0, l = 0, nm = 0;
    T c = 0, f = 0, h = 0, s = 0, x = 0, y = 0, z = 0;
    T anorm = 0, g = 0, scale = 0;

    JET_THROW_INVALID_ARG_WITH_MESSAGE_IF(m < n,
                                          "Number of rows of input matrix must "
                                          "be greater than or equal to "
                                          "columns.");

    // Prepare workspace
    VectorN<T> rv1(n, 0);
    u = a;
    w.resize(n, 0);
    v.resize(n, n, 0);

    // Householder reduction to bidiagonal form
    for (i = 0; i < n; i++) {
        // left-hand reduction
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0;
        if (i < m) {
            for (k = i; k < m; k++) {
                scale += std::fabs(u(k, i));
            }
            if (scale) {
                for (k = i; k < m; k++) {
                    u(k, i) /= scale;
                    s += u(k, i) * u(k, i);
                }
                f = u(i, i);
                g = -internal::sign(std::sqrt(s), f);
                h = f * g - s;
                u(i, i) = f - g;
                if (i != n - 1) {
                    for (j = l; j < n; j++) {
                        for (s = 0, k = i; k < m; k++) {
                            s += u(k, i) * u(k, j);
                        }
                        f = s / h;
                        for (k = i; k < m; k++) {
                            u(k, j) += f * u(k, i);
                        }
                    }
                }
                for (k = i; k < m; k++) {
                    u(k, i) *= scale;
                }
            }
        }
        w[i] = scale * g;

        // right-hand reduction
        g = s = scale = 0;
        if (i < m && i != n - 1) {
            for (k = l; k < n; k++) {
                scale += std::fabs(u(i, k));
            }
            if (scale) {
                for (k = l; k < n; k++) {
                    u(i, k) /= scale;
                    s += u(i, k) * u(i, k);
                }
                f = u(i, l);
                g = -internal::sign(std::sqrt(s), f);
                h = f * g - s;
                u(i, l) = f - g;
                for (k = l; k < n; k++) {
                    rv1[k] = (T)u(i, k) / h;
                }
                if (i != m - 1) {
                    for (j = l; j < m; j++) {
                        for (s = 0, k = l; k < n; k++) {
                            s += u(j, k) * u(i, k);
                        }
                        for (k = l; k < n; k++) {
                            u(j, k) += s * rv1[k];
                        }
                    }
                }
                for (k = l; k < n; k++) {
                    u(i, k) *= scale;
                }
            }
        }
        anorm = std::max(anorm, (std::fabs((T)w[i]) + std::fabs(rv1[i])));
    }

    // accumulate the right-hand transformation
    for (i = n - 1; i >= 0; i--) {
        if (i < n - 1) {
            if (g) {
                for (j = l; j < n; j++) {
                    v(j, i) = ((u(i, j) / u(i, l)) / g);
                }
                // T division to avoid underflow
                for (j = l; j < n; j++) {
                    for (s = 0, k = l; k < n; k++) {
                        s += u(i, k) * v(k, j);
                    }
                    for (k = l; k < n; k++) {
                        v(k, j) += s * v(k, i);
                    }
                }
            }
            for (j = l; j < n; j++) {
                v(i, j) = v(j, i) = 0;
            }
        }
        v(i, i) = 1;
        g = rv1[i];
        l = i;
    }

    // accumulate the left-hand transformation
    for (i = n - 1; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        if (i < n - 1) {
            for (j = l; j < n; j++) {
                u(i, j) = 0;
            }
        }
        if (g) {
            g = 1 / g;
            if (i != n - 1) {
                for (j = l; j < n; j++) {
                    for (s = 0, k = l; k < m; k++) {
                        s += u(k, i) * u(k, j);
                    }
                    f = (s / u(i, i)) * g;
                    for (k = i; k < m; k++) {
                        u(k, j) += f * u(k, i);
                    }
                }
            }
            for (j = i; j < m; j++) {
                u(j, i) = u(j, i) * g;
            }
        } else {
            for (j = i; j < m; j++) {
                u(j, i) = 0;
            }
        }
        ++u(i, i);
    }

    // diagonalize the bidiagonal form
    for (k = n - 1; k >= 0; k--) {
        // loop over singular values
        for (its = 0; its < 30; its++) {
            // loop over allowed iterations
            flag = 1;
            for (l = k; l >= 0; l--) {
                // test for splitting
                nm = l - 1;
                if (std::fabs(rv1[l]) + anorm == anorm) {
                    flag = 0;
                    break;
                }
                if (std::fabs((T)w[nm]) + anorm == anorm) {
                    break;
                }
            }
            if (flag) {
                c = 0;
                s = 1;
                for (i = l; i <= k; i++) {
                    f = s * rv1[i];
                    if (std::fabs(f) + anorm != anorm) {
                        g = w[i];
                        h = internal::pythag(f, g);
                        w[i] = (T)h;
                        h = 1 / h;
                        c = g * h;
                        s = -f * h;
                        for (j = 0; j < m; j++) {
                            y = u(j, nm);
                            z = u(j, i);
                            u(j, nm) = y * c + z * s;
                            u(j, i) = z * c - y * s;
                        }
                    }
                }
            }
            z = w[k];
            if (l == k) {
                // convergence
                if (z < 0) {
                    // make singular value nonnegative
                    w[k] = -z;
                    for (j = 0; j < n; j++) {
                        v(j, k) = -v(j, k);
                    }
                }
                break;
            }
            if (its >= 30) {
                throw("No convergence after 30 iterations");
            }

            // shift from bottom 2 x 2 minor
            x = w[l];
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
            g = internal::pythag(f, (T)1);
            f = ((x - z) * (x + z) +
                 h * ((y / (f + internal::sign(g, f))) - h)) /
                x;

            // next QR transformation
            c = s = 1;
            for (j = l; j <= nm; j++) {
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = internal::pythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++) {
                    x = v(jj, j);
                    z = v(jj, i);
                    v(jj, j) = x * c + z * s;
                    v(jj, i) = z * c - x * s;
                }
                z = internal::pythag(f, h);
                w[j] = z;
                if (z) {
                    z = 1 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++) {
                    y = u(jj, j);
                    z = u(jj, i);
                    u(jj, j) = y * c + z * s;
                    u(jj, i) = z * c - y * s;
                }
            }
            rv1[l] = 0;
            rv1[k] = f;
            w[k] = x;
        }
    }
}

template <typename T, size_t M, size_t N>
void svd(const Matrix<T, M, N>& a, Matrix<T, M, N>& u, Vector<T, N>& w,
         Matrix<T, N, N>& v) {
    const int m = (int)M;
    const int n = (int)N;

    int flag, i = 0, its = 0, j = 0, jj = 0, k = 0, l = 0, nm = 0;
    T c = 0, f = 0, h = 0, s = 0, x = 0, y = 0, z = 0;
    T anorm = 0, g = 0, scale = 0;

    static_assert(M >= N,
                  "Number of rows of input matrix must be greater than or "
                  "equal to columns.");

    // Prepare workspace
    Vector<T, N> rv1;
    u = a;
    w = Vector<T, N>();
    v = Matrix<T, N, N>();

    // Householder reduction to bidiagonal form
    for (i = 0; i < n; i++) {
        // left-hand reduction
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0;
        if (i < m) {
            for (k = i; k < m; k++) {
                scale += std::fabs(u(k, i));
            }
            if (scale) {
                for (k = i; k < m; k++) {
                    u(k, i) /= scale;
                    s += u(k, i) * u(k, i);
                }
                f = u(i, i);
                g = -internal::sign(std::sqrt(s), f);
                h = f * g - s;
                u(i, i) = f - g;
                if (i != n - 1) {
                    for (j = l; j < n; j++) {
                        for (s = 0, k = i; k < m; k++) {
                            s += u(k, i) * u(k, j);
                        }
                        f = s / h;
                        for (k = i; k < m; k++) {
                            u(k, j) += f * u(k, i);
                        }
                    }
                }
                for (k = i; k < m; k++) {
                    u(k, i) *= scale;
                }
            }
        }
        w[i] = scale * g;

        // right-hand reduction
        g = s = scale = 0;
        if (i < m && i != n - 1) {
            for (k = l; k < n; k++) {
                scale += std::fabs(u(i, k));
            }
            if (scale) {
                for (k = l; k < n; k++) {
                    u(i, k) /= scale;
                    s += u(i, k) * u(i, k);
                }
                f = u(i, l);
                g = -internal::sign(std::sqrt(s), f);
                h = f * g - s;
                u(i, l) = f - g;
                for (k = l; k < n; k++) {
                    rv1[k] = (T)u(i, k) / h;
                }
                if (i != m - 1) {
                    for (j = l; j < m; j++) {
                        for (s = 0, k = l; k < n; k++) {
                            s += u(j, k) * u(i, k);
                        }
                        for (k = l; k < n; k++) {
                            u(j, k) += s * rv1[k];
                        }
                    }
                }
                for (k = l; k < n; k++) {
                    u(i, k) *= scale;
                }
            }
        }
        anorm = std::max(anorm, (std::fabs((T)w[i]) + std::fabs(rv1[i])));
    }

    // accumulate the right-hand transformation
    for (i = n - 1; i >= 0; i--) {
        if (i < n - 1) {
            if (g) {
                for (j = l; j < n; j++) {
                    v(j, i) = ((u(i, j) / u(i, l)) / g);
                }
                // T division to avoid underflow
                for (j = l; j < n; j++) {
                    for (s = 0, k = l; k < n; k++) {
                        s += u(i, k) * v(k, j);
                    }
                    for (k = l; k < n; k++) {
                        v(k, j) += s * v(k, i);
                    }
                }
            }
            for (j = l; j < n; j++) {
                v(i, j) = v(j, i) = 0;
            }
        }
        v(i, i) = 1;
        g = rv1[i];
        l = i;
    }

    // accumulate the left-hand transformation
    for (i = n - 1; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        if (i < n - 1) {
            for (j = l; j < n; j++) {
                u(i, j) = 0;
            }
        }
        if (g) {
            g = 1 / g;
            if (i != n - 1) {
                for (j = l; j < n; j++) {
                    for (s = 0, k = l; k < m; k++) {
                        s += u(k, i) * u(k, j);
                    }
                    f = (s / u(i, i)) * g;
                    for (k = i; k < m; k++) {
                        u(k, j) += f * u(k, i);
                    }
                }
            }
            for (j = i; j < m; j++) {
                u(j, i) = u(j, i) * g;
            }
        } else {
            for (j = i; j < m; j++) {
                u(j, i) = 0;
            }
        }
        ++u(i, i);
    }

    // diagonalize the bidiagonal form
    for (k = n - 1; k >= 0; k--) {
        // loop over singular values
        for (its = 0; its < 30; its++) {
            // loop over allowed iterations
            flag = 1;
            for (l = k; l >= 0; l--) {
                // test for splitting
                nm = l - 1;
                if (std::fabs(rv1[l]) + anorm == anorm) {
                    flag = 0;
                    break;
                }
                if (std::fabs((T)w[nm]) + anorm == anorm) {
                    break;
                }
            }
            if (flag) {
                c = 0;
                s = 1;
                for (i = l; i <= k; i++) {
                    f = s * rv1[i];
                    if (std::fabs(f) + anorm != anorm) {
                        g = w[i];
                        h = internal::pythag(f, g);
                        w[i] = (T)h;
                        h = 1 / h;
                        c = g * h;
                        s = -f * h;
                        for (j = 0; j < m; j++) {
                            y = u(j, nm);
                            z = u(j, i);
                            u(j, nm) = y * c + z * s;
                            u(j, i) = z * c - y * s;
                        }
                    }
                }
            }
            z = w[k];
            if (l == k) {
                // convergence
                if (z < 0) {
                    // make singular value nonnegative
                    w[k] = -z;
                    for (j = 0; j < n; j++) {
                        v(j, k) = -v(j, k);
                    }
                }
                break;
            }
            if (its >= 30) {
                throw("No convergence after 30 iterations");
            }

            // shift from bottom 2 x 2 minor
            x = w[l];
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);
            g = internal::pythag(f, (T)1);
            f = ((x - z) * (x + z) +
                 h * ((y / (f + internal::sign(g, f))) - h)) /
                x;

            // next QR transformation
            c = s = 1;
            for (j = l; j <= nm; j++) {
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = internal::pythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++) {
                    x = v(jj, j);
                    z = v(jj, i);
                    v(jj, j) = x * c + z * s;
                    v(jj, i) = z * c - x * s;
                }
                z = internal::pythag(f, h);
                w[j] = z;
                if (z) {
                    z = 1 / z;
                    c = f * z;
                    s = h * z;
                }
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++) {
                    y = u(jj, j);
                    z = u(jj, i);
                    u(jj, j) = y * c + z * s;
                    u(jj, i) = z * c - y * s;
                }
            }
            rv1[l] = 0;
            rv1[k] = f;
            w[k] = x;
        }
    }
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_SVD_H_
