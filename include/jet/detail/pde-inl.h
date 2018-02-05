// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef INCLUDE_JET_DETAIL_PDE_INL_H_
#define INCLUDE_JET_DETAIL_PDE_INL_H_

#include <jet/math_utils.h>

namespace jet {

template <typename T>
std::array<T, 2> upwind1(T* D0, T dx) {
    T invdx = 1/dx;
    std::array<T, 2> dfx;
    dfx[0] = invdx*(D0[1] - D0[0]);
    dfx[1] = invdx*(D0[2] - D0[1]);
    return dfx;
}

template <typename T>
T upwind1(T* D0, T dx, bool isDirectionPositive) {
    T invdx = 1/dx;
    return isDirectionPositive ?
        (invdx*(D0[1] - D0[0])) : invdx*(D0[2] - D0[1]);
}

template <typename T>
T cd2(T* D0, T dx) {
    T hinvdx = 0.5f/dx;
    return hinvdx*(D0[2] - D0[0]);
}

template <typename T>
std::array<T, 2> eno3(T* D0, T dx) {
    T invdx = 1/dx;
    T hinvdx = invdx/2;
    T tinvdx = invdx/3;
    T D1[6], D2[5], D3[2];
    T dQ1, dQ2, dQ3;
    T c, cstar;
    int Kstar;
    std::array<T, 2> dfx;

    D1[0] = invdx*(D0[1] - D0[0]);
    D1[1] = invdx*(D0[2] - D0[1]);
    D1[2] = invdx*(D0[3] - D0[2]);
    D1[3] = invdx*(D0[4] - D0[3]);
    D1[4] = invdx*(D0[5] - D0[4]);
    D1[5] = invdx*(D0[6] - D0[5]);

    D2[0] = hinvdx*(D1[1] - D1[0]);
    D2[1] = hinvdx*(D1[2] - D1[1]);
    D2[2] = hinvdx*(D1[3] - D1[2]);
    D2[3] = hinvdx*(D1[4] - D1[3]);
    D2[4] = hinvdx*(D1[5] - D1[4]);

    for (int K = 0; K < 2; ++K) {
        if (std::fabs(D2[K+1]) < std::fabs(D2[K+2])) {
            c = D2[K+1];
            Kstar = K-1;
            D3[0] = tinvdx*(D2[K+1] - D2[K]);
            D3[1] = tinvdx*(D2[K+2] - D2[K+1]);
        } else {
            c = D2[K+2];
            Kstar = K;
            D3[0] = tinvdx*(D2[K+2] - D2[K+1]);
            D3[1] = tinvdx*(D2[K+3] - D2[K+2]);
        }

        if (std::fabs(D3[0]) < std::fabs(D3[1])) {
            cstar = D3[0];
        } else {
            cstar = D3[1];
        }

        dQ1 = D1[K+2];
        dQ2 = c*(2*(1-K)-1)*dx;
        dQ3 = cstar*(3*square(1-Kstar) - 6*(1-Kstar) + 2)*dx*dx;

        dfx[K] = dQ1 + dQ2 + dQ3;
    }

    return dfx;
}

template <typename T>
T eno3(T* D0, T dx, bool isDirectionPositive) {
    T invdx = 1/dx;
    T hinvdx = invdx/2;
    T tinvdx = invdx/3;
    T D1[6], D2[5], D3[2];
    T dQ1, dQ2, dQ3;
    T c, cstar;
    int Kstar;

    D1[0] = invdx*(D0[1] - D0[0]);
    D1[1] = invdx*(D0[2] - D0[1]);
    D1[2] = invdx*(D0[3] - D0[2]);
    D1[3] = invdx*(D0[4] - D0[3]);
    D1[4] = invdx*(D0[5] - D0[4]);
    D1[5] = invdx*(D0[6] - D0[5]);

    D2[0] = hinvdx*(D1[1] - D1[0]);
    D2[1] = hinvdx*(D1[2] - D1[1]);
    D2[2] = hinvdx*(D1[3] - D1[2]);
    D2[3] = hinvdx*(D1[4] - D1[3]);
    D2[4] = hinvdx*(D1[5] - D1[4]);

    int K = isDirectionPositive ? 0 : 1;

    if (std::fabs(D2[K+1]) < std::fabs(D2[K+2])) {
        c = D2[K+1];
        Kstar = K-1;
        D3[0] = tinvdx*(D2[K+1] - D2[K]);
        D3[1] = tinvdx*(D2[K+2] - D2[K+1]);
    } else {
        c = D2[K+2];
        Kstar = K;
        D3[0] = tinvdx*(D2[K+2] - D2[K+1]);
        D3[1] = tinvdx*(D2[K+3] - D2[K+2]);
    }

    if (std::fabs(D3[0]) < std::fabs(D3[1])) {
        cstar = D3[0];
    } else {
        cstar = D3[1];
    }

    dQ1 = D1[K+2];
    dQ2 = c*(2*(1-K)-1)*dx;
    dQ3 = cstar*(3*square(1-Kstar) - 6*(1-Kstar) + 2)*dx*dx;

    return dQ1 + dQ2 + dQ3;
}

template <typename T>
std::array<T, 2> weno5(T* v, T h, T eps) {
    static const T c_1_3 = T(1.0/3.0), c_1_4 = T(0.25), c_1_6 = T(1.0/6.0);
    static const T c_5_6 = T(5.0/6.0), c_7_6 = T(7.0/6.0), c_11_6 = T(11.0/6.0);
    static const T c_13_12 = T(13.0/12.0);

    T hInv = T(1)/h;
    std::array<T, 2> dfx;
    T vdev[5];

    for (int K = 0; K < 2; ++K) {
        if (K == 0) {
            for (int m = 0; m < 5; ++m) {
                vdev[m] = (v[m+1] - v[m  ]) * hInv;
            }
        } else {
            for (int m = 0; m < 5; ++m) {
                vdev[m] = (v[6-m] - v[5-m]) * hInv;
            }
        }

        T phix1 =   vdev[0] * c_1_3  - vdev[1] * c_7_6 + vdev[2] * c_11_6;
        T phix2 = - vdev[1] * c_1_6  + vdev[2] * c_5_6 + vdev[3] * c_1_3;
        T phix3 =   vdev[2] * c_1_3  + vdev[3] * c_5_6 - vdev[4] * c_1_6;

        T s1 = c_13_12 * square(vdev[0] - 2*vdev[1] + vdev[2])
             + c_1_4 * square(vdev[0] - 4*vdev[1] + 3*vdev[2]);
        T s2 = c_13_12 * square(vdev[1] - 2*vdev[2] + vdev[3])
             + c_1_4 * square(vdev[1] - vdev[3]);
        T s3 = c_13_12 * square(vdev[2] - 2*vdev[3] + vdev[4])
             + c_1_4 * square(3*vdev[2] - 4*vdev[3] + vdev[4]);

        T alpha1 = T(0.1 / square(s1 + eps));
        T alpha2 = T(0.6 / square(s2 + eps));
        T alpha3 = T(0.3 / square(s3 + eps));

        T sum = alpha1 + alpha2 + alpha3;

        dfx[K] = (alpha1 * phix1 + alpha2 * phix2 + alpha3 * phix3) / sum;
    }

    return dfx;
}

template <typename T>
T weno5(T* v, T h, bool isDirectionPositive, T eps) {
    static const T c_1_3 = T(1.0/3.0), c_1_4 = T(0.25), c_1_6 = T(1.0/6.0);
    static const T c_5_6 = T(5.0/6.0), c_7_6 = T(7.0/6.0), c_11_6 = T(11.0/6.0);
    static const T c_13_12 = T(13.0/12.0);

    T hInv = T(1)/h;
    T vdev[5];

    if (isDirectionPositive) {
        for (int m = 0; m < 5; ++m) {
            vdev[m] = (v[m+1] - v[m  ]) * hInv;
        }
    } else {
        for (int m = 0; m < 5; ++m) {
            vdev[m] = (v[6-m] - v[5-m]) * hInv;
        }
    }

    T phix1 =   vdev[0] * c_1_3  - vdev[1] * c_7_6 + vdev[2] * c_11_6;
    T phix2 = - vdev[1] * c_1_6  + vdev[2] * c_5_6 + vdev[3] * c_1_3;
    T phix3 =   vdev[2] * c_1_3  + vdev[3] * c_5_6 - vdev[4] * c_1_6;

    T s1 = c_13_12 * square(vdev[0] - 2*vdev[1] + vdev[2])
         + c_1_4 * square(vdev[0] - 4*vdev[1] + 3*vdev[2]);
    T s2 = c_13_12 * square(vdev[1] - 2*vdev[2] + vdev[3])
         + c_1_4 * square(vdev[1] - vdev[3]);
    T s3 = c_13_12 * square(vdev[2] - 2*vdev[3] + vdev[4])
         + c_1_4 * square(3*vdev[2] - 4*vdev[3] + vdev[4]);

    T alpha1 = T(0.1 / square(s1 + eps));
    T alpha2 = T(0.6 / square(s2 + eps));
    T alpha3 = T(0.3 / square(s3 + eps));

    T sum = alpha1 + alpha2 + alpha3;

    return (alpha1 * phix1 + alpha2 * phix2 + alpha3 * phix3) / sum;
}

}  // namespace jet

#endif  // INCLUDE_JET_DETAIL_PDE_INL_H_
