// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array1.h>
#include <jet/sph_kernels3.h>

using namespace jet;

JET_TESTS(SphStdKernel3);

JET_BEGIN_TEST_F(SphStdKernel3, Operator) {
    Array1<double> x(101);
    Array1<double> y(101);

    double r1 = 1.0;
    SphStdKernel3 kernel1(r1);

    for (int i = 0; i <= 100; ++i) {
        double t = 2.0 * (r1 * i / 100.0) - r1;
        x[i] = t;
        y[i] = kernel1(x[i]);
    }

    saveData(x.constAccessor(), "kernel1.#line2,x.npy");
    saveData(y.constAccessor(), "kernel1.#line2,y.npy");

    double r2 = 1.2;
    SphStdKernel3 kernel2(r2);

    for (int i = 0; i <= 100; ++i) {
        double t = 2.0 * (r2 * i / 100.0) - r2;
        x[i] = t;
        y[i] = kernel2(x[i]);
    }

    saveData(x.constAccessor(), "kernel2.#line2,x.npy");
    saveData(y.constAccessor(), "kernel2.#line2,y.npy");

    double r3 = 1.5;
    SphStdKernel3 kernel3(r3);

    for (int i = 0; i <= 100; ++i) {
        double t = 2.0 * (r3 * i / 100.0) - r3;
        x[i] = t;
        y[i] = kernel3(x[i]);
    }

    saveData(x.constAccessor(), "kernel3.#line2,x.npy");
    saveData(y.constAccessor(), "kernel3.#line2,y.npy");
}
JET_END_TEST_F

JET_TESTS(SphSpikyKernel3);

JET_BEGIN_TEST_F(SphSpikyKernel3, Derivatives) {
    Array1<double> x(101);
    Array1<double> y0(101);
    Array1<double> y1(101);
    Array1<double> y2(101);

    double r = 1.0;
    SphSpikyKernel3 spiky(r);

    for (int i = 0; i <= 100; ++i) {
        double t = 2.0 * (r * i / 100.0) - r;
        x[i] = t;
        y0[i] = spiky(std::abs(x[i]));
        y1[i] = spiky.firstDerivative(std::abs(x[i]));
        y2[i] = spiky.secondDerivative(std::abs(x[i]));
    }

    saveData(x.constAccessor(), "spiky.#line2,x.npy");
    saveData(y0.constAccessor(), "spiky.#line2,y.npy");
    saveData(x.constAccessor(), "spiky_1st.#line2,x.npy");
    saveData(y1.constAccessor(), "spiky_1st.#line2,y.npy");
    saveData(x.constAccessor(), "spiky_2nd.#line2,x.npy");
    saveData(y2.constAccessor(), "spiky_2nd.#line2,y.npy");

    SphStdKernel3 stdKernel(r);

    for (int i = 0; i <= 100; ++i) {
        double t = 2.0 * (r * i / 100.0) - r;
        x[i] = t;
        y0[i] = stdKernel(std::abs(x[i]));
        y1[i] = stdKernel.firstDerivative(std::abs(x[i]));
        y2[i] = stdKernel.secondDerivative(std::abs(x[i]));
    }

    saveData(x.constAccessor(), "std.#line2,x.npy");
    saveData(y0.constAccessor(), "std.#line2,y.npy");
    saveData(x.constAccessor(), "std_1st.#line2,x.npy");
    saveData(y1.constAccessor(), "std_1st.#line2,y.npy");
    saveData(x.constAccessor(), "std_2nd.#line2,x.npy");
    saveData(y2.constAccessor(), "std_2nd.#line2,y.npy");
}
JET_END_TEST_F
