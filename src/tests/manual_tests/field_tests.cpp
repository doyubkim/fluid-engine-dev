// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/array_utils.h>
#include <jet/array2.h>
#include <jet/scalar_field2.h>
#include <jet/scalar_field3.h>
#include <jet/vector_field2.h>
#include <jet/vector_field3.h>

using namespace jet;

class MyCustomScalarField3 final : public ScalarField3 {
 public:
    double sample(const Vector3D& x) const override {
        return std::sin(x.x) * std::sin(x.y) * std::sin(x.z);
    }

    Vector3D gradient(const Vector3D& x) const override {
        return Vector3D(
            std::cos(x.x) * std::sin(x.y) * std::sin(x.z),
            std::sin(x.x) * std::cos(x.y) * std::sin(x.z),
            std::sin(x.x) * std::sin(x.y) * std::cos(x.z));
    }

    double laplacian(const Vector3D& x) const override {
        return
            -std::sin(x.x) * std::sin(x.y) * std::sin(x.z)
            -std::sin(x.x) * std::sin(x.y) * std::sin(x.z)
            -std::sin(x.x) * std::sin(x.y) * std::sin(x.z);
    }
};

JET_TESTS(ScalarField3);

JET_BEGIN_TEST_F(ScalarField3, Sample) {
    MyCustomScalarField3 field;
    Array2<double> data(50, 50);

    for (int j = 0; j < 50; ++j) {
        for (int i = 0; i < 50; ++i) {
            Vector3D x(0.04 * kPiD * i, 0.04 * kPiD * j, kHalfPiD);
            data(i, j) = field.sample(x);
        }
    }

    saveData(data.constAccessor(), "data_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(ScalarField3, Gradient) {
    MyCustomScalarField3 field;
    Array2<double> dataU(20, 20);
    Array2<double> dataV(20, 20);

    for (int j = 0; j < 20; ++j) {
        for (int i = 0; i < 20; ++i) {
            Vector3D x(0.1 * kPiD * i, 0.1 * kPiD * j, kHalfPiD);
            Vector3D g = field.gradient(x);
            dataU(i, j) = g.x;
            dataV(i, j) = g.y;
        }
    }

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(ScalarField3, Laplacian) {
    MyCustomScalarField3 field;
    Array2<double> data(50, 50);

    for (int j = 0; j < 50; ++j) {
        for (int i = 0; i < 50; ++i) {
            Vector3D x(0.04 * kPiD * i, 0.04 * kPiD * j, kHalfPiD);
            data(i, j) = field.laplacian(x);
        }
    }

    saveData(data.constAccessor(), "data_#grid2.npy");
}
JET_END_TEST_F

class MyCustomVectorField3 final : public VectorField3 {
 public:
    Vector3D sample(const Vector3D& x) const override {
        return Vector3D(
            std::sin(x.x) * std::sin(x.y),
            std::sin(x.y) * std::sin(x.z),
            std::sin(x.z) * std::sin(x.x));
    }

    double divergence(const Vector3D& x) const override {
        return std::cos(x.x) * std::sin(x.y)
             + std::cos(x.y) * std::sin(x.z)
             + std::cos(x.z) * std::sin(x.x);
    }

    Vector3D curl(const Vector3D& x) const override {
        return Vector3D(
            -std::sin(x.y) * std::cos(x.z),
            -std::sin(x.z) * std::cos(x.x),
            -std::sin(x.x) * std::cos(x.y));
    }
};

class MyCustomVectorField3_2 final : public VectorField3 {
 public:
    Vector3D sample(const Vector3D& x) const override {
        return Vector3D(-x.y, x.x, 0.0);
    }
};

JET_TESTS(VectorField3);

JET_BEGIN_TEST_F(VectorField3, Sample) {
    MyCustomVectorField3 field;
    Array2<double> dataU(20, 20);
    Array2<double> dataV(20, 20);

    for (int j = 0; j < 20; ++j) {
        for (int i = 0; i < 20; ++i) {
            Vector3D x(0.1 * kPiD * i, 0.1 * kPiD * j, kHalfPiD);
            dataU(i, j) = field.sample(x).x;
            dataV(i, j) = field.sample(x).y;
        }
    }

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(VectorField3, Divergence) {
    MyCustomVectorField3 field;
    Array2<double> data(50, 50);

    for (int j = 0; j < 50; ++j) {
        for (int i = 0; i < 50; ++i) {
            Vector3D x(0.04 * kPiD * i, 0.04 * kPiD * j, kHalfPiD);
            data(i, j) = field.divergence(x);
        }
    }

    saveData(data.constAccessor(), "data_#grid2.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(VectorField3, Curl) {
    MyCustomVectorField3 field;
    Array2<double> dataU(20, 20);
    Array2<double> dataV(20, 20);

    for (int j = 0; j < 20; ++j) {
        for (int i = 0; i < 20; ++i) {
            Vector3D x(0.1 * kPiD * i, 0.1 * kPiD * j, 0.5 * kHalfPiD);
            dataU(i, j) = field.curl(x).x;
            dataV(i, j) = field.curl(x).y;
        }
    }

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(VectorField3, Sample2) {
    MyCustomVectorField3_2 field;
    Array2<double> dataU(20, 20);
    Array2<double> dataV(20, 20);

    for (int j = 0; j < 20; ++j) {
        for (int i = 0; i < 20; ++i) {
            Vector3D x(0.05 * i - 0.5, 0.05 * j - 0.5, 0.5);
            dataU(i, j) = field.sample(x).x;
            dataV(i, j) = field.sample(x).y;
        }
    }

    saveData(dataU.constAccessor(), "data_#grid2,x.npy");
    saveData(dataV.constAccessor(), "data_#grid2,y.npy");
}
JET_END_TEST_F
