// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <jet/animation.h>
#include <jet/array1.h>

using namespace jet;

class SineAnimation : public Animation {
 public:
    double x = 0.0;

 protected:
    void onUpdate(const Frame& frame) override {
        x = std::sin(10.0 * frame.timeInSeconds());
    }
};

class SineWithDecayAnimation final : public Animation {
 public:
    double x = 0.0;

 protected:
    void onUpdate(const Frame& frame) override {
        double decay = exp(-frame.timeInSeconds());
        x = std::sin(10.0 * frame.timeInSeconds()) * decay;
    }
};

JET_TESTS(Animation);

JET_BEGIN_TEST_F(Animation, OnUpdateSine) {
    Array1<double> t(240);
    Array1<double> data(240);

    SineAnimation sineAnim;

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#line2,0000,x.npy");
    saveData(t.constAccessor(), 0, filename);
    snprintf(filename, sizeof(filename), "data.#line2,0000,y.npy");
    saveData(data.constAccessor(), 0, filename);

    for (Frame frame; frame.index < 240; frame.advance()) {
        sineAnim.update(frame);

        t[frame.index] = frame.timeInSeconds();
        data[frame.index] = sineAnim.x;

        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,x.npy",
            frame.index);
        saveData(t.constAccessor(), frame.index, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index);
        saveData(data.constAccessor(), frame.index, filename);
    }

    saveData(t.constAccessor(), "data.#line2,x.npy");
    saveData(data.constAccessor(), "data.#line2,y.npy");
}
JET_END_TEST_F

JET_BEGIN_TEST_F(Animation, OnUpdateSineWithDecay) {
    Array1<double> t(240);
    Array1<double> data(240);

    SineWithDecayAnimation sineWithDecayAnim;

    char filename[256];
    snprintf(filename, sizeof(filename), "data.#line2,0000,x.npy");
    saveData(t.constAccessor(), 0, filename);
    snprintf(filename, sizeof(filename), "data.#line2,0000,y.npy");
    saveData(data.constAccessor(), 0, filename);

    for (Frame frame; frame.index < 240; frame.advance()) {
        sineWithDecayAnim.update(frame);

        t[frame.index] = frame.timeInSeconds();
        data[frame.index] = sineWithDecayAnim.x;

        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,x.npy",
            frame.index);
        saveData(t.constAccessor(), frame.index, filename);
        snprintf(
            filename,
            sizeof(filename),
            "data.#line2,%04d,y.npy",
            frame.index);
        saveData(data.constAccessor(), frame.index, filename);
    }

    saveData(t.constAccessor(), "data.#line2,x.npy");
    saveData(data.constAccessor(), "data.#line2,y.npy");
}
JET_END_TEST_F
