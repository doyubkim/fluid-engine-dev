// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <manual_tests.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

using namespace std;
using namespace jet;

const size_t kBufferSize = 80;

static void updateWave(const double timeInterval, double* x, double* speed) {
    (*x) += timeInterval * (*speed);

    // Boundary reflection
    if ((*x) > 1.0) {
        (*speed) *= -1.0;
        (*x) = 1.0 + timeInterval * (*speed);
    } else if ((*x) < 0.0) {
        (*speed) *= -1.0;
        (*x) = timeInterval * (*speed);
    }
}

static void accumulateWaveToHeightField(
    const double x,
    const double waveLength,
    const double maxHeight,
    Array1<double>* heightField) {
    const double quarterWaveLength = 0.25 * waveLength;
    const int start = static_cast<int>((x - quarterWaveLength) * kBufferSize);
    const int end = static_cast<int>((x + quarterWaveLength) * kBufferSize);

    for (int i = start; i < end; ++i) {
        int iNew = i;
        if (i < 0) {
            iNew = -i - 1;
        } else if (i >= static_cast<int>(kBufferSize)) {
            iNew = 2 * kBufferSize - i - 1;
        }

        double distance = fabs((i + 0.5) / kBufferSize - x);
        double height = maxHeight * 0.5
            * (cos(min(distance * M_PI / quarterWaveLength, M_PI)) + 1.0);
        (*heightField)[iNew] += height;
    }
}

JET_TESTS(HelloFluidSim);

JET_BEGIN_TEST_F(HelloFluidSim, Run) {
    const double waveLengthX = 0.8;
    const double waveLengthY = 1.2;

    const double maxHeightX = 0.5;
    const double maxHeightY = 0.4;

    double x = 0.0;
    double y = 1.0;
    double speedX = 1.5;
    double speedY = -1.0;

    const int fps = 100;
    const double timeInterval = 1.0 / fps;

    Array1<double> heightField(kBufferSize);
    Array1<double> gridPoints(kBufferSize);
    char filename[256];

    for (size_t i = 0; i < kBufferSize; ++i) {
        gridPoints[i] = 3.0 * static_cast<double>(i) / kBufferSize;
    }

    for (int i = 0; i < 500; ++i) {
        // March through time
        updateWave(timeInterval, &x, &speedX);
        updateWave(timeInterval, &y, &speedY);

        // Clear height field
        for (double& height : heightField) {
            height = 0.0;
        }

        // Accumulate waves for each center point
        (void)waveLengthX;
        (void)waveLengthY;
        (void)maxHeightX;
        (void)maxHeightY;
        accumulateWaveToHeightField(x, waveLengthX, maxHeightX, &heightField);
        accumulateWaveToHeightField(y, waveLengthY, maxHeightY, &heightField);

        snprintf(filename, sizeof(filename), "data.#line2,%04d,x.npy", i);
        saveData(gridPoints.constAccessor(), filename);
        snprintf(filename, sizeof(filename), "data.#line2,%04d,y.npy", i);
        saveData(heightField.constAccessor(), filename);
    }
}
JET_END_TEST_F
