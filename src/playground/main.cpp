// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include "my_physics_solver.h"

#include <fstream>

using namespace jet;

void runSimulation(MyPhysicsSolver& solver, double frameIntervalInSeconds,
                   unsigned int numberOfFrames) {
    for (Frame frame(0, frameIntervalInSeconds); frame.index < numberOfFrames;
         ++frame) {
        printf("Updating frame %u\n", frame.index);
        solver.update(frame);
    }
}

int main() {
    // Set up output log file
    std::ofstream logFile("playground.log");
    if (logFile) {
        Logging::setAllStream(&logFile);
    }

    // Set up simulation
    MyPhysicsSolver solver;
    double frameIntervalInSeconds = 0.01;
    unsigned int numberOfFrames = 300;

    // Run the sim
    runSimulation(solver, frameIntervalInSeconds, numberOfFrames);

    return 0;
}
