// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifdef JET_USE_CUDA
#include "cuda_pci_sph_solver2_example.h"
#include "cuda_wc_sph_solver2_example.h"
#endif  // JET_USE_CUDA

#include "pci_sph_solver2_example.h"
#include "wc_sph_solver2_example.h"

#include <example_app.h>

#include <jet/jet.h>

using namespace jet;

int main(int, const char**) {
    Logging::mute();

    ExampleApp::initialize("Particle Sim", 1280, 1280);
    ExampleApp::addExample<WcSphSolver2Example>();
    ExampleApp::addExample<PciSphSolver2Example>();
#ifdef JET_USE_CUDA
    ExampleApp::addExample<CudaWcSphSolver2Example>();
    ExampleApp::addExample<CudaPciSphSolver2Example>();
#endif  // JET_USE_CUDA
    ExampleApp::run();
    ExampleApp::finalize();

    return 0;
}
