// Copyright (c) 2016 Doyub Kim

#include <jet/jet.h>
#include <pystring/pystring.h>

#ifdef JET_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <getopt.h>

#include <algorithm>
#include <fstream>
#include <string>

#define APP_NAME "sph_sim"

using namespace jet;

void saveParticlePos(
    const SphSystemData3Ptr& particles,
    const std::string& rootDir,
    unsigned int frameCnt) {
    Array1<Vector3D> positions(particles->numberOfParticles());
    copyRange1(
        particles->positions(), particles->numberOfParticles(), &positions);
    char basename[256];
    snprintf(basename, sizeof(basename), "frame_%06d.pos", frameCnt);
    std::string filename = pystring::os::path::join(rootDir, basename);
    std::ofstream file(filename.c_str());
    if (file) {
        printf("Writing %s...\n", filename.c_str());
        positions.serialize(&file);
        file.close();
    }
}

void printUsage() {
    printf(
        "Usage: " APP_NAME " "
        "-s spacing -l length -f frames -e example_num\n"
        "   -s, --spacing: target particle spacing (default is 0.02)\n"
        "   -f, --frames: total number of frames (default is 100)\n"
        "   -l, --log: log filename (default is " APP_NAME ".log)\n"
        "   -o, --output: output directory name "
        "(default is " APP_NAME "_output)\n"
        "   -e, --example: example number (between 1 and 3, default is 1)\n");
}

void printInfo(const SphSystemData3Ptr& particles) {
    printf("Number of particles: %zu\n", particles->numberOfParticles());
}

void runSimulation(
    const std::string& rootDir,
    SphSolver3* solver,
    size_t numberOfFrames) {
    auto particles = solver->sphSystemData();

    saveParticlePos(particles, rootDir, 0);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < numberOfFrames; frame.advance()) {
        solver->update(frame);
        saveParticlePos(
            particles,
            rootDir,
            frame.index);
    }
}

// Water-drop example (PCISPH)
void runExample1(
    const std::string& rootDir,
    double targetSpacing,
    unsigned int numberOfFrames) {
    BoundingBox3D domain(Vector3D(), Vector3D(1, 2, 1));

    // Initialize solvers
    PciSphSolver3 solver;

    SphSystemData3Ptr particles = solver.sphSystemData();
    particles->setTargetDensity(1000.0);
    particles->setTargetSpacing(targetSpacing);

    // Initialize source
    ImplicitSurfaceSet3Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet3>();
    surfaceSet->addSurface(
        std::make_shared<Plane3>(
            Vector3D(0, 1, 0), Vector3D(0, 0.25 * domain.height(), 0)));
    surfaceSet->addSurface(
        std::make_shared<Sphere3>(
            domain.midPoint(), 0.15 * domain.width()));

    BoundingBox3D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        surfaceSet,
        sourceBound,
        targetSpacing,
        Vector3D());
    emitter->emit(Frame(), particles);

    // Initialize boundary
    Box3Ptr box = std::make_shared<Box3>(domain);
    box->setIsNormalFlipped(true);
    RigidBodyCollider3Ptr collider = std::make_shared<RigidBodyCollider3>(box);
    solver.setCollider(collider);

    // Print simulation info
    printf("Running example 1 (water-drop with PCISPH)\n");
    printInfo(particles);

    // Run simulation
    runSimulation(rootDir, &solver, numberOfFrames);
}

// Water-drop example (SPH)
void runExample2(
    const std::string& rootDir,
    double targetSpacing,
    unsigned int numberOfFrames) {
    BoundingBox3D domain(Vector3D(), Vector3D(2, 1, 2));

    // Initialize solvers
    SphSolver3 solver;

    SphSystemData3Ptr particles = solver.sphSystemData();
    particles->setTargetDensity(1000.0);
    particles->setTargetSpacing(targetSpacing);

    // Initialize source
    ImplicitSurfaceSet3Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet3>();
    surfaceSet->addSurface(
        std::make_shared<Plane3>(
            Vector3D(0, 1, 0), Vector3D(0, 0.25 * domain.height(), 0)));
    surfaceSet->addSurface(
        std::make_shared<Sphere3>(
            domain.midPoint(), 0.15 * domain.width()));

    BoundingBox3D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        surfaceSet,
        sourceBound,
        targetSpacing,
        Vector3D());
    emitter->emit(Frame(), particles);

    // Initialize boundary
    Box3Ptr box = std::make_shared<Box3>(domain);
    box->setIsNormalFlipped(true);
    RigidBodyCollider3Ptr collider = std::make_shared<RigidBodyCollider3>(box);
    solver.setCollider(collider);

    // Print simulation info
    printf("Running example 1 (water-drop with PCISPH)\n");
    printInfo(particles);

    // Run simulation
    runSimulation(rootDir, &solver, numberOfFrames);
}

// Dam-breaking example
void runExample3(
    const std::string& rootDir,
    double targetSpacing,
    unsigned int numberOfFrames) {
    BoundingBox3D domain(Vector3D(), Vector3D(3, 2, 1.5));
    double lz = domain.depth();

    // Initialize solvers
    PciSphSolver3 solver;
    solver.setPseudoViscosityCoefficient(1.0);

    SphSystemData3Ptr particles = solver.sphSystemData();
    particles->setTargetDensity(1000.0);
    particles->setTargetSpacing(targetSpacing);

    // Initialize source
    ImplicitSurfaceSet3Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet3>();
    surfaceSet->addSurface(
        std::make_shared<Box3>(
            Vector3D(-0.5, -0.5, -0.5 * lz),
            Vector3D(0.5, 0.75, 0.75 * lz)));
    surfaceSet->addSurface(
        std::make_shared<Box3>(
            Vector3D(2.5, -0.5, 0.25 * lz),
            Vector3D(3.5, 0.75, 1.5 * lz)));

    BoundingBox3D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        surfaceSet,
        sourceBound,
        targetSpacing,
        Vector3D());
    emitter->emit(Frame(), particles);

    // Collider setting
    auto colliderSurfaceSet = std::make_shared<ImplicitSurfaceSet3>();
    colliderSurfaceSet->addSurface(
        std::make_shared<Cylinder3>(Vector3D(1, 0, 0.25 * lz), 0.1, 0.75));
    colliderSurfaceSet->addSurface(
        std::make_shared<Cylinder3>(Vector3D(1.5, 0, 0.5 * lz), 0.1, 0.75));
    colliderSurfaceSet->addSurface(
        std::make_shared<Cylinder3>(Vector3D(2, 0, 0.75 * lz), 0.1, 0.75));

    // Initialize boundary
    auto box = std::make_shared<Box3>(domain);
    box->setIsNormalFlipped(true);
    colliderSurfaceSet->addSurface(box);

    RigidBodyCollider3Ptr collider
        = std::make_shared<RigidBodyCollider3>(colliderSurfaceSet);
    solver.setCollider(collider);

    // Print simulation info
    printf("Running example 3 (dam-breaking with PCISPH)\n");
    printInfo(particles);

    // Run simulation
    runSimulation(rootDir, &solver, numberOfFrames);
}

int main(int argc, char* argv[]) {
    double targetSpacing = 0.02;
    unsigned int numberOfFrames = 100;
    int exampleNum = 1;
    std::string logFilename = APP_NAME ".log";
    std::string outputDir = APP_NAME "_output";

    // Parse options
    static struct option longOptions[] = {
        {"spacing",   optional_argument, 0, 's'},
        {"frames",    optional_argument, 0, 'f'},
        {"example",   optional_argument, 0, 'e'},
        {"log",       optional_argument, 0, 'l'},
        {"outputDir", optional_argument, 0, 'o'},
        {0,           0,                 0,  0 }
    };

    int opt = 0;
    int long_index = 0;
    while ((opt = getopt_long(
        argc, argv, "s:f:e:l:o:", longOptions, &long_index)) != -1) {
        switch (opt) {
            case 's':
                targetSpacing = atof(optarg);
                break;
            case 'f':
                numberOfFrames = static_cast<size_t>(atoi(optarg));
                break;
            case 'e':
                exampleNum = atoi(optarg);
                break;
            case 'l':
                logFilename = optarg;
                break;
            case 'o':
                outputDir = optarg;
                break;
            default:
                printUsage();
                exit(EXIT_FAILURE);
        }
    }

#ifdef JET_WINDOWS
    _mkdir(outputDir.c_str());
#else
    mkdir(outputDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif

    std::ofstream logFile(logFilename.c_str());
    if (logFile) {
        Logging::setAllStream(&logFile);
    }

    switch (exampleNum) {
        case 1:
            runExample1(outputDir, targetSpacing, numberOfFrames);
            break;
        case 2:
            runExample2(outputDir, targetSpacing, numberOfFrames);
            break;
        case 3:
            runExample3(outputDir, targetSpacing, numberOfFrames);
            break;
        default:
            printUsage();
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
