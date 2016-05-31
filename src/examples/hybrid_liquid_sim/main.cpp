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

#define APP_NAME "hybrid_liquid_sim"

using namespace jet;

void saveParticlePos(
    const ParticleSystemData3Ptr& particles,
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
        "-r resolution -l length -f frames -e example_num\n"
        "   -r, --resx: grid resolution in x-axis (default is 50)\n"
        "   -f, --frames: total number of frames (default is 100)\n"
        "   -l, --log: log filename (default is " APP_NAME ".log)\n"
        "   -o, --output: output directory name "
        "(default is " APP_NAME "_output)\n"
        "   -e, --example: example number (between 1 and 3, default is 1)\n");
}

void printInfo(
    const Size3& resolution,
    const BoundingBox3D& domain,
    const Vector3D& gridSpacing,
    size_t numberOfParticles) {
    printf(
        "Resolution: %zu x %zu x %zu\n",
        resolution.x, resolution.y, resolution.z);
    printf(
        "Domain: [%f, %f, %f] x [%f, %f, %f]\n",
        domain.lowerCorner.x, domain.lowerCorner.y, domain.lowerCorner.z,
        domain.upperCorner.x, domain.upperCorner.y, domain.upperCorner.z);
    printf(
        "Grid spacing: [%f, %f, %f]\n",
        gridSpacing.x, gridSpacing.y, gridSpacing.z);
    printf(
        "Number of particles: %zu\n",
        numberOfParticles);
}

void runSimulation(
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    const std::string& rootDir,
    PicSolver3* solver,
    size_t numberOfFrames) {
    auto particles = solver->particleSystemData();

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

// Water-drop example (FLIP)
void runExample1(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames) {
    Size3 resolution(resolutionX, 2 * resolutionX, resolutionX);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    FlipSolver3 solver;

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet3>();
    surfaceSet->addSurface(
        std::make_shared<Plane3>(
            Vector3D(0, 1, 0), Vector3D(0, 0.25 * domain.height(), 0)));
    surfaceSet->addSurface(
        std::make_shared<Sphere3>(
            domain.midPoint(), 0.15 * domain.width()));

    // Initialize particles
    auto particles = solver.particleSystemData();
    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        surfaceSet,
        domain,
        0.5 * dx,
        Vector3D());
    emitter->setPointGenerator(std::make_shared<GridPointGenerator3>());
    emitter->emit(Frame(), particles);

    // Print simulation info
    printf("Running example 1 (water-drop with FLIP)\n");
    printInfo(resolution, domain, gridSpacing, particles->numberOfParticles());

    // Run simulation
    runSimulation(
        resolution, gridSpacing, origin, rootDir, &solver, numberOfFrames);
}

// Water-drop example (PIC)
void runExample2(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames) {
    Size3 resolution(resolutionX, 2 * resolutionX, resolutionX);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    PicSolver3 solver;

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3Ptr surfaceSet = std::make_shared<ImplicitSurfaceSet3>();
    surfaceSet->addSurface(
        std::make_shared<Plane3>(
            Vector3D(0, 1, 0), Vector3D(0, 0.25 * domain.height(), 0)));
    surfaceSet->addSurface(
        std::make_shared<Sphere3>(
            domain.midPoint(), 0.15 * domain.width()));

    // Initialize particles
    auto particles = solver.particleSystemData();
    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        surfaceSet,
        domain,
        0.5 * dx,
        Vector3D());
    emitter->setPointGenerator(std::make_shared<GridPointGenerator3>());
    emitter->emit(Frame(), particles);

    // Print simulation info
    printf("Running example 2 (water-drop with PIC)\n");
    printInfo(resolution, domain, gridSpacing, particles->numberOfParticles());

    // Run simulation
    runSimulation(
        resolution, gridSpacing, origin, rootDir, &solver, numberOfFrames);
}

// Dam-breaking example
void runExample3(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames) {
    Size3 resolution(3 * resolutionX, 2 * resolutionX, (3 * resolutionX) / 2);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    FlipSolver3 solver;

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();
    double lz = domain.depth();

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

    // Initialize particles
    auto particles = solver.particleSystemData();
    auto emitter = std::make_shared<VolumeParticleEmitter3>(
        surfaceSet,
        domain,
        0.5 * dx,
        Vector3D());
    emitter->setPointGenerator(std::make_shared<GridPointGenerator3>());
    emitter->emit(Frame(), particles);

    // Collider setting
    auto columns = std::make_shared<ImplicitSurfaceSet3>();
    columns->addSurface(
        std::make_shared<Cylinder3>(Vector3D(1, 0, 0.25 * lz), 0.1, 0.75));
    columns->addSurface(
        std::make_shared<Cylinder3>(Vector3D(1.5, 0, 0.5 * lz), 0.1, 0.75));
    columns->addSurface(
        std::make_shared<Cylinder3>(Vector3D(2, 0, 0.75 * lz), 0.1, 0.75));
    auto collider = std::make_shared<RigidBodyCollider3>(columns);
    solver.setCollider(collider);

    // Print simulation info
    printf("Running example 3 (dam-breaking)\n");
    printInfo(resolution, domain, gridSpacing, particles->numberOfParticles());

    // Run simulation
    runSimulation(
        resolution, gridSpacing, origin, rootDir, &solver, numberOfFrames);
}

int main(int argc, char* argv[]) {
    size_t resolutionX = 50;
    unsigned int numberOfFrames = 100;
    int exampleNum = 1;
    std::string logFilename = APP_NAME ".log";
    std::string outputDir = APP_NAME "_output";

    // Parse options
    static struct option longOptions[] = {
        {"resx",      optional_argument, 0, 'r'},
        {"frames",    optional_argument, 0, 'f'},
        {"example",   optional_argument, 0, 'e'},
        {"log",       optional_argument, 0, 'l'},
        {"outputDir", optional_argument, 0, 'o'},
        {0,           0,                 0,  0 }
    };

    int opt = 0;
    int long_index = 0;
    while ((opt = getopt_long(
        argc, argv, "r:f:e:l:o:", longOptions, &long_index)) != -1) {
        switch (opt) {
            case 'r':
                resolutionX = static_cast<size_t>(atoi(optarg));
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
            runExample1(outputDir, resolutionX, numberOfFrames);
            break;
        case 2:
            runExample2(outputDir, resolutionX, numberOfFrames);
            break;
        case 3:
            runExample3(outputDir, resolutionX, numberOfFrames);
            break;
        default:
            printUsage();
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
