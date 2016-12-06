// Copyright (c) 2016 Doyub Kim

#include <jet/jet.h>
#include <pystring/pystring.h>

#ifdef JET_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <getopt.h>

#include <fstream>
#include <string>

#define APP_NAME "level_set_liquid_sim"

using namespace jet;

void saveTriangleMesh(
    const TriangleMesh3& mesh,
    const std::string& rootDir,
    unsigned int frameCnt) {
    char basename[256];
    snprintf(basename, sizeof(basename), "frame_%06d.obj", frameCnt);
    std::string filename = pystring::os::path::join(rootDir, basename);
    std::ofstream file(filename.c_str());
    if (file) {
        printf("Writing %s...\n", filename.c_str());
        mesh.writeObj(&file);
        file.close();
    }
}

void triangulateAndSave(
    const ScalarGrid3Ptr& sdf,
    const std::string& rootDir,
    unsigned int frameCnt) {
    TriangleMesh3 mesh;
    int flag = kDirectionAll & ~kDirectionDown;
    marchingCubes(
        sdf->constDataAccessor(),
        sdf->gridSpacing(),
        sdf->dataOrigin(),
        &mesh,
        0.0,
        flag);
    saveTriangleMesh(mesh, rootDir, frameCnt);
}

void printUsage() {
    printf(
        "Usage: " APP_NAME " "
        "-r resolution -l length -f frames -e example_num\n"
        "   -r, --resx: grid resolution in x-axis (default is 50)\n"
        "   -f, --frames: total number of frames (default is 100)\n"
        "   -p, --fps: frames per second (default is 60.0)\n"
        "   -l, --log: log filename (default is " APP_NAME ".log)\n"
        "   -o, --output: output directory name "
        "(default is " APP_NAME "_output)\n"
        "   -e, --example: example number (between 1 and 4, default is 1)\n"
        "   -h, --help: print this message\n");
}

void printInfo(
    const Size3& resolution,
    const BoundingBox3D& domain,
    const Vector3D& gridSpacing) {
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
}

void runSimulation(
    const std::string& rootDir,
    LevelSetLiquidSolver3* solver,
    size_t numberOfFrames,
    double fps) {
    auto sdf = solver->signedDistanceField();
    triangulateAndSave(sdf, rootDir, 0);

    Frame frame(1, 1.0 / fps);
    for ( ; frame.index < numberOfFrames; frame.advance()) {
        solver->update(frame);
        triangulateAndSave(sdf, rootDir, frame.index);
    }
}

// Water-drop example
void runExample1(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames,
    double fps) {
    Size3 resolution(resolutionX, 2 * resolutionX, resolutionX);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    LevelSetLiquidSolver3 solver;

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Plane3>(
            Vector3D(0, 1, 0), Vector3D(0, 0.25 * domain.height(), 0)));
    surfaceSet.addExplicitSurface(
        std::make_shared<Sphere3>(
            domain.midPoint(), 0.15 * domain.width()));

    auto sdf = solver.signedDistanceField();
    sdf->fill([&] (const Vector3D& pt) {
        return surfaceSet.signedDistance(pt);
    });

    // Print simulation info
    printf("Running example 1 (water-drop)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(rootDir, &solver, numberOfFrames, fps);
}

// Dam-breaking example
void runExample2(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames,
    double fps) {
    Size3 resolution(3 * resolutionX, 2 * resolutionX, (3 * resolutionX) / 2);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    LevelSetLiquidSolver3 solver;

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();
    double lz = domain.depth();

    // Initialize source
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Box3>(
            Vector3D(-0.5, -0.5, -0.5 * lz),
            Vector3D(0.5, 0.75, 0.75 * lz)));
    surfaceSet.addExplicitSurface(
        std::make_shared<Box3>(
            Vector3D(2.5, -0.5, 0.25 * lz),
            Vector3D(3.5, 0.75, 1.5 * lz)));
    auto sdf = solver.signedDistanceField();
    sdf->fill([&] (const Vector3D& pt) {
        return surfaceSet.signedDistance(pt);
    });

    // Collider setting
    auto columns = std::make_shared<ImplicitSurfaceSet3>();
    columns->addExplicitSurface(
        std::make_shared<Cylinder3>(Vector3D(1, 0, 0.25 * lz), 0.1, 0.75));
    columns->addExplicitSurface(
        std::make_shared<Cylinder3>(Vector3D(1.5, 0, 0.5 * lz), 0.1, 0.75));
    columns->addExplicitSurface(
        std::make_shared<Cylinder3>(Vector3D(2, 0, 0.75 * lz), 0.1, 0.75));
    auto collider = std::make_shared<RigidBodyCollider3>(columns);
    solver.setCollider(collider);

    // Print simulation info
    printf("Running example 2 (dam-breaking)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(rootDir, &solver, numberOfFrames, fps);
}

// High-viscosity example (bunny-drop)
void runExample3(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames,
    double fps) {
    Size3 resolution(resolutionX, resolutionX, resolutionX);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    LevelSetLiquidSolver3 solver;
    solver.setViscosityCoefficient(1.0);
    solver.setIsGlobalCompensationEnabled(true);

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    VertexCenteredScalarGrid3 bunnySdf;
    std::ifstream sdfFile("bunny.sdf", std::ifstream::binary);
    if (sdfFile) {
        bunnySdf.deserialize(&sdfFile);
        sdfFile.close();
    } else {
        fprintf(stderr, "Cannot open bunny.sdf\n");
        fprintf(
            stderr,
            "Run\nbin/obj2sdf -i resources/bunny.obj"
            " -o bunny.sdf\nto generate the sdf file.\n");
        exit(EXIT_FAILURE);
    }

    auto sdf = solver.signedDistanceField();
    sdf->fill([&] (const Vector3D& pt) {
        return bunnySdf.sample(pt);
    });

    // Print simulation info
    printf("Running example 3 (high-viscosity)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(rootDir, &solver, numberOfFrames, fps);
}

// Low-viscosity example (bunny-drop)
void runExample4(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames,
    double fps) {
    Size3 resolution(resolutionX, resolutionX, resolutionX);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    LevelSetLiquidSolver3 solver;
    solver.setIsGlobalCompensationEnabled(true);

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    VertexCenteredScalarGrid3 bunnySdf;
    std::ifstream sdfFile("bunny.sdf", std::ifstream::binary);
    if (sdfFile) {
        bunnySdf.deserialize(&sdfFile);
        sdfFile.close();
    } else {
        fprintf(stderr, "Cannot open bunny.sdf\n");
        fprintf(
            stderr,
            "Run\nbin/obj2sdf -i resources/bunny.obj -r %zu"
            " -o bunny.sdf\nto generate the sdf file.\n",
            resolutionX);
        exit(EXIT_FAILURE);
    }

    auto sdf = solver.signedDistanceField();
    sdf->fill([&] (const Vector3D& pt) {
        return bunnySdf.sample(pt);
    });

    // Print simulation info
    printf("Running example 4 (low-viscosity)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(rootDir, &solver, numberOfFrames, fps);
}

int main(int argc, char* argv[]) {
    size_t resolutionX = 50;
    unsigned int numberOfFrames = 100;
    double fps = 60.0;
    int exampleNum = 1;
    std::string logFilename = APP_NAME ".log";
    std::string outputDir = APP_NAME "_output";

    // Parse options
    static struct option longOptions[] = {
        {"resx",      optional_argument, 0, 'r'},
        {"frames",    optional_argument, 0, 'f'},
        {"fps",       optional_argument, 0, 'p'},
        {"example",   optional_argument, 0, 'e'},
        {"log",       optional_argument, 0, 'l'},
        {"outputDir", optional_argument, 0, 'o'},
        {"help",      optional_argument, 0, 'h'},
        {0,           0,                 0,  0 }
    };

    int opt = 0;
    int long_index = 0;
    while ((opt = getopt_long(
        argc, argv, "r:f:p:e:l:o:h", longOptions, &long_index)) != -1) {
        switch (opt) {
            case 'r':
                resolutionX = static_cast<size_t>(atoi(optarg));
                break;
            case 'f':
                numberOfFrames = static_cast<size_t>(atoi(optarg));
                break;
            case 'p':
                fps = atof(optarg);
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
            case 'h':
                printUsage();
                exit(EXIT_SUCCESS);
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
            runExample1(outputDir, resolutionX, numberOfFrames, fps);
            break;
        case 2:
            runExample2(outputDir, resolutionX, numberOfFrames, fps);
            break;
        case 3:
            runExample3(outputDir, resolutionX, numberOfFrames, fps);
            break;
        case 4:
            runExample4(outputDir, resolutionX, numberOfFrames, fps);
            break;
        default:
            printUsage();
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
