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
#include <vector>

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

void printInfo(const LevelSetLiquidSolver3Ptr& solver) {
    auto grids = solver->gridSystemData();
    Size3 resolution = grids->resolution();
    BoundingBox3D domain = grids->boundingBox();
    Vector3D gridSpacing = grids->gridSpacing();

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
    const LevelSetLiquidSolver3Ptr& solver,
    size_t numberOfFrames,
    double fps) {
    auto sdf = solver->signedDistanceField();

    for (Frame frame(0, 1.0 / fps); frame.index < numberOfFrames; ++frame) {
        solver->update(frame);

        triangulateAndSave(sdf, rootDir, frame.index);
    }
}

// Water-drop example
void runExample1(
    const std::string& rootDir,
    size_t resX,
    unsigned int numberOfFrames,
    double fps) {
    // Build solver
    auto solver = LevelSetLiquidSolver3::builder()
        .withResolution({resX, 2 * resX, resX})
        .withDomainSizeX(1.0)
        .makeShared();

    auto grids = solver->gridSystemData();
    BoundingBox3D domain = grids->boundingBox();

    // Build emitter
    auto plane = Plane3::builder()
        .withNormal({0, 1, 0})
        .withPoint({0, 0.25 * domain.height(), 0})
        .makeShared();

    auto sphere = Sphere3::builder()
        .withCenter(domain.midPoint())
        .withRadius(0.15 * domain.width())
        .makeShared();

    auto surfaceSet = ImplicitSurfaceSet3::builder()
        .withExplicitSurfaces({plane, sphere})
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(surfaceSet)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addSignedDistanceTarget(solver->signedDistanceField());

    // Print simulation info
    printf("Running example 1 (water-drop)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, fps);
}

// Dam-breaking example
void runExample2(
    const std::string& rootDir,
    size_t resX,
    unsigned int numberOfFrames,
    double fps) {
    // Build solver
    auto solver = LevelSetLiquidSolver3::builder()
        .withResolution({3 * resX, 2 * resX, (3 * resX) / 2})
        .withDomainSizeX(3.0)
        .makeShared();

    auto grids = solver->gridSystemData();
    BoundingBox3D domain = grids->boundingBox();
    double lz = domain.depth();

    // Build emitter
    auto box1 = Box3::builder()
        .withLowerCorner({-0.5, -0.5, -0.5 * lz})
        .withUpperCorner({0.5, 0.75, 0.75 * lz})
        .makeShared();

    auto box2 = Box3::builder()
        .withLowerCorner({2.5, -0.5, 0.25 * lz})
        .withUpperCorner({3.5, 0.75, 1.5 * lz})
        .makeShared();

    auto boxSet = ImplicitSurfaceSet3::builder()
        .withExplicitSurfaces({box1, box2})
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(boxSet)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addSignedDistanceTarget(solver->signedDistanceField());

    // Build collider
    auto cyl1 = Cylinder3::builder()
        .withCenter({1, 0.375, 0.375})
        .withRadius(0.1)
        .withHeight(0.75)
        .makeShared();

    auto cyl2 = Cylinder3::builder()
        .withCenter({1.5, 0.375, 0.75})
        .withRadius(0.1)
        .withHeight(0.75)
        .makeShared();

    auto cyl3 = Cylinder3::builder()
        .withCenter({2, 0.375, 1.125})
        .withRadius(0.1)
        .withHeight(0.75)
        .makeShared();

    auto cylSet = ImplicitSurfaceSet3::builder()
        .withExplicitSurfaces({cyl1, cyl2, cyl3})
        .makeShared();

    auto collider = RigidBodyCollider3::builder()
        .withSurface(cylSet)
        .makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 2 (dam-breaking)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, fps);
}

// High-viscosity example (bunny-drop)
void runExample3(
    const std::string& rootDir,
    size_t resX,
    unsigned int numberOfFrames,
    double fps) {
    // Build solver
    auto solver = LevelSetLiquidSolver3::builder()
        .withResolution({resX, resX, resX})
        .withDomainSizeX(1.0)
        .makeShared();

    solver->setViscosityCoefficient(1.0);
    solver->setIsGlobalCompensationEnabled(true);

    auto grids = solver->gridSystemData();

    // Build emitters
    VertexCenteredScalarGrid3 bunnySdf;
    std::ifstream sdfFile("bunny.sdf", std::ifstream::binary);
    if (sdfFile) {
        std::vector<uint8_t> buffer(
            (std::istreambuf_iterator<char>(sdfFile)),
            (std::istreambuf_iterator<char>()));
        bunnySdf.deserialize(buffer);
        sdfFile.close();
    } else {
        fprintf(stderr, "Cannot open bunny.sdf\n");
        fprintf(
            stderr,
            "Run\nbin/obj2sdf -i resources/bunny.obj"
            " -o bunny.sdf\nto generate the sdf file.\n");
        exit(EXIT_FAILURE);
    }

    auto bunny = CustomImplicitSurface3::builder()
        .withSignedDistanceFunction(bunnySdf.sampler())
        .withResolution(grids->gridSpacing().x)
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(bunny)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addSignedDistanceTarget(solver->signedDistanceField());

    // Print simulation info
    printf("Running example 3 (high-viscosity)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, fps);
}

// Low-viscosity example (bunny-drop)
void runExample4(
    const std::string& rootDir,
    size_t resX,
    unsigned int numberOfFrames,
    double fps) {
    // Build solver
    auto solver = LevelSetLiquidSolver3::builder()
        .withResolution({resX, resX, resX})
        .withDomainSizeX(1.0)
        .makeShared();

    solver->setViscosityCoefficient(0.0);
    solver->setIsGlobalCompensationEnabled(true);

    auto grids = solver->gridSystemData();

    // Build emitters
    VertexCenteredScalarGrid3 bunnySdf;
    std::ifstream sdfFile("bunny.sdf", std::ifstream::binary);
    if (sdfFile) {
        std::vector<uint8_t> buffer(
            (std::istreambuf_iterator<char>(sdfFile)),
            (std::istreambuf_iterator<char>()));
        bunnySdf.deserialize(buffer);
        sdfFile.close();
    } else {
        fprintf(stderr, "Cannot open bunny.sdf\n");
        fprintf(
            stderr,
            "Run\nbin/obj2sdf -i resources/bunny.obj"
            " -o bunny.sdf\nto generate the sdf file.\n");
        exit(EXIT_FAILURE);
    }

    auto bunny = CustomImplicitSurface3::builder()
        .withSignedDistanceFunction(bunnySdf.sampler())
        .withResolution(grids->gridSpacing().x)
        .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
        .withSourceRegion(bunny)
        .makeShared();

    solver->setEmitter(emitter);
    emitter->addSignedDistanceTarget(solver->signedDistanceField());

    // Print simulation info
    printf("Running example 4 (low-viscosity)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, fps);
}

int main(int argc, char* argv[]) {
    size_t resX = 50;
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
                resX = static_cast<size_t>(atoi(optarg));
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
            runExample1(outputDir, resX, numberOfFrames, fps);
            break;
        case 2:
            runExample2(outputDir, resX, numberOfFrames, fps);
            break;
        case 3:
            runExample3(outputDir, resX, numberOfFrames, fps);
            break;
        case 4:
            runExample4(outputDir, resX, numberOfFrames, fps);
            break;
        default:
            printUsage();
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
