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

#define APP_NAME "smoke_sim"

using namespace jet;

const size_t kEdgeBlur = 3;
const float kEdgeBlurF = 3.f;

inline float smoothStep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

// Export density field to Mitsuba volume file.
void saveVolume(
    const ScalarGrid3Ptr& density,
    const std::string& rootDir,
    unsigned int frameCnt) {
    char basename[256];
    snprintf(basename, sizeof(basename), "frame_%06d.vol", frameCnt);
    std::string filename = pystring::os::path::join(rootDir, basename);
    std::ofstream file(filename.c_str(), std::ofstream::binary);
    if (file) {
        printf("Writing %s...\n", filename.c_str());

        // Mitsuba 0.5.0 gridvolume format
        char header[48];
        memset(header, 0, sizeof(header));

        header[0] = 'V';
        header[1] = 'O';
        header[2] = 'L';
        header[3] = 3;
        int32_t* encoding = reinterpret_cast<int32_t*>(header + 4);
        encoding[0] = 1;  // 32-bit float
        encoding[1] = static_cast<int32_t>(density->dataSize().x);
        encoding[2] = static_cast<int32_t>(density->dataSize().y);
        encoding[3] = static_cast<int32_t>(density->dataSize().z);
        encoding[4] = 1;  // number of channels
        BoundingBox3D domain = density->boundingBox();
        float* bbox = reinterpret_cast<float*>(encoding + 5);
        bbox[0] = static_cast<float>(domain.lowerCorner.x);
        bbox[1] = static_cast<float>(domain.lowerCorner.y);
        bbox[2] = static_cast<float>(domain.lowerCorner.z);
        bbox[3] = static_cast<float>(domain.upperCorner.x);
        bbox[4] = static_cast<float>(domain.upperCorner.y);
        bbox[5] = static_cast<float>(domain.upperCorner.z);

        file.write(header, sizeof(header));

        Array3<float> data(density->dataSize());
        data.parallelForEachIndex([&](size_t i, size_t j, size_t k) {
            float d = static_cast<float>((*density)(i, j, k));

            // Blur the edge for less-noisy rendering
            if (i < kEdgeBlur) {
                d *= smoothStep(0.f, kEdgeBlurF, static_cast<float>(i));
            }
            if (i > data.size().x - 1 - kEdgeBlur) {
                d *= smoothStep(
                    0.f,
                    kEdgeBlurF,
                    static_cast<float>((data.size().x - 1) - i));
            }
            if (j < kEdgeBlur) {
                d *= smoothStep(0.f, kEdgeBlurF, static_cast<float>(j));
            }
            if (j > data.size().y - 1 - kEdgeBlur) {
                d *= smoothStep(
                    0.f,
                    kEdgeBlurF,
                    static_cast<float>((data.size().y - 1) - j));
            }
            if (k < kEdgeBlur) {
                d *= smoothStep(0.f, kEdgeBlurF, static_cast<float>(k));
            }
            if (k > data.size().z - 1 - kEdgeBlur) {
                d *= smoothStep(
                    0.f,
                    kEdgeBlurF,
                    static_cast<float>((data.size().z - 1) - k));
            }

            data(i, j, k) = d;
        });
        file.write(
            reinterpret_cast<const char*>(data.data()),
            sizeof(float) * data.size().x * data.size().y * data.size().z);

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
        "   -e, --example: example number (between 1 and 5, default is 1)\n");
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
    std::function<double(const Vector3D&)> sourceFunc,
    std::function<double(double, const Vector3D&)> uFilterFunc,
    GridSmokeSolver3* solver,
    size_t numberOfFrames) {
    auto density = solver->smokeDensity();
    auto densityPos = density->dataPosition();
    auto temperature = solver->temperature();
    auto temperaturePos = temperature->dataPosition();
    auto velocity = solver->velocity();
    auto uPos = velocity->uPosition();

    saveVolume(solver->smokeDensity(), rootDir, 0);

    Frame frame(1, 1.0 / 60.0);
    for ( ; frame.index < numberOfFrames; frame.advance()) {
        density->parallelForEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                double current = (*density)(i, j, k);
                (*density)(i, j, k)
                    = std::max(current, sourceFunc(densityPos(i, j, k)));
            });
        temperature->parallelForEachDataPointIndex(
            [&] (size_t i, size_t j, size_t k) {
                double current = (*temperature)(i, j, k);
                (*temperature)(i, j, k)
                    = std::max(current, sourceFunc(temperaturePos(i, j, k)));
            });
        velocity->parallelForEachUIndex(
            [&] (size_t i, size_t j, size_t k) {
                velocity->u(i, j, k)
                    = uFilterFunc(velocity->u(i, j, k), uPos(i, j, k));
            });

        solver->update(frame);
        saveVolume(solver->smokeDensity(), rootDir, frame.index);
    }
}

void runSimulation(
    const std::string& rootDir,
    const std::function<double(const Vector3D&)>& sourceFunc,
    GridSmokeSolver3* solver,
    size_t numberOfFrames) {
    runSimulation(
        rootDir,
        sourceFunc,
        [] (double u, const Vector3D&) { return u; },
        solver,
        numberOfFrames);
}

void runExample1(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames) {
    Size3 resolution(resolutionX, 2 * resolutionX, resolutionX);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    GridSmokeSolver3 solver;
    solver.setAdvectionSolver(std::make_shared<CubicSemiLagrangian3>());

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Box3>(
            Vector3D(0.45, -1, 0.45), Vector3D(0.55, 0.05, 0.55)));
    auto sourceFunc = [&] (const Vector3D& pt) {
        // Convert SDF to density-like field
        return 1.0 - smearedHeavisideSdf(surfaceSet.signedDistance(pt) / dx);
    };

    solver.smokeDensity()->fill(sourceFunc);
    solver.temperature()->fill(sourceFunc);

    // Collider setting
    auto sphere = std::make_shared<Sphere3>(
        Vector3D(0.5, 0.3, 0.5), 0.075 * domain.width());
    auto collider = std::make_shared<RigidBodyCollider3>(sphere);
    solver.setCollider(collider);

    // Print simulation info
    printf("Running example 1 (rising smoke with cubic-spline advection)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(rootDir, sourceFunc, &solver, numberOfFrames);
}

void runExample2(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames) {
    Size3 resolution(resolutionX, 2 * resolutionX, resolutionX);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    GridSmokeSolver3 solver;
    solver.setAdvectionSolver(std::make_shared<SemiLagrangian3>());

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Box3>(
            Vector3D(0.45, -1, 0.45), Vector3D(0.55, 0.05, 0.55)));
    auto sourceFunc = [&] (const Vector3D& pt) {
        // Convert SDF to density-like field
        return 1.0 - smearedHeavisideSdf(surfaceSet.signedDistance(pt) / dx);
    };

    solver.smokeDensity()->fill(sourceFunc);
    solver.temperature()->fill(sourceFunc);

    // Collider setting
    auto sphere = std::make_shared<Sphere3>(
        Vector3D(0.5, 0.3, 0.5), 0.075 * domain.width());
    auto collider = std::make_shared<RigidBodyCollider3>(sphere);
    solver.setCollider(collider);

    // Print simulation info
    printf("Running example 2 (rising smoke with linear advection)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(rootDir, sourceFunc, &solver, numberOfFrames);
}

void runExample3(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames) {
    Size3 resolution(resolutionX, resolutionX / 4 * 5, resolutionX / 2);
    Vector3D origin(-1, -0.15, -0.5);
    double dx = 2.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    GridSmokeSolver3 solver;
    solver.setAdvectionSolver(std::make_shared<CubicSemiLagrangian3>());

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    VertexCenteredScalarGrid3 bunnySdf;
    std::ifstream sdfFile("dragon.sdf", std::ifstream::binary);
    if (sdfFile) {
        bunnySdf.deserialize(&sdfFile);
        sdfFile.close();
    } else {
        fprintf(stderr, "Cannot open dragon.sdf\n");
        fprintf(
            stderr,
            "Run\nbin/obj2sdf -i resources/dragon.obj"
            " -o dragon.sdf\nto generate the sdf file.\n");
        exit(EXIT_FAILURE);
    }
    auto sourceFunc = [&] (const Vector3D& pt) {
        // Convert SDF to density-like field
        return 1.0 - smearedHeavisideSdf(bunnySdf.sample(pt) / dx);
    };

    solver.smokeDensity()->fill(sourceFunc);
    solver.temperature()->fill(sourceFunc);

    // Print simulation info
    printf("Running example 3 (rising dragon)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(rootDir, sourceFunc, &solver, numberOfFrames);
}

void runExample4(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames) {
    Size3 resolution(resolutionX, 6 * resolutionX / 5, resolutionX / 2);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    GridSmokeSolver3 solver;
    solver.setBuoyancyTemperatureFactor(2.0);

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Box3>(
            Vector3D(0.05, 0.1, 0.225), Vector3D(0.1, 0.15, 0.275)));
    auto sourceFunc = [&] (const Vector3D& pt) {
        // Convert SDF to density-like field
        return 1.0 - smearedHeavisideSdf(surfaceSet.signedDistance(pt) / dx);
    };

    solver.smokeDensity()->fill(sourceFunc);
    solver.temperature()->fill(sourceFunc);

    // Print simulation info
    printf("Running example 4 (rising smoke with cubic-spline advection)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(
        rootDir,
        sourceFunc,
        [&] (double u, const Vector3D& uPos) {
            double sdf = surfaceSet.signedDistance(uPos);
            if (sdf < 0.05) {
                return 0.5;
            } else {
                return u;
            }
        },
        &solver,
        numberOfFrames);
}

void runExample5(
    const std::string& rootDir,
    size_t resolutionX,
    unsigned int numberOfFrames) {
    Size3 resolution(resolutionX, 6 * resolutionX / 5, resolutionX / 2);
    Vector3D origin;
    double dx = 1.0 / resolutionX;
    Vector3D gridSpacing(dx, dx, dx);

    // Initialize solvers
    GridSmokeSolver3 solver;
    solver.setBuoyancyTemperatureFactor(2.0);
    solver.setAdvectionSolver(std::make_shared<SemiLagrangian3>());

    // Initialize grids
    auto grids = solver.gridSystemData();
    grids->resize(resolution, gridSpacing, origin);
    BoundingBox3D domain = grids->boundingBox();

    // Initialize source
    ImplicitSurfaceSet3 surfaceSet;
    surfaceSet.addExplicitSurface(
        std::make_shared<Box3>(
            Vector3D(0.05, 0.1, 0.225), Vector3D(0.1, 0.15, 0.275)));
    auto sourceFunc = [&] (const Vector3D& pt) {
        // Convert SDF to density-like field
        return 1.0 - smearedHeavisideSdf(surfaceSet.signedDistance(pt) / dx);
    };

    solver.smokeDensity()->fill(sourceFunc);
    solver.temperature()->fill(sourceFunc);

    // Print simulation info
    printf("Running example 5 (rising smoke with linear advection)\n");
    printInfo(resolution, domain, gridSpacing);

    // Run simulation
    runSimulation(
        rootDir,
        sourceFunc,
        [&] (double u, const Vector3D& uPos) {
            double sdf = surfaceSet.signedDistance(uPos);
            if (sdf < 0.05) {
                return 0.5;
            } else {
                return u;
            }
        },
        &solver,
        numberOfFrames);
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
        case 4:
            runExample4(outputDir, resolutionX, numberOfFrames);
            break;
        case 5:
            runExample5(outputDir, resolutionX, numberOfFrames);
            break;
        default:
            printUsage();
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
