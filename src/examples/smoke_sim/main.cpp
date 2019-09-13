// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/jet.h>
#include <pystring/pystring.h>

#ifdef JET_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <example_utils/clara_utils.h>
#include <clara.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define APP_NAME "smoke_sim"

using namespace jet;

const size_t kEdgeBlur = 3;
const float kEdgeBlurF = 3.f;
const double kTgaScale = 10.0;

inline float smoothStep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
}

// Export density field to Mitsuba volume file.
void saveVolumeAsVol(const ScalarGrid3Ptr& density, const std::string& rootDir,
                     int frameCnt) {
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
                d *= smoothStep(0.f, kEdgeBlurF,
                                static_cast<float>((data.size().x - 1) - i));
            }
            if (j < kEdgeBlur) {
                d *= smoothStep(0.f, kEdgeBlurF, static_cast<float>(j));
            }
            if (j > data.size().y - 1 - kEdgeBlur) {
                d *= smoothStep(0.f, kEdgeBlurF,
                                static_cast<float>((data.size().y - 1) - j));
            }
            if (k < kEdgeBlur) {
                d *= smoothStep(0.f, kEdgeBlurF, static_cast<float>(k));
            }
            if (k > data.size().z - 1 - kEdgeBlur) {
                d *= smoothStep(0.f, kEdgeBlurF,
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

void saveVolumeAsTga(const ScalarGrid3Ptr& density, const std::string& rootDir,
                     int frameCnt) {
    char basename[256];
    snprintf(basename, sizeof(basename), "frame_%06d.tga", frameCnt);
    std::string filename = pystring::os::path::join(rootDir, basename);
    std::ofstream file(filename.c_str(), std::ofstream::binary);
    if (file) {
        printf("Writing %s...\n", filename.c_str());

        Size3 dataSize = density->dataSize();

        std::array<char, 18> header;
        header.fill(0);

        int imgWidth = static_cast<int>(dataSize.x);
        int imgHeight = static_cast<int>(dataSize.y);

        header[2] = 2;
        header[12] = (char)(imgWidth & 0xff);
        header[13] = (char)((imgWidth & 0xff00) >> 8);
        header[14] = (char)(imgHeight & 0xff);
        header[15] = (char)((imgHeight & 0xff00) >> 8);
        header[16] = 24;

        file.write(header.data(), header.size());

        Array2<double> hdrImg(dataSize.x, dataSize.y);
        hdrImg.parallelForEachIndex([&](size_t i, size_t j) {
            double sum = 0.0;
            for (size_t k = 0; k < dataSize.z; ++k) {
                sum += (*density)(i, j, k);
            }
            hdrImg(i, j) = kTgaScale * sum / static_cast<double>(dataSize.z);
        });

        std::vector<char> img(3 * dataSize.x * dataSize.y);
        for (size_t i = 0; i < dataSize.x * dataSize.y; ++i) {
            char val = static_cast<char>(clamp(hdrImg[i], 0.0, 1.0) * 255.0);
            img[3 * i + 0] = val;
            img[3 * i + 1] = val;
            img[3 * i + 2] = val;
        }
        file.write(img.data(), img.size());

        file.close();
    }
}

void printInfo(const GridSmokeSolver3Ptr& solver) {
    auto grids = solver->gridSystemData();
    Size3 resolution = grids->resolution();
    BoundingBox3D domain = grids->boundingBox();
    Vector3D gridSpacing = grids->gridSpacing();

    printf("Resolution: %zu x %zu x %zu\n", resolution.x, resolution.y,
           resolution.z);
    printf("Domain: [%f, %f, %f] x [%f, %f, %f]\n", domain.lowerCorner.x,
           domain.lowerCorner.y, domain.lowerCorner.z, domain.upperCorner.x,
           domain.upperCorner.y, domain.upperCorner.z);
    printf("Grid spacing: [%f, %f, %f]\n", gridSpacing.x, gridSpacing.y,
           gridSpacing.z);
}

void runSimulation(const std::string& rootDir,
                   const GridSmokeSolver3Ptr& solver, int numberOfFrames,
                   const std::string& format, double fps) {
    auto density = solver->smokeDensity();

    for (Frame frame(0, 1.0 / fps); frame.index < numberOfFrames; ++frame) {
        solver->update(frame);

        if (format == "vol") {
            saveVolumeAsVol(density, rootDir, frame.index);
        } else if (format == "tga") {
            saveVolumeAsTga(density, rootDir, frame.index);
        }
    }
}

void runExample1(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    auto solver =
        GridSmokeSolver3::builder()
            .withResolution({resolutionX, 2 * resolutionX, resolutionX})
            .withDomainSizeX(1.0)
            .makeShared();

    solver->setAdvectionSolver(std::make_shared<CubicSemiLagrangian3>());

    auto grids = solver->gridSystemData();
    BoundingBox3D domain = grids->boundingBox();

    // Build emitter
    auto box = Box3::builder()
                   .withLowerCorner({0.45, -1, 0.45})
                   .withUpperCorner({0.55, 0.05, 0.55})
                   .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
                       .withSourceRegion(box)
                       .withIsOneShot(false)
                       .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0, 1);
    emitter->addStepFunctionTarget(solver->temperature(), 0, 1);

    // Build collider
    auto sphere = Sphere3::builder()
                      .withCenter({0.5, 0.3, 0.5})
                      .withRadius(0.075 * domain.width())
                      .makeShared();

    auto collider =
        RigidBodyCollider3::builder().withSurface(sphere).makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 1 (rising smoke with cubic-spline advection)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

void runExample2(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    auto solver =
        GridSmokeSolver3::builder()
            .withResolution({resolutionX, 2 * resolutionX, resolutionX})
            .withDomainSizeX(1.0)
            .makeShared();

    solver->setAdvectionSolver(std::make_shared<SemiLagrangian3>());

    auto grids = solver->gridSystemData();
    BoundingBox3D domain = grids->boundingBox();

    // Build emitter
    auto box = Box3::builder()
                   .withLowerCorner({0.45, -1, 0.45})
                   .withUpperCorner({0.55, 0.05, 0.55})
                   .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
                       .withSourceRegion(box)
                       .withIsOneShot(false)
                       .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0, 1);
    emitter->addStepFunctionTarget(solver->temperature(), 0, 1);

    // Build collider
    auto sphere = Sphere3::builder()
                      .withCenter({0.5, 0.3, 0.5})
                      .withRadius(0.075 * domain.width())
                      .makeShared();

    auto collider =
        RigidBodyCollider3::builder().withSurface(sphere).makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 2 (rising smoke with linear advection)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

void runExample3(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    auto solver =
        GridSmokeSolver3::builder()
            .withResolution({resolutionX, resolutionX / 4 * 5, resolutionX / 2})
            .withDomainSizeX(2.0)
            .withOrigin({-1, -0.15, -0.5})
            .makeShared();

    // Build emitter
    auto dragonMesh = TriangleMesh3::builder().makeShared();
    std::ifstream objFile(RESOURCES_DIR "/dragon.obj");
    if (objFile) {
        dragonMesh->readObj(&objFile);
    } else {
        fprintf(stderr, "Cannot open resources/dragon.obj\n");
        exit(EXIT_FAILURE);
    }
    auto dragon = ImplicitTriangleMesh3::builder()
                      .withTriangleMesh(dragonMesh)
                      .withResolutionX(resolutionX)
                      .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
                       .withSourceRegion(dragon)
                       .withIsOneShot(false)
                       .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0, 1);
    emitter->addStepFunctionTarget(solver->temperature(), 0, 1);

    // Print simulation info
    printf("Running example 3 (rising dragon)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

void runExample4(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    auto solver =
        GridSmokeSolver3::builder()
            .withResolution({resolutionX, 6 * resolutionX / 5, resolutionX / 2})
            .withDomainSizeX(1.0)
            .makeShared();

    solver->setBuoyancyTemperatureFactor(2.0);

    // Build emitter
    auto box = Box3::builder()
                   .withLowerCorner({0.05, 0.1, 0.225})
                   .withUpperCorner({0.1, 0.15, 0.275})
                   .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
                       .withSourceRegion(box)
                       .withIsOneShot(false)
                       .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0, 1);
    emitter->addStepFunctionTarget(solver->temperature(), 0, 1);
    emitter->addTarget(solver->velocity(),
                       [](double sdf, const Vector3D&, const Vector3D& oldVal) {
                           if (sdf < 0.05) {
                               return Vector3D(0.5, oldVal.y, oldVal.z);
                           } else {
                               return Vector3D(oldVal);
                           }
                       });

    // Print simulation info
    printf("Running example 4 (rising smoke with cubic-spline advection)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

void runExample5(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    auto solver =
        GridSmokeSolver3::builder()
            .withResolution({resolutionX, 6 * resolutionX / 5, resolutionX / 2})
            .withDomainSizeX(1.0)
            .makeShared();

    solver->setBuoyancyTemperatureFactor(2.0);
    solver->setAdvectionSolver(std::make_shared<SemiLagrangian3>());

    // Build emitter
    auto box = Box3::builder()
                   .withLowerCorner({0.05, 0.1, 0.225})
                   .withUpperCorner({0.1, 0.15, 0.275})
                   .makeShared();

    auto emitter = VolumeGridEmitter3::builder()
                       .withSourceRegion(box)
                       .withIsOneShot(false)
                       .makeShared();

    solver->setEmitter(emitter);
    emitter->addStepFunctionTarget(solver->smokeDensity(), 0, 1);
    emitter->addStepFunctionTarget(solver->temperature(), 0, 1);
    emitter->addTarget(solver->velocity(),
                       [](double sdf, const Vector3D&, const Vector3D& oldVal) {
                           if (sdf < 0.05) {
                               return Vector3D(0.5, oldVal.y, oldVal.z);
                           } else {
                               return Vector3D(oldVal);
                           }
                       });

    // Print simulation info
    printf("Running example 5 (rising smoke with linear advection)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

int main(int argc, char* argv[]) {
    bool showHelp = false;
    size_t resolutionX = 50;
    int numberOfFrames = 100;
    double fps = 60.0;
    int exampleNum = 1;
    std::string logFilename = APP_NAME ".log";
    std::string outputDir = APP_NAME "_output";
    std::string format = "tga";

    // Parsing
    auto parser =
        clara::Help(showHelp) |
        clara::Opt(resolutionX, "resolutionX")["-r"]["--resx"](
            "grid resolution in x-axis (default is 50)") |
        clara::Opt(numberOfFrames, "numberOfFrames")["-f"]["--frames"](
            "total number of frames (default is 100)") |
        clara::Opt(
            fps, "fps")["-p"]["--fps"]("frames per second (default is 60.0)") |
        clara::Opt(exampleNum, "exampleNum")["-e"]["--example"](
            "example number (between 1 and 5, default is 1)") |
        clara::Opt(logFilename, "logFilename")["-l"]["--log"](
            "log file name (default is " APP_NAME ".log)") |
        clara::Opt(outputDir, "outputDir")["-o"]["--output"](
            "output directory name (default is " APP_NAME "_output)") |
        clara::Opt(format, "format")["-m"]["--format"](
            "particle output format (tag or vol. default is tag)");

    auto result = parser.parse(clara::Args(argc, argv));
    if (!result) {
        std::cerr << "Error in command line: " << result.errorMessage() << '\n';
        exit(EXIT_FAILURE);
    }

    if (showHelp) {
        std::cout << toString(parser) << '\n';
        exit(EXIT_SUCCESS);
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
            runExample1(outputDir, resolutionX, numberOfFrames, format, fps);
            break;
        case 2:
            runExample2(outputDir, resolutionX, numberOfFrames, format, fps);
            break;
        case 3:
            runExample3(outputDir, resolutionX, numberOfFrames, format, fps);
            break;
        case 4:
            runExample4(outputDir, resolutionX, numberOfFrames, format, fps);
            break;
        case 5:
            runExample5(outputDir, resolutionX, numberOfFrames, format, fps);
            break;
        default:
            std::cout << toString(parser) << '\n';
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
