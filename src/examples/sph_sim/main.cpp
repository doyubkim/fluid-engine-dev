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

#define APP_NAME "sph_sim"

using namespace jet;

void saveParticleAsPos(const ParticleSystemData3Ptr& particles,
                       const std::string& rootDir, int frameCnt) {
    Array1<Vector3D> positions(particles->numberOfParticles());
    copyRange1(particles->positions(), particles->numberOfParticles(),
               &positions);
    char basename[256];
    snprintf(basename, sizeof(basename), "frame_%06d.pos", frameCnt);
    std::string filename = pystring::os::path::join(rootDir, basename);
    std::ofstream file(filename.c_str(), std::ios::binary);
    if (file) {
        printf("Writing %s...\n", filename.c_str());
        std::vector<uint8_t> buffer;
        serialize(positions.constAccessor(), &buffer);
        file.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
        file.close();
    }
}

void saveParticleAsXyz(const ParticleSystemData3Ptr& particles,
                       const std::string& rootDir, int frameCnt) {
    Array1<Vector3D> positions(particles->numberOfParticles());
    copyRange1(particles->positions(), particles->numberOfParticles(),
               &positions);
    char basename[256];
    snprintf(basename, sizeof(basename), "frame_%06d.xyz", frameCnt);
    std::string filename = pystring::os::path::join(rootDir, basename);
    std::ofstream file(filename.c_str());
    if (file) {
        printf("Writing %s...\n", filename.c_str());
        for (const auto& pt : positions) {
            file << pt.x << ' ' << pt.y << ' ' << pt.z << std::endl;
        }
        file.close();
    }
}

void printInfo(const SphSolver3Ptr& solver) {
    auto particles = solver->sphSystemData();
    printf("Number of particles: %zu\n", particles->numberOfParticles());
}

void runSimulation(const std::string& rootDir, const SphSolver3Ptr& solver,
                   int numberOfFrames, const std::string& format, double fps) {
    auto particles = solver->sphSystemData();

    for (Frame frame(0, 1.0 / fps); frame.index < numberOfFrames; ++frame) {
        solver->update(frame);
        if (format == "xyz") {
            saveParticleAsXyz(particles, rootDir, frame.index);
        } else if (format == "pos") {
            saveParticleAsPos(particles, rootDir, frame.index);
        }
    }
}

// Water-drop example (PCISPH)
void runExample1(const std::string& rootDir, double targetSpacing,
                 int numberOfFrames, const std::string& format, double fps) {
    BoundingBox3D domain(Vector3D(), Vector3D(1, 2, 1));

    // Build solver
    auto solver = PciSphSolver3::builder()
                      .withTargetDensity(1000.0)
                      .withTargetSpacing(targetSpacing)
                      .makeShared();

    solver->setPseudoViscosityCoefficient(0.0);

    // Build emitter
    BoundingBox3D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

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

    auto emitter = VolumeParticleEmitter3::builder()
                       .withImplicitSurface(surfaceSet)
                       .withSpacing(targetSpacing)
                       .withMaxRegion(sourceBound)
                       .withIsOneShot(true)
                       .makeShared();

    solver->setEmitter(emitter);

    // Build collider
    auto box = Box3::builder()
                   .withIsNormalFlipped(true)
                   .withBoundingBox(domain)
                   .makeShared();

    auto collider = RigidBodyCollider3::builder().withSurface(box).makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 1 (water-drop with PCISPH)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

// Water-drop example (SPH)
void runExample2(const std::string& rootDir, double targetSpacing,
                 int numberOfFrames, const std::string& format, double fps) {
    BoundingBox3D domain(Vector3D(), Vector3D(1, 2, 1));

    auto solver = SphSolver3::builder()
                      .withTargetDensity(1000.0)
                      .withTargetSpacing(targetSpacing)
                      .makeShared();

    solver->setPseudoViscosityCoefficient(0.0);

    // Build emitter
    BoundingBox3D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

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

    auto emitter = VolumeParticleEmitter3::builder()
                       .withImplicitSurface(surfaceSet)
                       .withSpacing(targetSpacing)
                       .withMaxRegion(sourceBound)
                       .withIsOneShot(true)
                       .makeShared();

    solver->setEmitter(emitter);

    // Build collider
    auto box = Box3::builder()
                   .withIsNormalFlipped(true)
                   .withBoundingBox(domain)
                   .makeShared();

    auto collider = RigidBodyCollider3::builder().withSurface(box).makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 2 (water-drop with SPH)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

// Dam-breaking example
void runExample3(const std::string& rootDir, double targetSpacing,
                 int numberOfFrames, const std::string& format, double fps) {
    BoundingBox3D domain(Vector3D(), Vector3D(3, 2, 1.5));
    double lz = domain.depth();

    // Build solver
    auto solver = PciSphSolver3::builder()
                      .withTargetDensity(1000.0)
                      .withTargetSpacing(targetSpacing)
                      .makeShared();

    solver->setPseudoViscosityCoefficient(0.0);
    solver->setTimeStepLimitScale(10.0);

    // Build emitter
    BoundingBox3D sourceBound(domain);
    sourceBound.expand(-targetSpacing);

    auto box1 =
        Box3::builder()
            .withLowerCorner({0, 0, 0})
            .withUpperCorner({0.5 + 0.001, 0.75 + 0.001, 0.75 * lz + 0.001})
            .makeShared();

    auto box2 =
        Box3::builder()
            .withLowerCorner({2.5 - 0.001, 0, 0.25 * lz - 0.001})
            .withUpperCorner({3.5 + 0.001, 0.75 + 0.001, 1.5 * lz + 0.001})
            .makeShared();

    auto boxSet = ImplicitSurfaceSet3::builder()
                      .withExplicitSurfaces({box1, box2})
                      .makeShared();

    auto emitter = VolumeParticleEmitter3::builder()
                       .withSurface(boxSet)
                       .withMaxRegion(sourceBound)
                       .withSpacing(targetSpacing)
                       .makeShared();

    solver->setEmitter(emitter);

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

    auto box = Box3::builder()
                   .withIsNormalFlipped(true)
                   .withBoundingBox(domain)
                   .makeShared();

    auto surfaceSet = ImplicitSurfaceSet3::builder()
                          .withExplicitSurfaces({cyl1, cyl2, cyl3, box})
                          .makeShared();

    auto collider =
        RigidBodyCollider3::builder().withSurface(surfaceSet).makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 3 (dam-breaking with PCISPH)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

int main(int argc, char* argv[]) {
    bool showHelp = false;
    double targetSpacing = 0.02;
    int numberOfFrames = 100;
    double fps = 60.0;
    int exampleNum = 1;
    std::string logFilename = APP_NAME ".log";
    std::string outputDir = APP_NAME "_output";
    std::string format = "xyz";

    // Parsing
    auto parser =
        clara::Help(showHelp) |
        clara::Opt(targetSpacing, "targetSpacing")["-s"]["--spacing"](
            "target particle spacing (default is 0.02)") |
        clara::Opt(numberOfFrames, "numberOfFrames")["-f"]["--frames"](
            "total number of frames (default is 100)") |
        clara::Opt(
            fps, "fps")["-p"]["--fps"]("frames per second (default is 60.0)") |
        clara::Opt(exampleNum, "exampleNum")["-e"]["--example"](
            "example number (between 1 and 3, default is 1)") |
        clara::Opt(logFilename, "logFilename")["-l"]["--log"](
            "log file name (default is " APP_NAME ".log)") |
        clara::Opt(outputDir, "outputDir")["-o"]["--output"](
            "output directory name (default is " APP_NAME "_output)") |
        clara::Opt(format, "format")["-m"]["--format"](
            "particle output format (xyz or pos. default is xyz)");

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
            runExample1(outputDir, targetSpacing, numberOfFrames, format, fps);
            break;
        case 2:
            runExample2(outputDir, targetSpacing, numberOfFrames, format, fps);
            break;
        case 3:
            runExample3(outputDir, targetSpacing, numberOfFrames, format, fps);
            break;
        default:
            std::cout << toString(parser) << '\n';
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
