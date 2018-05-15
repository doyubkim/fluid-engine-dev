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

#define APP_NAME "hybrid_liquid_sim"

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

void printInfo(const PicSolver3Ptr& solver) {
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

void runSimulation(const std::string& rootDir, const PicSolver3Ptr& solver,
                   int numberOfFrames, const std::string& format, double fps) {
    auto particles = solver->particleSystemData();

    for (Frame frame(0, 1.0 / fps); frame.index < numberOfFrames; ++frame) {
        solver->update(frame);
        if (format == "xyz") {
            saveParticleAsXyz(particles, rootDir, frame.index);
        } else if (format == "pos") {
            saveParticleAsPos(particles, rootDir, frame.index);
        }
    }
}

// Water-drop example (FLIP)
void runExample1(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    auto solver =
        FlipSolver3::builder()
            .withResolution({resolutionX, 2 * resolutionX, resolutionX})
            .withDomainSizeX(1.0)
            .makeShared();

    auto grids = solver->gridSystemData();
    auto particles = solver->particleSystemData();

    Vector3D gridSpacing = grids->gridSpacing();
    double dx = gridSpacing.x;
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

    auto emitter1 = VolumeParticleEmitter3::builder()
                        .withSurface(plane)
                        .withSpacing(0.5 * dx)
                        .withMaxRegion(domain)
                        .withIsOneShot(true)
                        .makeShared();
    emitter1->setPointGenerator(std::make_shared<GridPointGenerator3>());

    auto emitter2 = VolumeParticleEmitter3::builder()
                        .withSurface(sphere)
                        .withSpacing(0.5 * dx)
                        .withMaxRegion(domain)
                        .withIsOneShot(true)
                        .makeShared();
    emitter2->setPointGenerator(std::make_shared<GridPointGenerator3>());

    auto emitterSet = ParticleEmitterSet3::builder()
                          .withEmitters({emitter1, emitter2})
                          .makeShared();

    solver->setParticleEmitter(emitterSet);

    // Print simulation info
    printf("Running example 1 (water-drop with FLIP)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

// Water-drop example (PIC)
void runExample2(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    auto solver =
        PicSolver3::builder()
            .withResolution({resolutionX, 2 * resolutionX, resolutionX})
            .withDomainSizeX(1.0)
            .makeShared();

    auto grids = solver->gridSystemData();
    auto particles = solver->particleSystemData();

    Vector3D gridSpacing = grids->gridSpacing();
    double dx = gridSpacing.x;
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

    auto emitter1 = VolumeParticleEmitter3::builder()
                        .withSurface(plane)
                        .withSpacing(0.5 * dx)
                        .withMaxRegion(domain)
                        .withIsOneShot(true)
                        .makeShared();
    emitter1->setPointGenerator(std::make_shared<GridPointGenerator3>());

    auto emitter2 = VolumeParticleEmitter3::builder()
                        .withSurface(sphere)
                        .withSpacing(0.5 * dx)
                        .withMaxRegion(domain)
                        .withIsOneShot(true)
                        .makeShared();
    emitter2->setPointGenerator(std::make_shared<GridPointGenerator3>());

    auto emitterSet = ParticleEmitterSet3::builder()
                          .withEmitters({emitter1, emitter2})
                          .makeShared();

    solver->setParticleEmitter(emitterSet);

    // Print simulation info
    printf("Running example 1 (water-drop with PIC)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

// Dam-breaking example (FLIP)
void runExample3(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    Size3 resolution{3 * resolutionX, 2 * resolutionX, (3 * resolutionX) / 2};
    auto solver = FlipSolver3::builder()
                      .withResolution(resolution)
                      .withDomainSizeX(3.0)
                      .makeShared();
    solver->setUseCompressedLinearSystem(true);

    auto grids = solver->gridSystemData();
    double dx = grids->gridSpacing().x;
    BoundingBox3D domain = grids->boundingBox();
    double lz = domain.depth();

    // Build emitter
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
                       .withMaxRegion(domain)
                       .withSpacing(0.5 * dx)
                       .makeShared();

    emitter->setPointGenerator(std::make_shared<GridPointGenerator3>());
    solver->setParticleEmitter(emitter);

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

    auto collider =
        RigidBodyCollider3::builder().withSurface(cylSet).makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 3 (dam-breaking with FLIP)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

// Dam-breaking example (PIC)
void runExample4(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    Size3 resolution{3 * resolutionX, 2 * resolutionX, (3 * resolutionX) / 2};
    auto solver = PicSolver3::builder()
                      .withResolution(resolution)
                      .withDomainSizeX(3.0)
                      .makeShared();
    solver->setUseCompressedLinearSystem(true);

    auto grids = solver->gridSystemData();
    double dx = grids->gridSpacing().x;
    BoundingBox3D domain = grids->boundingBox();
    double lz = domain.depth();

    // Build emitter
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
                       .withMaxRegion(domain)
                       .withSpacing(0.5 * dx)
                       .makeShared();

    emitter->setPointGenerator(std::make_shared<GridPointGenerator3>());
    solver->setParticleEmitter(emitter);

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

    auto collider =
        RigidBodyCollider3::builder().withSurface(cylSet).makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 4 (dam-breaking with PIC)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

// Dam-breaking example (APIC)
void runExample5(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    Size3 resolution{3 * resolutionX, 2 * resolutionX, (3 * resolutionX) / 2};
    auto solver = ApicSolver3::builder()
                      .withResolution(resolution)
                      .withDomainSizeX(3.0)
                      .makeShared();
    solver->setUseCompressedLinearSystem(true);

    auto grids = solver->gridSystemData();
    double dx = grids->gridSpacing().x;
    BoundingBox3D domain = grids->boundingBox();
    double lz = domain.depth();

    // Build emitter
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
                       .withMaxRegion(domain)
                       .withSpacing(0.5 * dx)
                       .makeShared();

    emitter->setPointGenerator(std::make_shared<GridPointGenerator3>());
    solver->setParticleEmitter(emitter);

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

    auto collider =
        RigidBodyCollider3::builder().withSurface(cylSet).makeShared();

    solver->setCollider(collider);

    // Print simulation info
    printf("Running example 5 (dam-breaking with APIC)\n");
    printInfo(solver);

    // Run simulation
    runSimulation(rootDir, solver, numberOfFrames, format, fps);
}

void runExample6(const std::string& rootDir, size_t resolutionX,
                 int numberOfFrames, const std::string& format, double fps) {
    // Build solver
    auto solver = ApicSolver3::builder()
                      .withResolution({resolutionX, resolutionX, resolutionX})
                      .withDomainSizeX(1.0)
                      .makeShared();
    solver->setUseCompressedLinearSystem(true);

    // Build collider
    auto sphere = Sphere3::builder()
                      .withCenter({0.5, 0.5, 0.5})
                      .withRadius(0.4)
                      .withIsNormalFlipped(true)
                      .makeShared();

    auto collider =
        RigidBodyCollider3::builder().withSurface(sphere).makeShared();

    solver->setCollider(collider);

    // Manually emit particles
    printf("Start emitting particles...\n");
    std::mt19937 rng;
    std::uniform_real_distribution<> dist(-0.1 * solver->gridSpacing().x,
                                          0.1 * solver->gridSpacing().x);
    BccLatticePointGenerator pointGenerator;
    pointGenerator.forEachPoint(
        BoundingBox3D({0.75, 0, 0}, {1, 1, 1}), 0.5 * solver->gridSpacing().x,
        [&](const Vector3D& pt) -> bool {
            Vector3D newPos = pt + Vector3D{dist(rng), dist(rng), dist(rng)};
            if ((pt - Vector3D{0.5, 0.5, 0.5}).length() < 0.4) {
                solver->particleSystemData()->addParticle(newPos);
            }
            return true;
        });
    printf("Number of particles: %zu\n",
           solver->particleSystemData()->numberOfParticles());

    // Print simulation info
    printf("Running example 6 (sphere boundary with APIC)\n");
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
    std::string format = "xyz";

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
            "example number (between 1 and 6, default is 1)") |
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
        case 6:
            runExample6(outputDir, resolutionX, numberOfFrames, format, fps);
            break;
        default:
            std::cout << toString(parser) << '\n';
            exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
