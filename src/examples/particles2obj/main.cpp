// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/jet.h>
#include <pystring/pystring.h>

#include <example_utils/clara_utils.h>
#include <example_utils/io_utils.h>
#include <clara.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace jet;

const std::string kSpherical = "spherical";
const std::string kSph = "sph";
const std::string kZhuBridson = "zhu_bridson";
const std::string kAnisotropic = "anisotropic";

double sSphCutOffDensity = 0.5;
double sZhuBridsonCutOffThreshold = 0.25;
double sAnisoCutOffDensity = 0.5;
double sAnisoPositionSmoothingFactor = 0.5;
size_t sAnisoMinNumNeighbors = 25;

void printInfo(const Size3& resolution, const BoundingBox3D& domain,
               const Vector3D& gridSpacing, size_t numberOfParticles,
               const std::string& method) {
    printf("Resolution: %zu x %zu x %zu\n", resolution.x, resolution.y,
           resolution.z);
    printf("Domain: [%f, %f, %f] x [%f, %f, %f]\n", domain.lowerCorner.x,
           domain.lowerCorner.y, domain.lowerCorner.z, domain.upperCorner.x,
           domain.upperCorner.y, domain.upperCorner.z);
    printf("Grid spacing: [%f, %f, %f]\n", gridSpacing.x, gridSpacing.y,
           gridSpacing.z);
    printf("Number of particles: %zu\n", numberOfParticles);
    printf("Reconstruction method: %s\n", method.c_str());
}

void triangulateAndSave(const ScalarGrid3& sdf,
                        const std::string& objFilename) {
    TriangleMesh3 mesh;
    marchingCubes(sdf.constDataAccessor(), sdf.gridSpacing(), sdf.dataOrigin(),
                  &mesh, 0.0, kDirectionAll);

    std::ofstream file(objFilename.c_str());
    if (file) {
        printf("Writing %s...\n", objFilename.c_str());
        mesh.writeObj(&file);
        file.close();
    } else {
        printf("Cannot write file %s.\n", objFilename.c_str());
        exit(EXIT_FAILURE);
    }
}

void particlesToObj(const Array1<Vector3D>& positions, const Size3& resolution,
                    const Vector3D& gridSpacing, const Vector3D& origin,
                    double kernelRadius, const std::string& method,
                    const std::string& objFilename) {
    PointsToImplicit3Ptr converter;
    if (method == kSpherical) {
        converter =
            std::make_shared<SphericalPointsToImplicit3>(kernelRadius, false);
    } else if (method == kSph) {
        converter = std::make_shared<SphPointsToImplicit3>(
            kernelRadius, sSphCutOffDensity, false);
    } else if (method == kZhuBridson) {
        converter = std::make_shared<ZhuBridsonPointsToImplicit3>(
            kernelRadius, sZhuBridsonCutOffThreshold, false);
    } else {
        converter = std::make_shared<AnisotropicPointsToImplicit3>(
            kernelRadius, sAnisoCutOffDensity, sAnisoPositionSmoothingFactor,
            sAnisoMinNumNeighbors, false);
    }

    VertexCenteredScalarGrid3 sdf(resolution, gridSpacing, origin);
    printInfo(resolution, sdf.boundingBox(), gridSpacing, positions.size(),
              method);

    converter->convert(positions, &sdf);

    triangulateAndSave(sdf, objFilename);
}

int main(int argc, char* argv[]) {
    bool showHelp = false;
    std::string inputFilename;
    std::string outputFilename;
    Size3 resolution(100, 100, 100);
    Vector3D gridSpacing(0.01, 0.01, 0.01);
    Vector3D origin;
    std::string method = "anisotropic";
    double kernelRadius = 0.2;

    std::string strResolution;
    std::string strGridSpacing;
    std::string strOrigin;
    std::string strMethod;

    // Parsing
    auto parser =
        clara::Help(showHelp) |
        clara::Opt(inputFilename, "inputFilename")["-i"]["--input"](
            "input pos/xyz file name") |
        clara::Opt(outputFilename,
                   "outputFilename")["-o"]["--output"]("output obj file name") |
        clara::Opt(strResolution, "resolution")["-r"]["--resolution"](
            "grid resolution in CSV format (default is 100,100,100)") |
        clara::Opt(strGridSpacing, "gridSpacing")["-g"]["--grid_spacing"](
            "grid spacing in CSV format (default is 0.01,0.01,0.01)") |
        clara::Opt(strOrigin, "origin")["-n"]["--origin"](
            "domain origin in CSV format (default is 0,0,0)") |
        clara::Opt(method, "method")["-m"]["--method"](
            "spherical, sph, zhu_bridson, and anisotropic "
            "followed by optional method-dependent parameters (default is "
            "anisotropic)") |
        clara::Opt(kernelRadius, "kernelRadius")["-k"]["--kernel"](
            "interpolation kernel radius (default is 0.2)");

    auto result = parser.parse(clara::Args(argc, argv));
    if (!result) {
        std::cerr << "Error in command line: " << result.errorMessage() << '\n';
        exit(EXIT_FAILURE);
    }

    if (showHelp) {
        std::cout << toString(parser) << '\n';
        exit(EXIT_SUCCESS);
    }

    // Resolution
    if (!strResolution.empty()) {
        std::vector<std::string> tokens;
        pystring::split(strResolution, tokens, ",");

        if (tokens.size() == 1) {
            resolution.x = resolution.y = resolution.z =
                static_cast<size_t>(atoi(strResolution.c_str()));
        } else if (tokens.size() == 3) {
            resolution.x = static_cast<size_t>(atoi(tokens[0].c_str()));
            resolution.y = static_cast<size_t>(atoi(tokens[1].c_str()));
            resolution.z = static_cast<size_t>(atoi(tokens[2].c_str()));
        }
    }

    // Grid spacing
    if (!strGridSpacing.empty()) {
        std::vector<std::string> tokens;
        pystring::split(strGridSpacing, tokens, ",");

        if (tokens.size() == 1) {
            gridSpacing.x = gridSpacing.y = gridSpacing.z =
                atof(strGridSpacing.c_str());
        } else if (tokens.size() == 3) {
            gridSpacing.x = atof(tokens[0].c_str());
            gridSpacing.y = atof(tokens[1].c_str());
            gridSpacing.z = atof(tokens[2].c_str());
        }
    }

    // Origin
    if (!strOrigin.empty()) {
        std::vector<std::string> tokens;
        pystring::split(strOrigin, tokens, ",");

        if (tokens.size() == 1) {
            origin.x = origin.y = origin.z = atof(strOrigin.c_str());
        } else if (tokens.size() == 3) {
            origin.x = atof(tokens[0].c_str());
            origin.y = atof(tokens[1].c_str());
            origin.z = atof(tokens[2].c_str());
        }
    }

    // Method
    if (!strMethod.empty()) {
        std::vector<std::string> tokens;
        pystring::split(strMethod, tokens, ",");

        method = tokens[0];

        if (method == kSpherical) {
            // No other options accepted
        } else if (method == kSph) {
            if (tokens.size() > 1) {
                sSphCutOffDensity = atof(tokens[1].c_str());
            }
        } else if (method == kZhuBridson) {
            if (tokens.size() > 1) {
                sZhuBridsonCutOffThreshold = atof(tokens[1].c_str());
            }
        } else if (method == kAnisotropic) {
            if (tokens.size() > 1) {
                sAnisoCutOffDensity = atof(tokens[1].c_str());
            }
            if (tokens.size() > 2) {
                sAnisoPositionSmoothingFactor = atof(tokens[2].c_str());
            }
            if (tokens.size() > 3) {
                sAnisoMinNumNeighbors =
                    static_cast<size_t>(atoi(tokens[3].c_str()));
            }
        } else {
            fprintf(stderr, "Unknown method %s.\n", method.c_str());
            std::cout << toString(parser) << '\n';
            exit(EXIT_FAILURE);
        }
    }

    if (inputFilename.empty() || outputFilename.empty()) {
        std::cout << toString(parser) << '\n';
        exit(EXIT_FAILURE);
    }

    // Read particle positions
    Array1<Vector3D> positions;
    if (!readPositions(inputFilename, positions)) {
        printf("Cannot read file %s.\n", inputFilename.c_str());
        exit(EXIT_FAILURE);
    }

    // Run marching cube and save it to the disk
    particlesToObj(positions, resolution, gridSpacing, origin, kernelRadius,
                   method, outputFilename);

    return EXIT_SUCCESS;
}
