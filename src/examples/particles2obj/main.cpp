// Copyright (c) 2017 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/jet.h>
#include <pystring/pystring.h>

#include <getopt.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

using namespace jet;

const std::string kSpherical = "spherical";
const std::string kSph = "sph";
const std::string kZhuBridson = "zhu_bridson";
const std::string kAnisotropic = "anisotropic";

void printUsage() {
    printf(
        "Usage: particles2obj "
        "-i input_pos -o output_obj "
        "-r resx,resy,resz "
        "-g dx,dy,dz "
        "-n ox,oy,oz "
        "-k kernel_radius\n"
        "   -i, --input: input particle position filename\n"
        "   -o, --output: output obj filename\n"
        "   -r, --resolution: grid resolution in CSV format "
        "(default: 100,100,100)\n"
        "   -g, --gridspacing: grid spacing in CSV format "
        "(default: 0.01,0.01,0.01)\n"
        "   -n, --origin: domain origin in CSV format (default: 0,0,0)\n"
        "   -m, --method: spherical, sph, zhu_bridson, and anisotropic "
        "(default: anisotropic)\n"
        "   -k, --kernel: interpolation kernel radius (default: 0.2)\n"
        "   -h, --help: print this message\n");
}

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
        converter = std::make_shared<SphericalPointsToImplicit3>(
            0.5 * kernelRadius, false);
    } else if (method == kSph) {
        converter =
            std::make_shared<SphPointsToImplicit3>(kernelRadius, 0.5, false);
    } else if (method == kZhuBridson) {
        converter = std::make_shared<ZhuBridsonPointsToImplicit3>(
            2.0 * kernelRadius, 0.25, false);
    } else {
        converter = std::make_shared<AnisotropicPointsToImplicit3>(
            kernelRadius, 0.5, 0.1, 25, false);
    }

    VertexCenteredScalarGrid3 sdf(resolution, gridSpacing, origin);
    printInfo(resolution, sdf.boundingBox(), gridSpacing, positions.size(),
              method);

    converter->convert(positions, &sdf);

    triangulateAndSave(sdf, objFilename);
}

int main(int argc, char* argv[]) {
    std::string inputFilename;
    std::string outputFilename;
    Size3 resolution(100, 100, 100);
    Vector3D gridSpacing(0.01, 0.01, 0.01);
    Vector3D origin;
    std::string method = "anisotropic";
    double kernelRadius = 0.2;

    // Parse options
    static struct option longOptions[] = {
        {"input", required_argument, 0, 'i'},
        {"output", required_argument, 0, 'o'},
        {"resolution", optional_argument, 0, 'r'},
        {"gridspacing", optional_argument, 0, 'g'},
        {"origin", optional_argument, 0, 'n'},
        {"kernel", optional_argument, 0, 'k'},
        {"method", optional_argument, 0, 'm'},
        {"help", optional_argument, 0, 'h'},
        {0, 0, 0, 0}};

    int opt = 0;
    int long_index = 0;
    while ((opt = getopt_long(argc, argv, "i:o:r:g:n:k:m:h", longOptions,
                              &long_index)) != -1) {
        switch (opt) {
            case 'i':
                inputFilename = optarg;
                break;
            case 'o':
                outputFilename = optarg;
                break;
            case 'r': {
                std::vector<std::string> tokens;
                pystring::split(optarg, tokens, ",");

                if (tokens.size() == 1) {
                    resolution.x = resolution.y = resolution.z =
                        static_cast<size_t>(atoi(optarg));

                } else if (tokens.size() == 3) {
                    resolution.x = static_cast<size_t>(atoi(tokens[0].c_str()));
                    resolution.y = static_cast<size_t>(atoi(tokens[1].c_str()));
                    resolution.z = static_cast<size_t>(atoi(tokens[2].c_str()));
                }
                break;
            }
            case 'g': {
                std::vector<std::string> tokens;
                pystring::split(optarg, tokens, ",");
                if (tokens.size() == 1) {
                    gridSpacing.x = gridSpacing.y = gridSpacing.z =
                        atof(optarg);
                } else if (tokens.size() == 3) {
                    gridSpacing.x = atof(tokens[0].c_str());
                    gridSpacing.y = atof(tokens[1].c_str());
                    gridSpacing.z = atof(tokens[2].c_str());
                }
                break;
            }
            case 'n': {
                std::vector<std::string> tokens;
                pystring::split(optarg, tokens, ",");
                if (tokens.size() == 1) {
                    origin.x = origin.y = origin.z = atof(optarg);
                } else if (tokens.size() == 3) {
                    origin.x = atof(tokens[0].c_str());
                    origin.y = atof(tokens[1].c_str());
                    origin.z = atof(tokens[2].c_str());
                }
                break;
            }
            case 'k': {
                kernelRadius = atof(optarg);
                break;
            }
            case 'm':
                if (optarg == kSpherical || optarg == kSph ||
                    optarg == kZhuBridson || optarg == kAnisotropic) {
                    method = optarg;
                } else {
                    fprintf(stderr, "Unknown method %s.\n", optarg);
                    printUsage();
                    exit(EXIT_SUCCESS);
                }
                break;
            case 'h':
                printUsage();
                exit(EXIT_SUCCESS);
            default:
                printUsage();
                exit(EXIT_FAILURE);
        }
    }

    if (inputFilename.empty() || outputFilename.empty()) {
        printUsage();
        exit(EXIT_FAILURE);
    }

    // Read particle positions
    Array1<Vector3D> positions;
    std::ifstream positionFile(inputFilename.c_str(), std::ifstream::binary);
    if (positionFile) {
        std::vector<uint8_t> buffer(
            (std::istreambuf_iterator<char>(positionFile)),
            (std::istreambuf_iterator<char>()));
        deserialize(buffer, &positions);
        positionFile.close();
    } else {
        printf("Cannot read file %s.\n", inputFilename.c_str());
        exit(EXIT_FAILURE);
    }

    // Run marching cube and save it to the disk
    particlesToObj(positions, resolution, gridSpacing, origin, kernelRadius,
                   method, outputFilename);

    return EXIT_SUCCESS;
}
