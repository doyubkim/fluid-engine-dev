// Copyright (c) 2016 Doyub Kim

#include <jet/jet.h>

#include <getopt.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

using namespace jet;

void printUsage() {
    printf(
        "Usage: obj2sdf "
        "-i input_obj -o output_sdf -r resolution -m margin_scale\n"
        "   -i, --input: input obj filename\n"
        "   -o, --output: output sdf filename\n"
        "   -r, --resx: grid resolution in x-axis (default: 100)\n"
        "   -m, --margin: margin scale around the sdf (default: 0.2)\n"
        "   -h, --help: print this message\n");
}

void saveTriangleMeshData(
    const TriangleMesh3& data,
    const std::string& filename) {
    std::ofstream file(filename.c_str());
    if (file) {
        data.writeObj(&file);
        file.close();
    }
}

int main(int argc, char* argv[]) {
    std::string inputFilename;
    std::string outputFilename;
    size_t resolutionX = 100;
    double marginScale = 0.2;

    // Parse options
    static struct option longOptions[] = {
        {"input",   required_argument,  0,  'i' },
        {"output",  required_argument,  0,  'o' },
        {"resx",    optional_argument,  0,  'r' },
        {"margin",  optional_argument,  0,  'm' },
        {"help",    optional_argument,  0,  'h' },
        {0,         0,                  0,   0  }
    };

    int opt = 0;
    int long_index = 0;
    while ((opt = getopt_long(
        argc, argv, "i:o:r:m:h", longOptions, &long_index)) != -1) {
        switch (opt) {
            case 'i':
                inputFilename = optarg;
                break;
            case 'o':
                outputFilename = optarg;
                break;
            case 'r':
                resolutionX = static_cast<size_t>(atoi(optarg));
                break;
            case 'm':
                marginScale = std::max(atof(optarg), 0.0);
                break;
            case 'h':
                printUsage();
                exit(EXIT_SUCCESS);
            default:
                printUsage();
                exit(EXIT_FAILURE);
        }
    }

    if (inputFilename.empty()) {
        printUsage();
        exit(EXIT_FAILURE);
    }

    if (outputFilename.empty()) {
        printUsage();
        exit(EXIT_FAILURE);
    }

    TriangleMesh3 triMesh;

    std::ifstream objFile(inputFilename.c_str());
    if (objFile) {
        printf("Reading obj file %s\n", inputFilename.c_str());
        triMesh.readObj(&objFile);
        objFile.close();
    } else {
        fprintf(stderr, "Failed to open file %s\n", inputFilename.c_str());
        exit(EXIT_FAILURE);
    }

    BoundingBox3D box = triMesh.boundingBox();
    Vector3D scale(box.width(), box.height(), box.depth());
    box.lowerCorner -= marginScale * scale;
    box.upperCorner += marginScale * scale;

    size_t resolutionY = static_cast<size_t>(
        std::ceil(resolutionX * box.height() / box.width()));
    size_t resolutionZ = static_cast<size_t>(
        std::ceil(resolutionX * box.depth() / box.width()));

    printf(
        "Vertex-centered grid size: %zu x %zu x %zu\n",
        resolutionX, resolutionY, resolutionZ);

    double dx = box.width() / resolutionX;

    VertexCenteredScalarGrid3 grid(
        resolutionX, resolutionY, resolutionZ,
        dx, dx, dx,
        box.lowerCorner.x, box.lowerCorner.y, box.lowerCorner.z);

    BoundingBox3D domain = grid.boundingBox();
    printf(
        "Domain size: [%f, %f, %f] x [%f, %f, %f]\n",
        domain.lowerCorner.x, domain.lowerCorner.y, domain.lowerCorner.z,
        domain.upperCorner.x, domain.upperCorner.y, domain.upperCorner.z);
    printf("Generating SDF...");

    triangleMeshToSdf(triMesh, &grid);

    printf("done\n");

    std::ofstream sdfFile(outputFilename.c_str(), std::ofstream::binary);
    if (sdfFile) {
        printf("Writing to vertex-centered grid %s\n", outputFilename.c_str());

        std::vector<uint8_t> buffer;
        grid.serialize(&buffer);
        sdfFile.write(reinterpret_cast<char*>(buffer.data()), buffer.size());
        sdfFile.close();
    } else {
        fprintf(stderr, "Failed to write file %s\n", outputFilename.c_str());
        exit(EXIT_FAILURE);
    }

    TriangleMesh3 triMesh2;
    marchingCubes(
        grid.constDataAccessor(),
        grid.gridSpacing(),
        grid.origin(),
        &triMesh2,
        0,
        kDirectionAll);

    saveTriangleMeshData(triMesh2, outputFilename + "_previz.obj");

    return EXIT_SUCCESS;
}
