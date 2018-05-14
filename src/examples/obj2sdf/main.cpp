// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#include <jet/jet.h>

#include <example_utils/clara_utils.h>
#include <clara.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace jet;

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
    bool showHelp = false;
    std::string inputFilename;
    std::string outputFilename;
    size_t resolutionX = 100;
    double marginScale = 0.2;

    // Parsing
    auto parser =
        clara::Help(showHelp) |
        clara::Opt(inputFilename,
                   "inputFilename")["-i"]["--input"]("input obj file name") |
        clara::Opt(outputFilename,
                   "outputFilename")["-o"]["--output"]("output sdf file name") |
        clara::Opt(resolutionX, "resolutionX")["-r"]["--resx"](
            "grid resolution in x-axis (default is 100)") |
        clara::Opt(marginScale, "marginScale")["-m"]["--margin"](
            "margin scale around the sdf (default is 0.2)");

    auto result = parser.parse(clara::Args(argc, argv));
    if (!result) {
        std::cerr << "Error in command line: " << result.errorMessage() << '\n';
        exit(EXIT_FAILURE);
    }

    if (showHelp) {
        std::cout << toString(parser) << '\n';
        exit(EXIT_SUCCESS);
    }

    if (inputFilename.empty()) {
        std::cout << toString(parser) << '\n';
        exit(EXIT_FAILURE);
    }

    if (outputFilename.empty()) {
        std::cout << toString(parser) << '\n';
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
