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

void printInfo(size_t numberOfParticles) {
    printf("Number of particles: %zu\n", numberOfParticles);
}

void particlesToXml(const Array1<Vector3D>& positions,
                    const std::string& xmlFilename) {
    printInfo(positions.size());

    std::ofstream file(xmlFilename.c_str());
    if (file) {
        printf("Writing %s...\n", xmlFilename.c_str());

        file << "<scene version=\"0.5.0\">";

        for (const auto& pos : positions) {
            file << "<shape type=\"instance\">";
            file << "<ref id=\"spheres\"/>";
            file << "<transform name=\"toWorld\">";

            char buffer[64];
            snprintf(buffer, sizeof(buffer),
                     "<translate x=\"%f\" y=\"%f\" z=\"%f\"/>", pos.x, pos.y,
                     pos.z);
            file << buffer;

            file << "</transform>";
            file << "</shape>";
        }

        file << "</scene>";

        file.close();
    } else {
        printf("Cannot write file %s.\n", xmlFilename.c_str());
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    bool showHelp = false;
    std::string inputFilename;
    std::string outputFilename;

    // Parsing
    auto parser =
        clara::Help(showHelp) |
        clara::Opt(inputFilename, "inputFilename")["-i"]["--input"](
            "input particle pos/xyz file name") |
        clara::Opt(outputFilename,
                   "outputFilename")["-o"]["--output"]("output xml file name");

    auto result = parser.parse(clara::Args(argc, argv));
    if (!result) {
        std::cerr << "Error in command line: " << result.errorMessage() << '\n';
        exit(EXIT_FAILURE);
    }

    if (showHelp) {
        std::cout << toString(parser) << '\n';
        exit(EXIT_SUCCESS);
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
    particlesToXml(positions, outputFilename);

    return EXIT_SUCCESS;
}
