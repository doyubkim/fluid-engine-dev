// Copyright (c) 2016 Doyub Kim

#include <jet/jet.h>
#include <pystring/pystring.h>

#include <getopt.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

using namespace jet;

void printUsage() {
    printf(
        "Usage: particles2xml "
        "-i input_pos -o output_xml \n"
        "   -i, --input: input particle position filename\n"
        "   -o, --output: output obj filename\n"
        "   -h, --help: print this message\n");
}

void printInfo(size_t numberOfParticles) {
    printf("Number of particles: %zu\n", numberOfParticles);
}

void particlesToXml(
    const Array1<Vector3D>& positions,
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
            snprintf(
                buffer,
                sizeof(buffer),
                "<translate x=\"%f\" y=\"%f\" z=\"%f\"/>",
                pos.x,
                pos.y,
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
    std::string inputFilename;
    std::string outputFilename;

    // Parse options
    static struct option longOptions[] = {
        {"input",       required_argument,  0,  'i' },
        {"output",      required_argument,  0,  'o' },
        {"help",        optional_argument,  0,  'h' },
        {0,             0,                  0,   0  }
    };

    int opt = 0;
    int long_index = 0;
    while ((opt = getopt_long(
        argc, argv, "i:o:h", longOptions, &long_index)) != -1) {
        switch (opt) {
            case 'i':
                inputFilename = optarg;
                break;
            case 'o':
                outputFilename = optarg;
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
    particlesToXml(
        positions,
        outputFilename);

    return EXIT_SUCCESS;
}
