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
        "-i input_pos -o output_xml "
        "   -i, --input: input particle position filename\n"
        "   -o, --output: output obj filename\n");
}

void printInfo(size_t numberOfParticles) {
    printf("Number of particles: %zu\n", numberOfParticles);
}

void particlesToXml(
    const Array1<Vector3D>& positions,
    const Size3& resolution,
    const Vector3D& gridSpacing,
    const Vector3D& origin,
    double kernelRadius,
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
    Size3 resolution(100, 100, 100);
    Vector3D gridSpacing(0.01, 0.01, 0.01);
    Vector3D origin;
    double kernelRadius = 0.2;

    // Parse options
    static struct option longOptions[] = {
        {"input",       required_argument,  0,  'i' },
        {"output",      required_argument,  0,  'o' },
        {"resolution",  optional_argument,  0,  'r' },
        {"gridspacing", optional_argument,  0,  'g' },
        {"origin",      optional_argument,  0,  'n' },
        {"kernel",      optional_argument,  0,  'k' },
        {0,             0,                  0,   0  }
    };

    int opt = 0;
    int long_index = 0;
    while ((opt = getopt_long(
        argc, argv, "i:o:r:g:n:k:", longOptions, &long_index)) != -1) {
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
                    resolution.x = resolution.y = resolution.z
                        = static_cast<size_t>(atoi(optarg));

                } else if (tokens.size() == 3) {
                    resolution.x
                        = static_cast<size_t>(atoi(tokens[0].c_str()));
                    resolution.y
                        = static_cast<size_t>(atoi(tokens[1].c_str()));
                    resolution.z
                        = static_cast<size_t>(atoi(tokens[2].c_str()));
                }
                break;
            }
            case 'g': {
                std::vector<std::string> tokens;
                pystring::split(optarg, tokens, ",");
                if (tokens.size() == 1) {
                    gridSpacing.x = gridSpacing.y = gridSpacing.z
                        = atof(optarg);
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
        positions.deserialize(&positionFile);
        positionFile.close();
    } else {
        printf("Cannot read file %s.\n", inputFilename.c_str());
        exit(EXIT_FAILURE);
    }

    // Run marching cube and save it to the disk
    particlesToXml(
        positions,
        resolution,
        gridSpacing,
        origin,
        kernelRadius,
        outputFilename);

    return EXIT_SUCCESS;
}
