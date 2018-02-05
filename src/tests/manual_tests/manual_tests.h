// Copyright (c) 2018 Doyub Kim
//
// I am making my contributions/submissions to this project solely in my
// personal capacity and am not conveying any rights to any intellectual
// property of any third parties.

#ifndef SRC_TESTS_MANUAL_TESTS_MANUAL_TESTS_H_
#define SRC_TESTS_MANUAL_TESTS_MANUAL_TESTS_H_

#include <jet/array_accessor1.h>
#include <jet/array_accessor2.h>
#include <jet/array_accessor3.h>
#include <jet/triangle_mesh3.h>

#include <cnpy/cnpy.h>
#include <gtest/gtest.h>
#include <pystring/pystring.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#ifdef JET_WINDOWS
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#define JET_TESTS_OUTPUT_DIR "manual_tests_output"

inline void createDirectory(const std::string& dirname) {
    std::vector<std::string> tokens;
    pystring::split(dirname, tokens, "/");
    std::string partialDir;
    for (const auto& token : tokens) {
        partialDir = pystring::os::path::join(partialDir, token);
#ifdef JET_WINDOWS
        _mkdir(partialDir.c_str());
#else
        mkdir(partialDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
    }
}

#define JET_TESTS(testSetName) \
    class testSetName##Tests : public ::testing::Test { \
     private: \
        std::string _testCollectionDir; \
        std::string _currentTestCaseName; \
        std::string _currentTestDir; \
     protected: \
        void SetUp() override { \
            _testCollectionDir \
                = pystring::os::path::join( \
                    JET_TESTS_OUTPUT_DIR, #testSetName); \
            createDirectory(_testCollectionDir); \
        } \
        void createTestDirectory(const std::string& name) { \
            _currentTestDir = getTestDirectoryName(name); \
            createDirectory(_currentTestDir); \
        } \
        std::string getFullFilePath(const std::string& name) { \
            if (!_currentTestDir.empty()) { \
                return pystring::os::path::join(_currentTestDir, name); \
            } else { \
                return name; \
            } \
        } \
        template <typename T> \
        void saveData( \
            const ConstArrayAccessor1<T>& data, \
            const std::string& name) { \
            std::string filename = getFullFilePath(name); \
            unsigned int dim[1] = { \
                static_cast<unsigned int>(data.size()) \
            }; \
            cnpy::npy_save(filename, data.data(), dim, 1, "w"); \
        } \
        template <typename T> \
        void saveData( \
            const ConstArrayAccessor1<T>& data, \
            size_t size, const std::string& name) { \
            std::string filename = getFullFilePath(name); \
            unsigned int dim[1] = { \
                static_cast<unsigned int>(size) \
            }; \
            cnpy::npy_save(filename, data.data(), dim, 1, "w"); \
        } \
        template <typename T> \
        void saveData( \
            const ConstArrayAccessor2<T>& data, \
            const std::string& name) { \
            std::string filename = getFullFilePath(name); \
            unsigned int dim[2] = { \
                static_cast<unsigned int>(data.height()), \
                static_cast<unsigned int>(data.width()) \
            }; \
            cnpy::npy_save(filename, data.data(), dim, 2, "w"); \
        } \
        template <typename T> \
        void saveData( \
            const ConstArrayAccessor2<T>& data, \
            unsigned int frameNum) { \
            char filename[256]; \
            snprintf( \
                filename, \
                sizeof(filename), \
                "data.#grid2,%04d.npy", \
                frameNum); \
            saveData(data, filename); \
        } \
        template <typename T> \
        void saveData( \
            const ConstArrayAccessor3<T>& data, \
            const std::string& name) { \
            std::string filename = getFullFilePath(name); \
            unsigned int dim[3] = { \
                static_cast<unsigned int>(data.depth()), \
                static_cast<unsigned int>(data.height()), \
                static_cast<unsigned int>(data.width()) \
            }; \
            cnpy::npy_save(filename, data.data(), dim, 3, "w"); \
        } \
        template <typename T> \
        void saveData( \
            const ConstArrayAccessor3<T>& data, \
            unsigned int frameNum) { \
            char filename[256]; \
            snprintf( \
                filename, \
                sizeof(filename), \
                "data.#grid3,%04d.npy", \
                frameNum); \
            saveData(data, filename); \
        } \
        template <typename ParticleSystem> \
        void saveParticleDataXy( \
            const std::shared_ptr<ParticleSystem>& particles, \
            unsigned int frameNum) { \
            size_t n = particles->numberOfParticles(); \
            Array1<double> x(n); \
            Array1<double> y(n); \
            auto positions = particles->positions(); \
            for (size_t i = 0; i < n; ++i) { \
                x[i] = positions[i].x; \
                y[i] = positions[i].y; \
            } \
            char filename[256]; \
            snprintf( \
                filename, \
                sizeof(filename), \
                "data.#point2,%04d,x.npy", \
                frameNum); \
            saveData(x.constAccessor(), filename); \
            snprintf( \
                filename, \
                sizeof(filename), \
                "data.#point2,%04d,y.npy", \
                frameNum); \
            saveData(y.constAccessor(), filename); \
        } \
        void saveTriangleMeshData( \
            const TriangleMesh3& data, \
            const std::string& name) { \
            std::string filename = getFullFilePath(name); \
            std::ofstream file(filename.c_str()); \
            if (file) { \
                data.writeObj(&file); \
                file.close(); \
            } \
        } \
        std::string getTestDirectoryName(const std::string& name) { \
            return pystring::os::path::join(_testCollectionDir, name); \
        } \
        std::string getResourceFileName(const std::string& name) { \
            return pystring::os::path::join(RESOURCES_DIR, name); \
        } \
    }; \

#define JET_BEGIN_TEST_F(testSetName, testCaseNmae) \
    TEST_F(testSetName##Tests, testCaseNmae) { \
        createTestDirectory(#testCaseNmae);

#define JET_END_TEST_F }

#endif  // SRC_TESTS_MANUAL_TESTS_MANUAL_TESTS_H_
