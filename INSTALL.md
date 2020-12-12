This document explains how to build, test, and install the SDK.

## Building the Code

To get started, clone the code from the repository and also download dependent libraries by running

```
git clone https://github.com/doyubkim/fluid-engine-dev.git
cd fluid-engine-dev.git
git submodule init
git submodule update
```

To build the code, a compiler that supports C++11 is required. Platform-specific build instructions are described below.

### Building from macOS

Jet supports OS X 10.10 Yosemite or higher. Also, Xcode 6.4 or higher and the command line tools are required for building Jet. Once ready, install [Homebrew](http://brew.sh) and run the following command line to setup [CMake](https://cmake.org/):

```
brew install cmake python
```

> Note that we want `brew` version of Python which is recommended. You can still use macOS's default Python.

Optionally, which is recommended, you can [Intel TBB](https://www.threadingbuildingblocks.org) for multithreading backend:

```
brew install tbb
```

Once CMake and Python is installed, build the code by running

```
mkdir build
cd build
cmake ..
make
```

> Of course, use `make -j<num_threads>` flag to boost up the build performance by using multithreads. Also, pass `-DJET_TASKING_SYSTEM=TBB` or `-DJET_TASKING_SYSTEM=CPP11Threads` to the `cmake` command in order to explicitly enable either Intel TBB or C++11 thread-based multithreading backend. If not specified, the build script will try to use Intel TBB first. If not found, it will fall back to C++11 threads.

This will build entire codebase. To run the unit test, execute

```
bin/unit_tests
```

It should show all the tests are passing.

### Building from Ubuntu

Jet supports Ubuntu 14.04 or higher. Using `apt-get`, install required tools and libraries by running,

```
sudo apt-get install build-essential python-dev python-pip cmake
```

Optionally, which is recommended, you can [Intel TBB](https://www.threadingbuildingblocks.org) for multithreading backend:

```
sudo apt-get install libtbb-dev
```

This will install GNU compilers, python, and CMake. Once installed, build the code by running

```
mkdir build
cd build
cmake ..
make
```

> Again, use `make -j<num_threads>` flag to boost up the build performance by using multithreads. Also, pass `-DJET_TASKING_SYSTEM=TBB`, `-DJET_TASKING_SYSTEM=OpenMP` or `-DJET_TASKING_SYSTEM=CPP11Threads` to the `cmake` command in order to explicitly enable either Intel TBB , OpenMP, or C++11 thread-based multithreading backend. If not specified, the build script will try to use Intel TBB first. If not found, it will fall back to OpenMP and then C++11 threads.

This will build entire codebase. To run the unit test, execute

```
bin/unit_tests
```

It should show all the tests are passing.

### Building from Windows

To build the code on Windows, CMake, Python, and Visual Studio 2015 (or higher) is required. Windows' version of CMake is available from [this website](https://cmake.org/), Python installer can be downloaded from [here](https://python.org/). For Python, version 2.7.9 or later is recommended. To install Visual Studio, the community edition of the tool can be downloaded from [Visual Studio Community 2015](https://www.Visualstudio.com/en-us/products/Visual-studio-community-vs.aspx). You can also use Visual Studio 2017.

Once everything is installed, run the following commands:

```
md build
cd build
cmake .. -G"Visual Studio 14 2015 Win64"
```
> Again, pass `-DJET_TASKING_SYSTEM=TBB`, `-DJET_TASKING_SYSTEM=OpenMP` or `-DJET_TASKING_SYSTEM=CPP11Threads` to the `cmake` command in order to explicitly enable either Intel TBB , OpenMP, or C++11 thread-based multithreading backend. If not specified, the build script will try to use Intel TBB first. If not found, it will fall back to OpenMP and then C++11 threads.

This will generate 64-bit version of VS 2015 solution and projects. (To build with Visual Studio 2017, just replace the parameter with `Visual Studio 15 2017 Win64`.) Once executed, you can find `jet.sln` solution file in the `build` directory. Open the solution file and hit `Ctrl + Shift + B` to build the entire solution. Set `unit_tests` as a start-up project and hit `Ctrl + F5` to run the test.

Alternatively, you can use MSBuild to build the solution from the command prompt. In such case, simply run:

```
MSBuild jet.sln /p:Configuration=Release
```

This will build the whole solution in release mode. Once built, run the following command to execute unit tests:

```
bin\Release\unit_tests.exe
```

### Running Tests

There are four different tests in the codebase including the unit test, manual test, time/memory performance tests, and Python API test. For the detailed instruction on how to run those tests, please checkout the documentation page from [the project website](http://doyubkim.github.io/fluid-engine-dev/documentation/).

### Code Coverage

Jet uses ```lcov``` for the code coverage. For macOS and Ubuntu platforms, the code coverage report can be generated by running

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j 8
lcov -c -i -d src/tests/unit_tests -o base.info
bin/unit_tests
lcov -c -d src/tests/unit_tests -o test.info
lcov -a base.info -a test.info -o coverage.info
lcov -r coverage.info '/usr/*' -o coverage.info
lcov -r coverage.info '*/external/*' -o coverage.info
lcov -r coverage.info '*/src/tests/*' -o coverage.info
lcov -l coverage.info
genhtml coverage.info -o out
```

This will exports the code coverage report ```index.html``` under `out` folder.

### Installing C++ SDK

For macOS and Ubuntu platforms, the library can be installed by running

```
cmake .. -DCMAKE_INSTALL_PREFIX=_INSTALL_PATH_
make
make install
```

This will install the header files and the static library `libjet.a` under `_INSTALL_PATH_`.

For Windows, run:

```
cmake .. -G"Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=_INSTALL_PATH_
```

Then, build `INSTALL` project under `jet.sln`. This will install the header files and the static library `jet.lib` under `_INSTALL_PATH_`.

### Installing Python SDK

To install the Python SDK, `pyjet`, run the following command from the project root directory (where `setup.py` lives):

```
pip install -U .
```

> You can also use `virtualenv` to isolate the SDK installation. Check out [the virtualenv documentation](https://virtualenv.pypa.io/en/stable/) for more details.

To run the test/example scripts, install other Python dependencies as follows:

```
pip install -r requirements.txt
```

Once installed, try running the unit test to see if the module is installed correctly:

```
pytest src/tests/python_tests
```

The tests should pass.

### Using Docker

You can also use pre-built docker image by pulling the latest version from Docker Hub:

```
docker pull doyubkim/fluid-engine-dev
```

Run a container and see if it can import `pyjet` module and the unit test passes:

```
docker run -it doyubkim/fluid-engine-dev
python import -c "pyjet"

docker run doyubkim/fluid-engine-dev /app/build/bin/unit_tests
```

You can also build the image from the source as well. From the root directory of this codebase, run:

```
docker build -t doyubkim/fluid-engine-dev .
```

> Warning: When you run Python examples using Intel TBB from Windows, you might encounter the following error:
```
import pyjet
ImportError: DLL load failed while importing pyjet:
The specified module could not be found. 
```

> It is a new Windows safety feature that changes how DLLs are loaded in Python 3.8+. To resolve this issue, I needed to tell Python how to find the library again like this:
```
import os
os.add_dll_directory(r'C:/Intel/tbb/bin/intel64/vc14') << (The path that Intel TBB is located)
import pyjet
```

> Open example file and add the code above to the first line. Now, you won't have any problems running it.

### Coding Style

Jet uses clang-format. Checkout [`.clang-format`](https://github.com/doyubkim/fluid-engine-dev/blob/main/.clang-format) file for the style guideline.

### Continuous Integration

The build quality is tracked by [Travis CI](https://travis-ci.org/doyubkim/fluid-engine-dev) for Linux and Mac. For Windows, [AppVeyor](https://ci.appveyor.com/project/doyubkim/fluid-engine-dev) is used. Any pull requests must pass all the builds.
