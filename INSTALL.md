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
brew install cmake
```

Once CMake is installed, build the code by running

```
mkdir build
cd build
cmake ..
make
```

This will build entire codebase. To run the unit test, execute

```
bin/unit_tests
```

### Building from Ubuntu

Jet supports Ubuntu 14.04 or higher. Using `apt-get`, install required tools and libraries by running,

```
sudo apt-get install build-essential python-dev cmake
```

This will install GNU compilers, python, and CMake. Once installed, build the code by running

```
mkdir build
cd build
cmake ..
make
```

This will build entire codebase. To run the unit test, execute

```
bin/unit_tests
```

### Building from Windows

To build the code on Windows, CMake, Python, and Visual Studio 2015 (or higher) is required. Windows' version of CMake is available from [this website](https://cmake.org/), Python installer can be downloaded from [here](https://python.org/). For Python, version 2.7.9 or later is recommended. To install Visual Studio, the community edition of the tool can be downloaded from [Visual Studio Community 2015](https://www.Visualstudio.com/en-us/products/Visual-studio-community-vs.aspx). You can also use Visual Studio 2017.

Once everything is installed, run the following commands:

```
md build
cd build
cmake .. -G"Visual Studio 14 2015 Win64"
```

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

There are three different tests in the codebase including the unit test, manual test, and performance test. For the detailed instruction on how to run those tests, please checkout the documentation page from [the project website](http://doyubkim.github.io/fluid-engine-dev/documentation/).

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
pip install .
```

and that's it!

### Coding Style

Jet uses clang-format. Checkout [`.clang-format`](https://github.com/doyubkim/fluid-engine-dev/blob/master/.clang-format) file for the style guideline.

### Continuous Integration

The build quality is tracked by [Travis CI](https://travis-ci.org/doyubkim/fluid-engine-dev) for Linux and Mac. For Windows, [AppVeyor](https://ci.appveyor.com/project/doyubkim/fluid-engine-dev) is used. Any pull requests must pass all the builds.
