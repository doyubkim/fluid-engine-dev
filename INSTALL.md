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

Jet supports OS X 10.10 Yosemite or higher. Also, Xcode 6.4 or higher and the command line tools are required for building Jet. Once ready, install [Homebrew](http://brew.sh) and run

```
brew install scons
```

This will install [Scons](http://scons.org/). Once installed, build the code by running

```
scons
```

This will build entire codebase. To run the unit test, execute

```
bin/unit_tests
```

The library can be installed by running

```
[sudo] scons install --dist=INSTALL_PATH
```

This will install the header files and the static library `libjet.a` under `INSTALL_PATH`.

### Building from Ubuntu

Jet supports Ubuntu 14.04 or higher. Using `apt-get`, install required tools and libraries by running,

```
sudo apt-get install build-essential python scons zlib1g-dev
```

This will install GNU compilers, pytho, [Scons](http://scons.org/) (the build tool), and [zlib](www.zlib.net). Once installed, build the code by running

```
scons
```

This will build entire codebase. To run the unit test, execute

```
bin/unit_tests
```

The library can be installed by running

```
[sudo] scons install --dist=INSTALL_PATH
```

This will install the header files and the static library `libjet.a` under `INSTALL_PATH`.

### Building from Windows

To build the code on Windows, Visual Studio 2015 is required. Free version of the tool can be downloaded from [Visual Studio Community 2015](https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx). In addition to Visual Studio, install [Python](https://www.python.org/) (2.7.9 or higher recommended) to run post-build events.

Once Visual Studio is installed, open the solution file `Jet.sln` using Visual Studio. Hit `Ctrl + Shift + B` to build the entire solution. Set `UnitTests` as a start-up project and hit `Ctrl + F5` to run the test. Once built, the distributable files (`jet.lib` and the header files) will be located under `dist` directory.

### Running Tests

There are three different tests in the codebase including the unit test, manual test, and performance test. For the detailed instruction on how to run those tests, please checkout the documentation page from [the project website](http://doyubkim.github.io/fluid-engine-dev/documentation/).

### Installing SDK

For macOS and Ubuntu platforms, the library can be installed by running

```
[sudo] scons install --dist=INSTALL_PATH
```

This will install the header files and the static library `libjet.a` under `INSTALL_PATH`.

For Windows, the binaries and header files will be located under `dist` directory after building the solution.

### Coding Style

Jet uses a modified version of [cpplint.py](https://github.com/google/styleguide/tree/gh-pages/cpplint) for checking the coding style. Please check out [3RD_PARTY.md](https://github.com/doyubkim/fluid-engine-dev/blob/master/3RD_PARTY.md) for the license of `cpplint.py`. Any pull requests must pass the linter. Use `bin/run_linters` (or `bin\run_linters.bat` for Windows) to test the header and source files of the library.

### Continuous Integration

The build quality is tracked by [Travis CI](https://travis-ci.org/doyubkim/fluid-engine-dev) for Linux and Mac. For Windows, [AppVeyor](https://ci.appveyor.com/project/doyubkim/fluid-engine-dev) is used. Any pull requests must pass all the builds.
