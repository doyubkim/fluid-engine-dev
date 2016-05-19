This document explains how to build, test, and install the SDK.

## Building
To get started, clone the code from the repository and also download dependent libraries by running

```
git clone https://github.com/doyubkim/fluid-engine-dev.git
cd fluid-engine-dev.git
git submodule init
git submodule update
```

To build the code, a compiler that supports C++11 is required. Platform-specific build instructions are described below.

### Building from Mac OS X

Jet supports Mac OS X 10.10 or higher. Also, Xcode 6.4 or higher and the command line tools are required for building Jet. Once ready, install [Homebrew](http://brew.sh) and run

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

This will install GNU compilers, [Scons](http://scons.org/) and [zlib](www.zlib.net). Once installed, build the code by running

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

To build the code on Windows, Visual Studio 2015 is required. Free version of the tool can be downloaded from [Visual Studio Community 2015](https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx). In addition to Visual Studio, install [Python](https://www.python.org/) (2.7.9 or higher) to run post-build events.

Once installed, open the solution file `Jet.sln` using Visual Studio. Hit `Ctrl + Shift + B` to build the entire solution. Set `UnitTests` as a start-up project and hit `Ctrl + F5` to run the test. One built, the distributable files (`jet.lib` and the header files) will be located under `dist` directory.

## Running Tests

There are three different tests in the codebase including the unit test, manual test, and performance test.

### Unit Tests

The unit test contains the minimum validation tests for the code. To discover all the unit test cases from Mac OS X or Linux, run

```
bin/list_unit_tests
```

or for Windows, run

```
bin\list_unit_tests.bat
```

As described above, run the following command to launch entire tests from Mac OS X or Linux:

```
bin/unit_tests
```

For Windows, run

```
bin\unit_tests.bat
```

To run specific tests, such as `VertexCenteredVectorGrid3.Fill`, run

```
bin/unit_tests VertexCenteredVectorGrid3.Fill
```

You can also use a pattern to run the tests such as

```
bin/unit_tests VertexCenteredVectorGrid3.*
```

Replace `bin/unit_tests` with `bin\unit_tests.bat` for two commands above when running from Windows.

### Manual Tests

The manual test is the collection of the tests that can be verified manually. The test outputs data files to `manual_tests_output` so that the result can be rendered as images for the validation. To list the entire test cases from Mac OS X or Linux, run

```
bin/list_manual_tests
```

or for Windows, run

```
bin\list_manual_tests.bat
```

Similar to the unit test, run the following command to run entire tests for Mac OS X and Linux:

```
bin/manual_tests
```

For Windows, run

```
bin\manual_tests.bat
```

However, the manual test includes quite intensive tests such as running a short fluid simulation. Thus, it is recommended to run specific tests for the fast debugging, and then run the entire tests for final validation. Specifying the tests is the same as the unit test, such as.

```
bin/manual_tests Cubic*
```

Replace `bin/manual_tests` with the `.bat` command for Windows.


The output files will be located at `manual_tests_output/TestName/CaseName/file`. To validate the test results, you need [Matplotlib](http://matplotlib.org/). The recommended way of installing the latest version of the library is to use `pip` such as

```
pip install matplotlib
```

The modern Python versions (2.7.9 and above) comes with `pip` by default. Once Matplotlib is installed, run the following:


```
bin/render_manual_tests_output
```

This command requires [Matplotlib](http://matplotlib.org/). The recommended way of installing the latest version of the library is to use `pip` such as

```
pip install matplotlib
```

Once renderered, the rendered image will be stored at the same directory where the test output files are located (`manual_tests_output/TestName/CaseName/file`). Also, to render the animations as mpeg movie files, [ffmpeg](https://www.ffmpeg.org/) is required for Mac OS X and Windows. For Linux, [mencoder](http://www.mplayerhq.hu/) is needed. For Mac OS X, ffmpeg can be installed via Homebrew. For Windows, the executable can be downloaded from the [website](https://www.ffmpeg.org/). For Ubuntu, you can use `apt-get`.

### Performance Tests

The performance test measures the computation timing. To list the entire test cases, run

```
bin/list_perf_tests
```

To run the tests, execute

```
bin/perf_tests
```

Similar to the unit test and manual test, you can pass the test name pattern to run specific tests, such as

```
bin/perf_tests Point*
```

For Windows, use `bin\list_perf_tests.bat` and `bin\perf_tests.bat`.

The measured performance for each test will be printed to the console, such as

```
...
[----------] 1 test from PointHashGridSearcher3
[ RUN      ] PointHashGridSearcher3.Build
[----------] PointHashGridSearcher3::build avg. 0.261923 sec.
[       OK ] PointHashGridSearcher3.Build (2732 ms)
[----------] 1 test from PointHashGridSearcher3 (2733 ms total)
...
```

## Installing SDK

For Mac OS X and Ubuntu platforms, the library can be installed by running

```
[sudo] scons install --dist=INSTALL_PATH
```

This will install the header files and the static library `libjet.a` under `INSTALL_PATH`.

For Windows, the binaries and header files will be located under `dist` directory after building the solution.

## Continuous Integration

The build quality is tracked by [Travis CI](https://travis-ci.org/doyubkim/fluid-engine-dev) for Linux and Mac. For Windows, [AppVeyor](https://ci.appveyor.com/project/doyubkim/fluid-engine-dev) is used. Any pull requests must pass all the builds.
