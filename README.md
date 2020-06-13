# Fluid Engine Dev - Jet

[![License](http://img.shields.io/:license-mit-blue.svg)](LICENSE.md) [![Build Status](https://travis-ci.org/doyubkim/fluid-engine-dev.svg?branch=main)](https://travis-ci.org/doyubkim/fluid-engine-dev/branches) [![Build status](https://ci.appveyor.com/api/projects/status/kulihlhy43vbwou6/branch/main?svg=true)](https://ci.appveyor.com/project/doyubkim/fluid-engine-dev/branch/main) [![codecov](https://codecov.io/gh/doyubkim/fluid-engine-dev/branch/main/graph/badge.svg)](https://codecov.io/gh/doyubkim/fluid-engine-dev)

Jet framework is a fluid simulation engine SDK for computer graphics applications that was created by Doyub Kim as part of the book, ["Fluid Engine Development"](https://www.crcpress.com/Fluid-Engine-Development/Kim/p/book/9781498719926). The code is built on C++11 and can be compiled with most of the commonly available compilers such as g++, clang++, or Microsoft Visual Studio. Jet currently supports macOS (10.10 or later), Ubuntu (14.04 or later), and Windows (Visual Studio 2015 or later). Other untested platforms that support C++11 also should be able to build Jet. The framework also provides Python API for faster prototyping.

The latest code is always available from the [`main`](https://github.com/doyubkim/fluid-engine-dev/tree/main) branch. Since the code evolves over time, the latest from the main branch could be somewhat different from the code in the book. To find the version that is consistent with the book, check out the branch [`book-1st-edition`](https://github.com/doyubkim/fluid-engine-dev/tree/book-1st-edition).

## Key Features
* Basic math and geometry operations and data structures
* Spatial query accelerators
* SPH and PCISPH fluid simulators
* Stable fluids-based smoke simulator
* Level set-based liquid simulator
* PIC, FLIP, and APIC fluid simulators
* Upwind, ENO, and FMM level set solvers
* Jacobi, Gauss-Seidel, SOR, MG, CG, ICCG, and MGPCG linear system solvers
* Spherical, SPH, Zhu & Bridson, and Anisotropic kernel for points-to-surface converter
* Converters between signed distance function and triangular mesh
* C++ and Python API
* Intel TBB, OpenMP, and C++11 multi-threading backends

Every simulator has both 2-D and 3-D implementations.

## Quick Start

You will need CMake to build the code. If you're using Windows, you need Visual Studio 2015 or 2017 in addition to CMake.

First, clone the code:

```
git clone https://github.com/doyubkim/fluid-engine-dev.git --recursive
cd fluid-engine-dev
```

### Python API

Build and install the package by running

```
pip install -U .
```

Now run some examples, such as:

```
python src/examples/python_examples/smoke_example01.py
```

### C++ API

For macOS or Linux:

```
mkdir build && cd build && cmake .. && make
```

For Windows:

```
mkdir build
cd build
cmake .. -G"Visual Studio 14 2015 Win64"
MSBuild jet.sln /p:Configuration=Release
```

Now run some examples, such as:

```
bin/hybrid_liquid_sim
```

### Docker

```
docker pull doyubkim/fluid-engine-dev:latest
```

Now run some examples, such as:

```
docker run -it doyubkim/fluid-engine-dev
[inside docker container]
/app/build/bin/hybrid_liquid_sim
```


### More Instructions of Building the Code

To learn how to build, test, and install the SDK, please check out [INSTALL.md](https://github.com/doyubkim/fluid-engine-dev/blob/main/INSTALL.md).

## Documentations

All the documentations for the framework can be found from [the project website](http://fluidenginedevelopment.org/documentation/) including the API reference.

## Examples

Here are some of the example simulations generated using Jet framework. Corresponding example codes can be found under src/examples. All images are rendered using [Mitsuba renderer](https://www.mitsuba-renderer.org/) and the Mitsuba scene files can be found from [the demo repository](https://github.com/doyubkim/fluid-engine-dev-demo/). Find out more demos from [the project website](http://fluidenginedevelopment.org/examples/).

### FLIP Simulation Example

![FLIP Example](https://github.com/doyubkim/fluid-engine-dev/raw/main/doc/img/flip_dam_breaking.png "FLIP Example")

### PIC Simulation Example

![PIC Example](https://github.com/doyubkim/fluid-engine-dev/raw/main/doc/img/pic_dam_breaking.png "PIC Example")

### Level Set Example with Different Viscosity

![Level Set Example](https://github.com/doyubkim/fluid-engine-dev/raw/main/doc/img/ls_bunny_drop.png "Level Set Example ")

### Smoke Simulation with Different Advection Methods

![Cubic-smoke Example](https://github.com/doyubkim/fluid-engine-dev/raw/main/doc/img/smoke_cubic.png "Cubic-smoke Example")
![Linear-smoke Example](https://github.com/doyubkim/fluid-engine-dev/raw/main/doc/img/smoke_linear.png "Linear-smoke Example")

### Point-to-Surface Examples

![Point-to-Surface Example](https://github.com/doyubkim/fluid-engine-dev/raw/main/doc/img/point_to_surface.png "Point-to-Surface Example")

> Top-left: spherical, top-right: SPH blobby, bottom-left: Zhu and Bridson's method, and bottom-right: Anisotropic kernel

## Developers

This repository is created and maintained by Doyub Kim (@doyubkim). Chris Ohk (@utilForever) is a co-developer of the framework since v1.3. [Many other contributors](https://github.com/doyubkim/fluid-engine-dev/graphs/contributors) also helped improving the codebase including Jefferson Amstutz (@jeffamstutz) who helped integrating Intel TBB and OpenMP backends.

## License

Jet is under the MIT license. For more information, check out [LICENSE.md](https://github.com/doyubkim/fluid-engine-dev/blob/main/LICENSE.md). Jet also utilizes other open source codes. Checkout [3RD_PARTY.md](https://github.com/doyubkim/fluid-engine-dev/blob/main/3RD_PARTY.md) for more details.

I am making my contributions/submissions to this project solely in my personal capacity and am not conveying any rights to any intellectual property of any third parties.

## Acknowledgement

We would like to thank [JetBrains](https://www.jetbrains.com/) for their support and allowing us to use their products for developing Jet Framework.

![JetBrains](doc/img/jetbrains.svg)
