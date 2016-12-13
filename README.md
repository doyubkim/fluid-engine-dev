# Fluid Engine Dev - Jet

[![License](http://img.shields.io/:license-mit-blue.svg)](LICENSE.md) [![Build Status](https://travis-ci.org/doyubkim/fluid-engine-dev.svg?branch=master)](https://travis-ci.org/doyubkim/fluid-engine-dev) [![Build status](https://ci.appveyor.com/api/projects/status/kulihlhy43vbwou6/branch/master?svg=true)](https://ci.appveyor.com/project/doyubkim/fluid-engine-dev/branch/master)

This project was created by Doyub Kim as part of the book, ["Fluid Engine Development"](https://www.crcpress.com/Fluid-Engine-Development/Kim/p/book/9781498719926), and Jet is the fluid simulation engine SDK introduced from the book.

## Key Features
* SPH and PCISPH fluid simulators
* Stable fluids-based smoke simulator
* Level set-based liquid simulator
* PIC and FLIP fluid simulators
* Upwind, ENO and FMM level set solvers
* Converters between signed distance function and triangular mesh

Every simulator has both 2-D and 3-D implementations.

## How to Build

To learn how to build, test, and install the SDK, please check out [INSTALL.md](INSTALL.md).

## Examples

Here are some of the example simulations generated using Jet framework. Corresponding example codes can be found under src/examples. All images are rendered using [Mitsuba renderer](https://www.mitsuba-renderer.org/).

![Examples](doc/img/examples.png "Examples")

## License

Jet is under the MIT license. For more information, check out [LICENSE.md](LICENSE.md). Jet also utilizes other open source codes. Checkout [3RD_PARTY.md](3RD_PARTY.md) for more details.
