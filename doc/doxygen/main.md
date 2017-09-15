Jet framework is a fluid simulation engine SDK for computer graphics applications that was created by Doyub Kim as part of the book, ["Fluid Engine Development"](https://www.crcpress.com/Fluid-Engine-Development/Kim/p/book/9781498719926). The code is built on C++11 and can be compiled with most of the commonly available compilers such as g++, clang++, or Microsoft Visual Studio. Jet currently supports macOS (10.10 or later), Ubuntu (14.04 or later), and Windows (Visual Studio 2015 or later). Other untested platforms that support C++11 also should be able to build Jet. The framework also provides Python API for faster prototyping.

### Key Features
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

Every simulator has both 2-D and 3-D implementations.
