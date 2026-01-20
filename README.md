# tatami bindings for knncolle

![Unit tests](https://github.com/knncolle/knncolle_tatami/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/knncolle/knncolle_tatami/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/knncolle/knncolle_tatami/branch/master/graph/badge.svg?token=7S231XHC0Q)](https://codecov.io/gh/knncolle/knncolle_tatami)

## Overview

This library implements a wrapper class to use [`tatami::Matrix`](https://github.com/tatami-inc/tatami) instances in the [**knncolle** library](https://github.com/knncolle/knncolle).
The goal is to support k-means clustering from alternative matrix representations (e.g., sparse, file-backed) without requiring realization into a `knncolle::SimpleMatrix`.

## Quick start

Not much to say, really. 
Just replace the usual `knncolle::SimpleMatrix` with an instance of a `knncolle_tatami::Matrix`.

```cpp
#include "knncolle_tatami/knncolle_tatami.hpp"

// Initialize this with an instance of a concrete tatami subclass.
std::shared_ptr<tatami::Matrix<double, int> > tmat;

knncolle_tatami::Matrix<int, double, double, int> wrapper(std::move(tmat));
knncolle::VptreeBuilder<int, double, double> builder;
auto index = builder.build_shared(wrapper);
auto res = knncolle::find_nearest_neighbors(*index, 5);
```

See the [reference documentation](https://knncolle.github.io/knncolle_tatami) for more details.

## Building projects 

### CMake with `FetchContent`

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  knncolle_tatami
  GIT_REPOSITORY https://github.com/knncolle/knncolle_tatami
  GIT_TAG master # or any version of interest
)

FetchContent_MakeAvailable(knncolle_tatami)
```

Then you can link to **knncolle_tatami** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe knncolle::knncolle_tatami)

# For libaries
target_link_libraries(mylib INTERFACE knncolle::knncolle_tatami)
```

By default, this will use `FetchContent` to fetch all external dependencies. 
Applications are advised to pin the versions of each dependency for stability - see [`extern/CMakeLists.txt`](extern/CMakeLists.txt) for suggested versions.
If you want to install them manually, use `-DKNNCOLLE_TATAMI_FETCH_EXTERN=OFF`.

### CMake with `find_package()`

To install the library, clone an appropriate version of this repository and run:

```sh
mkdir build && cd build
cmake .. -DKNNCOLLE_TATAMI_TESTS=OFF
cmake --build . --target install
```

Then we can use `find_package()` as usual:

```cmake
find_package(knncolle_knncolle_tatami CONFIG REQUIRED)
target_link_libraries(mylib INTERFACE knncolle::knncolle_tatami)
```

Again, this will automatically acquire all its dependencies, see recommendations above.

### Manual

If you're not using CMake, the simple approach is to just copy the files in `include/` - either directly or with Git submodules - and include their path during compilation with, e.g., GCC's `-I`.
This requires the external dependencies listed in [`extern/CMakeLists.txt`](extern/CMakeLists.txt). 
