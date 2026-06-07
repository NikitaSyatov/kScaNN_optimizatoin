cmake_minimum_required(VERSION 3.15)
project(kscann VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(KSCANN_USE_NEON "Enable ARM NEON SIMD optimisations" ON)
option(KSCANN_BUILD_TESTS "Build unit tests" ON)
option(KSCANN_BUILD_EXAMPLES "Build example programs" ON)

# Configuration header
configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

# Include directories for the whole project
set(KSCANN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(KSCANN_BINARY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Library source subdirectory
add_subdirectory(src)

# Tests and examples
if(KSCANN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(KSCANN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install targets (optional)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS kscann_lib DESTINATION lib)
