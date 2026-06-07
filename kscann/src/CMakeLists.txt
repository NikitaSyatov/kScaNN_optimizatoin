include(CheckCXXCompilerFlag)

set(SOURCES
    kscann.cpp
    kmeans.cpp
    product_quantizer.cpp
    ivf_index.cpp
    simd_utils.cpp
    rair_utils.cpp
)

# NEON for ARM64
if(KSCANN_USE_NEON AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    list(APPEND SOURCES simd_neon.cpp)
endif()

# AVX2 for x86_64 – disabled for now (can be added later with a proper kernel)
# if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
#     check_cxx_compiler_flag("-mavx2" HAS_AVX2)
#     if(HAS_AVX2)
#         list(APPEND SOURCES simd_avx2.cpp)
#     endif()
# endif()

add_library(kscann_lib STATIC ${SOURCES})

target_compile_options(kscann_lib PRIVATE -O3 -march=native)

if(KSCANN_USE_NEON AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_definitions(kscann_lib PRIVATE KSCANN_USE_NEON=1)
endif()

target_include_directories(kscann_lib
    PUBLIC
        ${KSCANN_INCLUDE_DIR}
        ${KSCANN_BINARY_INCLUDE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)