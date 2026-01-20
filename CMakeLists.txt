cmake_minimum_required(VERSION 3.24)

project(knncolle_tatami
    VERSION 0.1.0
    DESCRIPTION "tatami bindings for the k-means library"
    LANGUAGES CXX)

add_library(knncolle_tatami INTERFACE)
add_library(knncolle::knncolle_tatami ALIAS knncolle_tatami)

include(GNUInstallDirs)
target_include_directories(knncolle_tatami INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/knncolle_knncolle_tatami>)

target_compile_features(knncolle_tatami INTERFACE cxx_std_17)

# Dependencies
option(KNNCOLLE_TATAMI_FETCH_EXTERN "Automatically fetch knncolle_tatami's external dependencies." ON)
if(KNNCOLLE_TATAMI_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(knncolle_knncolle 3.0.0 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.1.0 CONFIG REQUIRED)
    find_package(tatami_tatami 4.0.4 CONFIG REQUIRED)
endif()

target_link_libraries(knncolle_tatami INTERFACE knncolle::knncolle ltla::sanisizer tatami::tatami)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(KNNCOLLE_TATAMI_TESTS "Build knncolle_tatami's test suite." ON)
else()
    option(KNNCOLLE_TATAMI_TESTS "Build knncolle_tatami's test suite." OFF)
endif()
if(KNNCOLLE_TATAMI_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/knncolle_knncolle_tatami)

install(TARGETS knncolle_tatami
    EXPORT knncolle_tatamiTargets)

install(EXPORT knncolle_tatamiTargets
    FILE knncolle_knncolle_tatamiTargets.cmake
    NAMESPACE knncolle::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/knncolle_knncolle_tatami)

include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/knncolle_knncolle_tatamiConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/knncolle_knncolle_tatami)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/knncolle_knncolle_tatamiConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/knncolle_knncolle_tatamiConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/knncolle_knncolle_tatamiConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/knncolle_knncolle_tatami)
