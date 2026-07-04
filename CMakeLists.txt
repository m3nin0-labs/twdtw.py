cmake_minimum_required(VERSION 3.15...3.30)
project(twdtw LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Locate the active Python and nanobind (installed as a build dependency)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(corecpp STABLE_ABI twdtw/core/twdtw.cpp)
target_compile_features(corecpp PRIVATE cxx_std_17)

# Land the compiled module inside the importable `twdtw` package.
install(TARGETS corecpp LIBRARY DESTINATION twdtw)
