# CMake generated Testfile for 
# Source directory: C:/PROFESSIONAL/Development/AI-HPC/TernixEngine
# Build directory: C:/PROFESSIONAL/Development/AI-HPC/TernixEngine/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[CoreTests]=] "C:/PROFESSIONAL/Development/AI-HPC/TernixEngine/build/ternix_tests.exe")
set_tests_properties([=[CoreTests]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/PROFESSIONAL/Development/AI-HPC/TernixEngine/CMakeLists.txt;80;add_test;C:/PROFESSIONAL/Development/AI-HPC/TernixEngine/CMakeLists.txt;0;")
subdirs("_deps/benchmark-build")
subdirs("_deps/googletest-build")
subdirs("_deps/pybind11-build")
