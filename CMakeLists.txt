cmake_minimum_required(VERSION 3.24)
project(git_sha1_cuda VERSION 0.3.0 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(GSV_SOURCE src/git_sha1_cuda.cu)

add_library(git_sha1_cuda SHARED ${GSV_SOURCE})
target_include_directories(git_sha1_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(git_sha1_cuda PUBLIC cxx_std_17 cuda_std_17)
target_compile_options(git_sha1_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
target_compile_definitions(git_sha1_cuda PRIVATE GSV_BUILD_SHARED)
set_target_properties(git_sha1_cuda PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON)

option(GSV_BUILD_STATIC "Build libgit_sha1_cuda_static.a" ON)
if(GSV_BUILD_STATIC)
  add_library(git_sha1_cuda_static STATIC ${GSV_SOURCE})
  target_include_directories(git_sha1_cuda_static
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>)
  target_compile_features(git_sha1_cuda_static PUBLIC cxx_std_17 cuda_std_17)
  target_compile_options(git_sha1_cuda_static PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
  set_target_properties(git_sha1_cuda_static PROPERTIES
    OUTPUT_NAME git_sha1_cuda_static
    POSITION_INDEPENDENT_CODE ON)
endif()

include(GNUInstallDirs)
install(TARGETS git_sha1_cuda
  EXPORT git_sha1_cudaTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(GSV_BUILD_STATIC)
  install(TARGETS git_sha1_cuda_static
    EXPORT git_sha1_cudaTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
install(FILES include/git_sha1_cuda.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/git_sha1_cudaConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/git_sha1_cudaConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/git_sha1_cuda)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/git_sha1_cudaConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(EXPORT git_sha1_cudaTargets
  FILE git_sha1_cudaTargets.cmake
  NAMESPACE git_sha1_cuda::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/git_sha1_cuda)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/git_sha1_cudaConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/git_sha1_cudaConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/git_sha1_cuda)

include(CTest)
if(BUILD_TESTING)
  add_executable(gsv_c_api_smoke tests/c_api_smoke.cpp)
  target_compile_features(gsv_c_api_smoke PRIVATE cxx_std_17)
  target_link_libraries(gsv_c_api_smoke PRIVATE git_sha1_cuda)
  add_test(NAME gsv_c_api_smoke COMMAND gsv_c_api_smoke)
  set_tests_properties(gsv_c_api_smoke PROPERTIES SKIP_RETURN_CODE 77)
endif()
