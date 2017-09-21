# CMake build control file for sparse matrix transpose testbench

cmake_minimum_required( VERSION 3.5 )

find_package( Boost 1.63 REQUIRED COMPONENTS system program_options filesystem )

if (NOT DEFINED HPX_ROOT)
  message(FATAL_ERROR "please supply a path to the HPX install directory with -DHPX_ROOT=<some-path>")
endif()

add_library( HPX::HPX SHARED IMPORTED )
set_target_properties( HPX::HPX PROPERTIES
  IMPORTED_LOCATION ${HPX_ROOT}/lib/libhpx.so
  INTERFACE_LINK_LIBRARIES Boost::filesystem
  INTERFACE_INCLUDE_DIRECTORIES ${HPX_ROOT}/include
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${HPX_ROOT}/include
)

# HPX builds shared libraries that refer to Boost libs without specifying an RPATH, probably by design
# It seems we cannot work around this in our own RPATH, so we need to run like this:
# LD_LIBRARY_PATH=${BOOST_ROOT}/stage/lib ./transpose ...

# Release build by default
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Default build type (Debug, Release, RelWithDebInfo)" FORCE )
endif()

# download and use Google Benchmark
include( ExternalProject )
ExternalProject_Add( GBENCH
  GIT_REPOSITORY https://github.com/google/benchmark
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gbench
  CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
  UPDATE_COMMAND ""
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property( GBENCH SOURCE_DIR BINARY_DIR )
add_library( gbench STATIC IMPORTED )

# create include directory, which will be absent at initial run
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ${SOURCE_DIR}/include )
set_target_properties( gbench PROPERTIES
  IMPORTED_LOCATION "${BINARY_DIR}/src/libbenchmark.a"
  INTERFACE_INCLUDE_DIRECTORIES "${SOURCE_DIR}/include"
  INTERFACE_LINK_LIBRARIES pthread
)
add_dependencies( gbench GBENCH )

add_executable( transpose main.cpp )

target_link_libraries( transpose gbench HPX::HPX ${HPX_ROOT}/lib/libhpx_init.a Boost::system Boost::program_options )

set_target_properties( transpose PROPERTIES
  CXX_STANDARD 14
)

# Create a target for running transpose on different size matrices
add_custom_command(
  OUTPUT nnz10.txt nnz100.txt
  COMMAND echo "# 10 nonzeros per row" > nnz10.txt
  COMMAND echo "# size   map->map  map->flatmap  mat->csr     flatmap->map  flatmap->flatmap  flatmap->csr      csr->map  csr->flatmap  csr->csr" >> nnz10.txt
  COMMAND transpose 1000 10 >> nnz10.txt
  COMMAND transpose 3200 10 >> nnz10.txt
  COMMAND transpose 10000 10 >> nnz10.txt
  COMMAND transpose 32000 10 >> nnz10.txt
  COMMAND transpose 100000 10 >> nnz10.txt
  COMMAND transpose 320000 10 >> nnz10.txt
  COMMAND transpose 1000000 10 >> nnz10.txt
  
  COMMAND echo "# 100 nonzeros per row" > nnz100.txt
  COMMAND echo "# size   map->map  map->flatmap  mat->csr     flatmap->map  flatmap->flatmap  flatmap->csr      csr->map  csr->flatmap  csr->csr" >> nnz100.txt
  COMMAND transpose 1000 100 >> nnz100.txt
  COMMAND transpose 3200 100 >> nnz100.txt
  COMMAND transpose 10000 100 >> nnz100.txt
  COMMAND transpose 32000 100 >> nnz100.txt
  COMMAND transpose 100000 100 >> nnz100.txt
  COMMAND transpose 320000 100 >> nnz100.txt
)

add_custom_target( benchmark DEPENDS nnz10.txt nnz100.txt )

