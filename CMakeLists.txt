# CMake build control file for sparse matrix transpose testbench

cmake_minimum_required( VERSION 3.5 )

if (NOT DEFINED HPX_ROOT)
  message(FATAL_ERROR "please supply a path to the HPX install directory with -DHPX_ROOT=<some-path>")
endif()

find_package( Boost 1.63 REQUIRED COMPONENTS system program_options filesystem )

add_library( HPX::HPX SHARED IMPORTED )
set_target_properties( HPX::HPX PROPERTIES
  IMPORTED_LOCATION ${HPX_ROOT}/lib/libhpx.so
  INTERFACE_LINK_LIBRARIES Boost::filesystem
)

include_directories( SYSTEM ${HPX_ROOT}/include )

add_executable( transpose main.cpp nothing.cpp )

target_link_libraries( transpose HPX::HPX ${HPX_ROOT}/lib/libhpx_init.a Boost::system Boost::program_options )

set_target_properties( transpose PROPERTIES
  CXX_STANDARD 14
)

# HPX builds shared libraries that refer to Boost libs without specifying an RPATH, probably by design
# It seems we cannot work around this in our own RPATH, so we need to run like this:
# LD_LIBRARY_PATH=${BOOST_ROOT}/stage/lib ./transpose ...

# Release build by default
if( NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Default build type (Debug, Release, RelWithDebInfo)" FORCE )
endif()

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

