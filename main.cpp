

#include <map>
#include <vector>
#include <iostream>
#include <algorithm>

#include "from_csr.hpp"
#include "from_map.hpp"

#include <hpx/hpx_init.hpp>

#include <boost/interprocess/containers/flat_map.hpp>

#include <benchmark/benchmark.h>

int main(int argc, char* argv[])
{
    // process and remove gbench arguments
    benchmark::Initialize(&argc, argv);

    // By default this should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    return hpx::init(argc, argv, cfg);
}

int hpx_main(int argc, char **argv)
{

  if (argc < 3)
  {
    std::cout << "Missing argument: Matrix-dimension, Nonzeros/row" << std::endl;
    return EXIT_FAILURE;
  }

  std::size_t N = atoi(argv[1]);
  std::size_t nnz_per_row = atoi(argv[2]);

  //
  // Matrix initialization
  //
  std::vector< std::map<unsigned int, double> > A_map(N);
  std::vector< boost::container::flat_map<unsigned int, double> > A_flatmap(N);
  std::vector< unsigned int > A_csr_rows(N+1);

  unsigned int total_nnz = 0;
  for (std::size_t row=0; row<N; ++row)
  {
    for (std::size_t j=0; j<nnz_per_row; ++j)
    {
      unsigned int col_idx = rand() % N;
      double value = 1.0 + double(rand()) / double(RAND_MAX);
      A_map[row][col_idx] = value;
      A_flatmap[row][col_idx] = value;
    }
    A_csr_rows[row] = total_nnz;
    total_nnz += A_map[row].size();
  }
  A_csr_rows[N] = total_nnz;

  std::vector<unsigned int> A_csr_cols(total_nnz);
  std::vector<double> A_csr_values(total_nnz);

  std::size_t index = 0;
  for (std::size_t row=0; row<N; ++row)
    for (std::map<unsigned int, double>::const_iterator it = A_map[row].begin(); it != A_map[row].end(); ++it, ++index)
    {
      A_csr_cols[index] = it->first;
      A_csr_values[index] = it->second;
    }

  // define and register benchmarks
  benchmark::RegisterBenchmark(
    "Map2Map",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<std::map<unsigned int, double> > B;
        run(A_map, B);
      }
    });

  benchmark::RegisterBenchmark(
    "Map2FlatMap",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<boost::container::flat_map<unsigned int, double> > B(N);
        for (std::size_t j=0; j<N; ++j)
          B[j].reserve(2*nnz_per_row);
        run(A_map, B);
      }
    });

  benchmark::RegisterBenchmark(
    "Map2CSR",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<unsigned int> B_rows;
        std::vector<unsigned int> B_cols;
        std::vector<double> B_values;
        run(A_map, B_rows, B_cols, B_values);
      }
    });

  benchmark::RegisterBenchmark(
    "FlatMap2Map",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<std::map<unsigned int, double> > B;
        run(A_flatmap, B);
      }
    });

  benchmark::RegisterBenchmark(
    "FlatMap2FlatMap",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<boost::container::flat_map<unsigned int, double> > B(N);
        for (std::size_t j=0; j<N; ++j)
          B[j].reserve(2*nnz_per_row);
        run(A_flatmap, B);
      }
    });

  benchmark::RegisterBenchmark(
    "FlatMap2CSR",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<unsigned int> B_rows;
        std::vector<unsigned int> B_cols;
        std::vector<double> B_values;
        run(A_flatmap, B_rows, B_cols, B_values);
      }
    });

  benchmark::RegisterBenchmark(
    "CSR2Map",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<std::map<unsigned int, double> > B;
        run(A_csr_rows, A_csr_cols, A_csr_values, B);
      }
    });

  benchmark::RegisterBenchmark(
    "CSR2FlatMap",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<boost::container::flat_map<unsigned int, double> > B(N);
        for (std::size_t j=0; j<N; ++j)
          B[j].reserve(2*nnz_per_row);
        run(A_csr_rows, A_csr_cols, A_csr_values, B);
      }
    });

  benchmark::RegisterBenchmark(
    "CSR2CSR",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<unsigned int> B_rows;
        std::vector<unsigned int> B_cols;
        std::vector<double> B_values;
        run(A_csr_rows, A_csr_cols, A_csr_values, B_rows, B_cols, B_values);
      }
    });

  benchmark::RegisterBenchmark(
    "CSR2CSR-HPX-SORT",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<unsigned int> B_rows;
        std::vector<unsigned int> B_cols;
        std::vector<double> B_values;
        run_hpx(A_csr_rows, A_csr_cols, A_csr_values, B_rows, B_cols, B_values);
      }
    });

  benchmark::RegisterBenchmark(
    "CSR2CSR-HPX-MERGE",
    [=](benchmark::State & state) {
      while (state.KeepRunning()) {
        std::vector<unsigned int> B_rows;
        std::vector<unsigned int> B_cols;
        std::vector<double> B_values;
        run_hpx_merge(A_csr_rows, A_csr_cols, A_csr_values, B_rows, B_cols, B_values);
      }
    });

  //
  // Run benchmarks
  //

  benchmark::RunSpecifiedBenchmarks();

  return hpx::finalize();
}
