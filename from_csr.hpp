#include <benchmark/benchmark.h>

#include <vector>
#include <map>

#include <hpx/hpx.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/zip_iterator.hpp>
#include <hpx/util/tuple.hpp>

//
// CSR to map
//
template<typename IndexT, typename NumericT, typename RowT>
void run(std::vector<IndexT> const & A_rows, std::vector<IndexT> const & A_cols, std::vector<NumericT> const & A_values,
         std::vector<RowT> & B)
{
  std::size_t N = A_rows.size() - 1;
  B.resize(N);

  IndexT row_start = A_rows[0];
  for (std::size_t row = 0; row < N; ++row)
  {
    IndexT row_stop  = A_rows[row+1];

    for (IndexT nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
      B[A_cols[nnz_index]][row] = A_values[nnz_index];

    row_start = row_stop;
  }

  benchmark::DoNotOptimize(B);
}


//
// CSR to CSR
//
template<typename IndexT, typename NumericT>
void run(std::vector<IndexT> const & A_rows, std::vector<IndexT> const & A_cols, std::vector<NumericT> const & A_values,
         std::vector<IndexT>       & B_rows, std::vector<IndexT>       & B_cols, std::vector<NumericT>       & B_values)
{
  B_cols.resize(A_cols.size());
  B_values.resize(A_values.size()); // note: initialization with zero not strictly required
  B_rows.resize(A_rows.size());     // note: initialization with zero not strictly required

  std::size_t N = A_rows.size() - 1;

  //
  // Stage 1: Compute pattern for B
  //
  IndexT row_start = 0;
  for (std::size_t row = 0; row < N; ++row)
  {
    IndexT row_stop  = A_rows[row+1];

    for (IndexT nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
      B_rows[A_cols[nnz_index]] += 1;

    row_start = row_stop;
  }

  // Bring row-start array in place using exclusive-scan:
  IndexT offset = 0;
  for (std::size_t row = 0; row < N; ++row)
  {
    IndexT tmp = B_rows[row];
    B_rows[row] = offset;
    offset += tmp;
  }
  B_rows[N] = offset;


  //
  // Stage 2: Fill with data
  //
  std::vector<IndexT> B_offsets(B_rows); // index of first unwritten element per row

  row_start = A_rows[0];
  for (std::size_t row = 0; row < N; ++row)
  {
    IndexT row_stop  = A_rows[row+1];

    for (IndexT nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      IndexT col_in_A = A_cols[nnz_index];
      IndexT B_nnz_index = B_offsets[col_in_A];
      B_cols[B_nnz_index] = row;
      B_values[B_nnz_index] = A_values[nnz_index];
      B_offsets[col_in_A] += 1;
    }

    row_start = row_stop;
  }

  benchmark::DoNotOptimize(B_rows);
  benchmark::DoNotOptimize(B_cols);
  benchmark::DoNotOptimize(B_values);
}

template<typename IndexT, typename NumericT>
void run_hpx(std::vector<IndexT> const & A_rows, std::vector<IndexT>         A_cols, std::vector<NumericT>         A_values,
             std::vector<IndexT>       & B_rows, std::vector<IndexT>       & B_cols, std::vector<NumericT>       & B_values)
{
  using namespace hpx::parallel;
  using namespace hpx::util;

  // create a vector holding the current row indices (future column indices) of each value
  std::vector<IndexT> row_ind; row_ind.resize(A_cols.size());
  for_loop(execution::par_unseq, 0, A_rows.size()-1,
           [&](IndexT i) {
             // fill the row indices for this row
             std::fill(row_ind.begin() + A_rows[i],
                       row_ind.begin() + A_rows[i+1],
                       i);
           });

  // sort a zip of (old row indices, old column indices, values)
  // by (old column indices, old row indices) to produce a column major order

  auto col_major_start = make_zip_iterator(A_cols.begin(), row_ind.begin(), A_values.begin());
  auto col_major_stop  = make_zip_iterator(A_cols.end(), row_ind.end(), A_values.end());

  // stable_sort doesn't seem to work with zip iterators in HPX, so no point investigating
  sort(execution::par_unseq, col_major_start, col_major_stop,
       [](auto a, auto b) {
           return std::tie(get<0>(a), get<1>(a)) < std::tie(get<0>(b), get<1>(b));
       });

  // swap the sorted row indices into place as the new columns
  B_cols = std::move(row_ind);    // row_ind is broken up so it indicates the new columns
  std::swap(B_values, A_values);  // A_values is now in the right order

  // scan the new row indices (the newly sorted A_cols) to locate row boundaries
  B_rows.resize(A_rows.size());         // assuming square matrix
  for_loop(execution::par_unseq, 0, B_rows.size(),
           [&](IndexT row) {
             auto it = std::lower_bound(A_cols.begin(), A_cols.end(), row);
             B_rows[row] = std::distance(A_cols.begin(), it);
           });

  benchmark::DoNotOptimize(B_rows);
  benchmark::DoNotOptimize(B_cols);
  benchmark::DoNotOptimize(B_values);
}
