#include <benchmark/benchmark.h>

#include <vector>
#include <map>

#include <hpx/hpx.hpp>
#include <hpx/include/parallel_numeric.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/parallel/algorithms/merge.hpp>
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

//
// Recursive Merge Approach
//

// When ordered by rows then by columns, the column and value arrays may be regarded
// as composed of subsequences, one per row, sorted the other way (column, then row),
// with the row being the same throughout the subsequence.  When viewed this way the col/value
// arrays appear like a partially completed merge sort - all that remains is to "merge"
// the subsequences (the rows) by column, stably.  This approach implements a stable
// merge of the rows in a binary tree constructed recursively.

// The primary engine of this work is the HPX "merge" algorithm.  The tricky part is
// it is not an in-place merge - a destination range must be supplied.  To avoid
// excessive allocations, we arrange the computation so that only one temporary
// sequence is required.  The original and temporary sequences are alternately used
// as the merge proceeds up the tree to the final result

// forward declare "regular" (destination supplied) version
template<typename RowIter, typename ColIter>
hpx::future<void>
merge_rows_by_col_to(RowIter row_idx_start,
                     RowIter row_idx_stop,
                     ColIter col_start,
                     ColIter dest_start);

// the "in-place" variant recursively calls the non-in-place variant, then uses merge with destination
// set to the original sequence

template<typename RowIter, typename ColIter>
hpx::future<void>
merge_rows_by_col_inplace(RowIter row_idx_start, // subrange of the row indices
                          RowIter row_idx_stop,
                          ColIter col_start,     // input - zipped {col,row,value}
                          ColIter tmp_start) {   // working space (same size)

  using namespace hpx::parallel;
  using namespace hpx::util;

  // the columns of interest begin at offset *row_idx_start and run
  // to right before *(row_idx_stop-1)
  if (std::distance(row_idx_start, row_idx_stop) < 3) {
    // nothing to do - one or fewer rows
    return hpx::future<void>();
  }

  // otherwise divide into roughly equal sets of rows and recurse
  RowIter row_idx_mid = row_idx_start + std::distance(row_idx_start, row_idx_stop) / 2;

  // recursively merge into two halves of tmp array
  // note deliberate overlap: delimiting rows N through N+M requires
  // M+1 row indices, plus end iterator
  return hpx::when_all(merge_rows_by_col_to(row_idx_start, row_idx_mid + 1, col_start, tmp_start),
                       merge_rows_by_col_to(row_idx_mid, row_idx_stop, col_start, tmp_start))
    .then([=](auto const&) {
        // merge those halves back into the original

        // note: using sequential execution policy here
        // there seems to be a bug in merge:
        // https://github.com/STEllAR-GROUP/hpx/issues/2964
        merge(execution::seq,
              tmp_start + *row_idx_start,
              tmp_start + *row_idx_mid,
              tmp_start + *row_idx_mid,
              tmp_start + *(row_idx_stop-1),
              col_start + *row_idx_start,
              [](auto const & a, auto const & b) {
                // hpx merge is stable both within ranges and between them
                // which means we need only look at the column number
                // (we are already in row order)
                return get<0>(a) < get<0>(b);
              });

       });

}

// the non "in-place" variant recursively calls the in-place variant, then uses merge with
// destination set to the supplied work sequence

template<typename RowIter, typename ColIter>
hpx::future<void>
merge_rows_by_col_to(RowIter row_idx_start,
                     RowIter row_idx_stop,
                     ColIter col_start,
                     ColIter dest_start) {

  using namespace hpx::parallel;
  using namespace hpx::util;

  // the columns of interest begin at offset *row_idx_start and run
  // to right before *(row_idx_stop-1)
  if (std::distance(row_idx_start, row_idx_stop) < 2) {
    // nothing to do - zero rows
    return hpx::future<void>();
  }

  if (std::distance(row_idx_start, row_idx_stop) == 2) {
    // no merging to do, but we still need to copy the sole row to the destination
    return copy(execution::par(execution::task),
                col_start + *row_idx_start,
                col_start + *(row_idx_stop-1),
                dest_start + *row_idx_start);
  }

  // otherwise divide into roughly equal sets of rows and recurse
  RowIter row_idx_mid = row_idx_start + std::distance(row_idx_start, row_idx_stop) / 2;

  // recursively merge the two halves of input array using dest as scratch storage
  return hpx::when_all(merge_rows_by_col_inplace(row_idx_start, row_idx_mid + 1, col_start, dest_start),
                       merge_rows_by_col_inplace(row_idx_mid, row_idx_stop, col_start, dest_start))
    .then([=](auto const &) {
        // merge the halves into the destination

        merge(execution::seq,
              col_start + *row_idx_start,
              col_start + *row_idx_mid,
              col_start + *row_idx_mid,
              col_start + *(row_idx_stop-1),
              dest_start + *row_idx_start,
              [](auto const & a, auto const & b) {
                return get<0>(a) < get<0>(b);
              });

      });

}

template<typename IndexT, typename NumericT>
void run_hpx_merge(std::vector<IndexT> const & A_rows, std::vector<IndexT>         A_cols, std::vector<NumericT>         A_values,
                   std::vector<IndexT>       & B_rows, std::vector<IndexT>       & B_cols, std::vector<NumericT>       & B_values)
{
  using namespace hpx::parallel;
  using namespace hpx::util;

  // create a vector holding the current row indices (future column indices) of each value
  std::vector<IndexT> row_ind(A_cols.size());
  auto fill_result = for_loop(execution::par(execution::task), 0, A_rows.size()-1,
                              [&row_ind, &A_rows](IndexT i) {
                                // fill the row indices for this row
                                std::fill(row_ind.begin() + A_rows[i],
                                          row_ind.begin() + A_rows[i+1],
                                          i);
                              });

  // zip up (old row indices, old column indices, values) for simultaneous handling
  auto col_major_start = make_zip_iterator(A_cols.begin(), row_ind.begin(), A_values.begin());

  // allocate temporary working storage
  const std::size_t entry_count = A_cols.size();
  std::vector<IndexT>   tmp_cols(entry_count);
  std::vector<IndexT>   tmp_rows(entry_count);
  std::vector<NumericT> tmp_values(entry_count);

  // apply recursive in-place merge via futures
  auto merge_result = fill_result.then(
    [&](auto const&) {
      // because each row is internally already sorted we can merge by row instead of sorting
      merge_rows_by_col_inplace(
        A_rows.begin(), A_rows.end(),
        col_major_start,
        hpx::util::make_zip_iterator(   // supply working space
          tmp_cols.begin(),
          tmp_rows.begin(),
          tmp_values.begin())).get();
    });

  // Finally, produce the new row boundaries from the newly sorted rows (former columns)
  B_rows.resize(A_rows.size());         // assuming square matrix
  merge_result.then(
    [&](auto const&) {
      for_loop(execution::par, 0, B_rows.size(),
               [&A_cols, &B_rows](IndexT row) {
                 auto it = std::lower_bound(A_cols.begin(), A_cols.end(), row);
                 B_rows[row] = std::distance(A_cols.begin(), it);
               });
    }).get();

  // the sorted row indices are the new columns
  B_cols = std::move(row_ind);
  B_values = std::move(A_values);

  benchmark::DoNotOptimize(B_rows);
  benchmark::DoNotOptimize(B_cols);
  benchmark::DoNotOptimize(B_values);
}
