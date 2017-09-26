#include <benchmark/benchmark.h>

#include <vector>
#include <queue>
#include <map>
#include <numeric>

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
  B_rows = std::vector<IndexT>(A_rows.size(), 0);

  std::size_t N = A_rows.size() - 1;

  //
  // Stage 1: Compute pattern for B
  //
  std::for_each(A_cols.begin(), A_cols.end(),
                [&B_rows](IndexT col) { ++B_rows[col+1]; });

  // Bring row-start array in place using exclusive-scan:
  std::partial_sum(B_rows.begin(), B_rows.end(), B_rows.begin());

  //
  // Stage 2: Fill with data
  //
  std::vector<IndexT> B_offsets(B_rows); // index of first unwritten element per row

  IndexT row_start = A_rows[0];
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

//
// CSR to CSR via heap based multi-range-merge alg
//

template<typename IndexT, typename NumericT>
void run_heap(std::vector<IndexT> const & A_rows, std::vector<IndexT> const & A_cols, std::vector<NumericT> const & A_values,
              std::vector<IndexT>       & B_rows, std::vector<IndexT>       & B_cols, std::vector<NumericT>       & B_values)
{
  B_cols.resize(A_cols.size());
  B_values.resize(A_values.size()); // note: initialization with zero not strictly required
  B_rows = std::vector<IndexT>(A_rows.size(), 0);

  std::size_t N = A_rows.size() - 1;

  //
  // Stage 1: Compute pattern for B
  //
  std::for_each(A_cols.begin(), A_cols.end(),
                [&B_rows](IndexT col) { ++B_rows[col+1]; });

  // Bring row-start array in place using exclusive-scan:
  std::partial_sum(B_rows.begin(), B_rows.end(), B_rows.begin());

  //
  // Stage 2: Create min heap for front-of-row data
  //

  using Triplet = std::pair<IndexT,    // column *index*, i.e. offset within A_cols
                            IndexT>;   // row number
  auto compare = [&A_cols](Triplet const & a, Triplet const & b) {
      // reverse compare for min heap
      // by column first, then row
      return std::tie(A_cols[std::get<0>(a)], std::get<1>(a)) > std::tie(A_cols[std::get<0>(b)], std::get<1>(b));
  };
  std::priority_queue<Triplet, std::vector<Triplet>, decltype(compare)> row_start_heap(compare);
  for(auto row_it = A_rows.begin(); row_it != A_rows.end()-1; ++row_it) {
      if (*row_it != *(row_it+1)) {
          // non-empty row; insert the first triplet
          row_start_heap.emplace(*row_it,
                                 std::distance(A_rows.begin(), row_it));
      }
  }

  //
  // Stage 3: Fill with data
  //

  std::size_t B_idx = 0;
  for (std::size_t B_idx = 0; !row_start_heap.empty(); ++B_idx) {
      // insert top of heap into output arrays
      Triplet const & min_elt = row_start_heap.top();

      B_values[B_idx] = A_values[std::get<0>(min_elt)];
      IndexT old_row  = std::get<1>(min_elt);
      B_cols[B_idx]   = old_row;    // the *row* becomes the column

      // update heap
      // we *always* remove the smallest value from the heap.
      // We *sometimes* (when there are remaining columns in the row) replace it with a new value
      row_start_heap.pop();
      std::size_t col_idx = std::get<0>(min_elt) + 1;
      if (col_idx < A_rows[old_row+1]) {
          // there is still data in the row; add next column to the heap
          row_start_heap.emplace(col_idx, old_row);
      }
  }


  benchmark::DoNotOptimize(B_rows);
  benchmark::DoNotOptimize(B_cols);
  benchmark::DoNotOptimize(B_values);
}
